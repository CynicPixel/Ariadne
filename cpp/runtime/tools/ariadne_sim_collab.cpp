// ---------------------------------------------------------------------------
// ariadne_sim_collab: 1.6 placeholder collaboration over a sim recording (design
// COLLAB_1.6): both agents run the 1.4/1.5 VO in replay; agent 1 sends a global
// descriptor per keyframe to the broker (agent 0, the lowest AgentId, §15.4);
// the broker retrieves candidates, requests local features, verifies with
// findEssentialMat + PnP, emits SE(3) POSE_EDGEs, and merges agent 1 into its
// frame with the best edge. EVERY inter-agent byte goes through the real
// serializer (WireCodec, ByteLedger: §15.7 counted from day one) and the
// idempotent inbox (§15.3).
//
//   ariadne_sim_collab --rec RECDIR --scene SCENEDIR --perception-config YAML
//       --slam-config YAML --collab-config YAML --out DIR [--inject-faults 1]
//
// --inject-faults 1 duplicates every 5th message and holds every 3rd one back
// until the NEXT flush (delivered after the newer ones), so same-class sequence
// gaps really occur and the inbox must buffer. Idempotency + order-independent
// outputs mean every result except the protocol counters is unchanged (the 1.6
// protocol check; Phase 1 exit gate); the run fails if no fault was exercised.
// ---------------------------------------------------------------------------
#include "collaboration/broker/edge_verifier.hpp"
#include "collaboration/protocol/inbox.hpp"
#include "collaboration/retrieval/mean_bit_descriptor.hpp"
#include "reconstruction/spool/keyframe_spool.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace ariadne;
using collaboration::KeyframeFeatures;

namespace {

struct CollabConfig {
    std::string name;
    int orb_features = 1000;
    double retrieval_threshold = 0.5;
    int max_candidates = 20;
    collaboration::VerifyParams verify;
    double gt_max_distance_m = 1.5, gt_max_angle_deg = 30.0;   // a TRUE opportunity (worldgen closure gate)
    double edge_correct_t_m = 0.05, edge_correct_r_deg = 1.0;   // evaluation: edge vs GT
    int negative_probe = 20;   // evaluation: hard negatives fed to the verifier outside the exchange
};

CollabConfig LoadCollab(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    CollabConfig c;
    c.name = n["name"].as<std::string>();
    c.orb_features = n["orb_features"].as<int>();
    c.retrieval_threshold = n["retrieval_threshold"].as<double>();
    c.max_candidates = n["max_candidates"].as<int>();
    const auto v = n["verify"];
    c.verify.ratio = v["ratio"].as<double>();
    c.verify.min_matches = v["min_matches"].as<int>();
    c.verify.essential_threshold_px = v["essential_threshold_px"].as<double>();
    c.verify.min_essential_inliers = v["min_essential_inliers"].as<int>();
    c.verify.min_essential_inlier_ratio = v["min_essential_inlier_ratio"].as<double>();
    c.verify.pnp_reprojection_px = v["pnp_reprojection_px"].as<double>();
    c.verify.min_pnp_inliers = v["min_pnp_inliers"].as<int>();
    c.gt_max_distance_m = n["gt_max_distance_m"].as<double>();
    c.gt_max_angle_deg = n["gt_max_angle_deg"].as<double>();
    c.edge_correct_t_m = n["edge_correct_t_m"].as<double>();
    c.edge_correct_r_deg = n["edge_correct_r_deg"].as<double>();
    c.negative_probe = n["negative_probe"].as<int>();
    return c;
}

struct AgentRun {
    std::vector<KeyframeFeatures> kfs;
    std::vector<runtime::TimedPose> est_body;   // T_O_B per frame
    std::map<int64_t, core::SE3> gt;            // T_W_B
    core::SE3 T_B_C;                            // body <- rectified left
};

double AngleDeg(const core::Mat3& R) {
    return std::acos(std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
}

core::PoseValue ToPoseValue(const core::SE3& T) {
    const Eigen::Quaterniond q(T.R);
    return {float(q.w()), float(q.x()), float(q.y()), float(q.z()), float(T.t.x()), float(T.t.y()), float(T.t.z())};
}

/// ledger.csv (design CONSOLE_1.10): one row per message the ByteLedger counted, with the wire stamp
/// (a keyframe time, NOT causal) and the causal send time the console replays on.
struct LedgerLog {
    std::ofstream csv;
    uint64_t seq = 0;
};

/// contracts.yaml (Phase 1 exit gate, design PHASE1_GATE): every §4/§18.2 record counted at each hop it
/// crosses in this run. Counted, never asserted from the code's shape: a zero means the hop did not happen.
struct ContractCounts {
    uint64_t frames[2] = {0, 0};                 // §4.1 StereoFrame delivered to the VO (via on_frame)
    uint64_t keyframes[2] = {0, 0};              // §4.2 Keyframe records received by the exchange
    uint64_t keyframes_complete[2] = {0, 0};     // ... with every Phase-1 field filled (see Missing)
    // Per field, how many records lack it. pose_cov is zero BY DEFINITION on the gauge keyframe (id 0: the
    // absolute covariance is relative to it), so that one is counted separately, not as missing.
    std::map<std::string, uint64_t> missing[2];
    uint64_t gauge_zero_cov[2] = {0, 0};
    uint64_t bundles[2] = {0, 0};                // §18.2 KeyframeBundle records built
    uint64_t bundles_spooled[2] = {0, 0};        // ... written across the C++ -> Python boundary (after stride)
    uint64_t edges_encoded = 0;                  // §4.3 LoopEdgeValue serialized (POSE_EDGE)
    uint64_t edges_decoded_identical = 0;        // ... decoded by agent 1, every field equal to what was sent
    bool merge_edge_received = false;            // the merge's edge is one agent 1 actually received
};

bool SameEdge(const core::LoopEdgeValue& a, const core::LoopEdgeValue& b) {
    const auto pv = [](const core::PoseValue& x, const core::PoseValue& y) {
        return x.qw == y.qw && x.qx == y.qx && x.qy == y.qy && x.qz == y.qz && x.tx == y.tx && x.ty == y.ty && x.tz == y.tz;
    };
    return a.agent_a == b.agent_a && a.keyframe_a == b.keyframe_a && a.agent_b == b.agent_b &&
           a.keyframe_b == b.keyframe_b && pv(a.measurement, b.measurement) && a.information == b.information &&
           a.frame_a.agent == b.frame_a.agent && a.frame_a.epoch == b.frame_a.epoch &&
           a.frame_b.agent == b.frame_b.agent && a.frame_b.epoch == b.frame_b.epoch && a.status == b.status &&
           a.inlier_count == b.inlier_count && a.retrieval_score == b.retrieval_score && a.pcm_score == b.pcm_score &&
           a.gnc_weight == b.gnc_weight && a.degeneracy_flag == b.degeneracy_flag;
}

/// The simulated link: serialize, count, optionally misbehave, decode, inbox.
class Link {
public:
    Link(const core::WireCodec& codec, core::ByteLedger& ledger, bool faults, LedgerLog& log)
        : codec_(codec), ledger_(ledger), faults_(faults), log_(log) {}
    void Send(const core::SerializedMessage& m, int64_t causal_ns) {
        {
            core::ByteRecord r = m.record;
            ledger_.Record(r);
            log_.csv << log_.seq++ << "," << wire::EnumNameMessageClass(r.type) << "," << r.sender << "," << r.receiver
                     << "," << r.payload_bytes << "," << r.transmitted_bytes << "," << r.wire_bytes << ","
                     << r.compressed << "," << r.t.ns << "," << causal_ns << "\n";
        }
        q_.push_back(m.bytes);
        if (faults_ && ++sent_ % 5 == 0) q_.push_back(m.bytes);   // duplicate
    }
    /// Deliver what is queued to `inbox`; returns the in-order deliveries. With
    /// faults, every 3rd fresh message is held back and arrives AFTER the next
    /// flush's messages; `drain` delivers everything (end of the exchange).
    std::vector<core::DecodedMessage> Flush(collaboration::Inbox& inbox, bool drain = false) {
        std::deque<std::vector<uint8_t>> deliver;
        std::deque<std::vector<uint8_t>> late = std::move(held_);
        held_.clear();
        for (auto& w : q_) {
            if (faults_ && !drain && ++flushed_ % 3 == 0) {
                held_.push_back(std::move(w));
            } else {
                deliver.push_back(std::move(w));
            }
        }
        for (auto& w : late) deliver.push_back(std::move(w));   // older messages arrive last
        if (drain) {
            for (auto& w : held_) deliver.push_back(std::move(w));
            held_.clear();
        }
        q_.clear();
        std::vector<core::DecodedMessage> out;
        for (const auto& w : deliver) {
            core::DecodedMessage d;
            const auto s = codec_.Decode(w, &d);
            if (s != core::WireCodec::Status::kOk) {
                throw std::runtime_error(std::string("link: decode failed: ") + core::WireCodec::StatusName(s));
            }
            for (auto& m : inbox.Offer(std::move(d))) out.push_back(std::move(m));
        }
        return out;
    }
    bool Idle() const { return q_.empty() && held_.empty(); }

private:
    const core::WireCodec& codec_;
    core::ByteLedger& ledger_;
    bool faults_;
    LedgerLog& log_;
    uint64_t sent_ = 0, flushed_ = 0;
    std::deque<std::vector<uint8_t>> q_, held_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        tools::StereoVoArgs va;
        std::string rec, scene, collab_path, out;
        bool faults = false;
        std::string spool_dir, recon_path;   // 1.7: optional reconstruction spool (design RECON_1.7)
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
            const std::string v = argv[++i];
            if (k == "--rec") rec = v;
            else if (k == "--scene") scene = v;
            else if (k == "--perception-config") va.perception_config = v;
            else if (k == "--slam-config") va.slam_config = v;
            else if (k == "--collab-config") collab_path = v;
            else if (k == "--out") out = v;
            else if (k == "--seed") va.seed = std::stoull(v);
            else if (k == "--inject-faults") faults = v == "1";
            else if (k == "--spool") spool_dir = v;
            else if (k == "--recon-config") recon_path = v;
            else throw std::invalid_argument("unknown argument " + k);
        }
        if (rec.empty() || scene.empty() || collab_path.empty() || out.empty() || va.perception_config.empty()) {
            throw std::invalid_argument("usage: ariadne_sim_collab --rec R --scene S --perception-config P "
                                        "--slam-config S --collab-config C --out DIR [--inject-faults 1] "
                                        "[--spool DIR --recon-config Y]");
        }
        if (spool_dir.empty() != recon_path.empty()) throw std::invalid_argument("--spool needs --recon-config");
        const CollabConfig cc = LoadCollab(collab_path);
        const std::string scene_id = YAML::LoadFile(scene + "/scene.json")["scene_id"].as<std::string>();
        const fs::path run_dir = fs::path(out) / (scene_id + (faults ? "-faults-" : "-") + tools::UtcNow("%Y%m%dT%H%M%SZ"));
        fs::create_directories(run_dir / "raw");
        std::unique_ptr<reconstruction::KeyframeSpool> spool;
        if (!spool_dir.empty()) {
            const YAML::Node sp = YAML::LoadFile(recon_path)["spool"];
            reconstruction::SpoolParams p;
            p.scale = sp["scale"].as<double>();
            p.keyframe_stride = sp["keyframe_stride"].as<int>();
            p.heldout_stride = sp["heldout_stride"].as<int>();
            spool = std::make_unique<reconstruction::KeyframeSpool>(fs::path(spool_dir) / run_dir.filename(), p);
        }

        // --- both agents: VO in replay; the §4.2 / §18.2 records arrive through the hook ----
        AgentRun runs[2];
        ContractCounts cc_count;
        const cv::Ptr<cv::ORB> orb = cv::ORB::create(cc.orb_features);
        for (uint32_t k = 0; k < 2; ++k) {
            runtime::SimSource src(rec, scene, int(k));
            for (const auto& g : src.GroundTruth()) runs[k].gt[g.t_ns] = g.T;
            tools::StereoVoArgs a = va;
            a.out = (run_dir / "raw").string();
            a.tool = "ariadne_sim_collab/agent" + std::to_string(k);
            a.on_body_pose = [&, k](int64_t t, const core::SE3& T) { runs[k].est_body.push_back({t, T}); };
            a.on_frame = [&, k](const tools::FrameView& fv) {
                ++cc_count.frames[k];
                if (!spool || fv.is_keyframe || !spool->WantHeldout(fv.frame_id) || runs[k].kfs.empty()) return;
                const KeyframeFeatures& any = runs[k].kfs.front();   // intrinsics are fixed per rig
                const cv::Mat rgb = fv.left_rect_rgb();
                spool->AddHeldout(k, fv.frame_id, fv.t_ns, fv.T_O_C,
                                  spool->Camera(any.camera[0], any.camera[1], any.camera[2], any.camera[3],
                                                rgb.cols, rgb.rows),
                                  rgb);
            };
            a.on_keyframe = [&, k](const tools::KeyframeView& kv) {
                const runtime::KeyframeBundle& b = kv.bundle;
                if (kv.keyframe.agent != k || b.agent != k || b.kf_id != kv.keyframe.id) {
                    throw std::logic_error("collab: keyframe records disagree on identity");
                }
                ++cc_count.keyframes[k];
                ++cc_count.bundles[k];
                // The 1.6 consumer completes the §4.2 record with what it extracts: ORB keypoints +
                // descriptors and the mean-bit global descriptor; the wire features derive from it.
                runtime::Keyframe kf = kv.keyframe;
                orb->detectAndCompute(kv.left_rect, cv::noArray(), kf.keypoints, kf.descriptors);
                const cv::Mat z = runtime::MaskedDepth32(b.depth, b.mask);
                KeyframeFeatures f;
                f.agent = k;
                f.id = kf.frame_id;
                f.t_ns = kf.t_ns;
                f.T_O_C = kf.T_world_cam;
                f.camera = {float(b.K.fx), float(b.K.fy), float(b.K.cx), float(b.K.cy), float(kv.baseline_m)};
                f.descriptors.assign(kf.descriptors.data, kf.descriptors.data + kf.descriptors.total());
                for (const auto& kp : kf.keypoints) {
                    f.keypoints.push_back(kp.pt.x);
                    f.keypoints.push_back(kp.pt.y);
                    const int x = cvRound(kp.pt.x), y = cvRound(kp.pt.y);
                    const float zz = z.at<float>(std::clamp(y, 0, z.rows - 1), std::clamp(x, 0, z.cols - 1));
                    f.points3d.push_back(std::isfinite(zz) ? float((kp.pt.x - b.K.cx) / b.K.fx * zz) : NAN);
                    f.points3d.push_back(std::isfinite(zz) ? float((kp.pt.y - b.K.cy) / b.K.fy * zz) : NAN);
                    f.points3d.push_back(zz);
                }
                kf.global_descriptor = collaboration::MeanBitDescriptor(f.descriptors);
                f.global = kf.global_descriptor;
                {
                    const bool gauge = kf.id == 0 && kf.pose_cov.isZero();
                    cc_count.gauge_zero_cov[k] += gauge;
                    const std::pair<const char*, bool> have[] = {
                        {"pose_cov", gauge || !kf.pose_cov.isZero()}, {"depth", !kf.depth.Z.empty()},
                        {"landmark_ids", !kf.landmark_ids.empty()}, {"keypoints", !kf.keypoints.empty()},
                        {"descriptors", !kf.descriptors.empty()}, {"global_descriptor", !kf.global_descriptor.empty()}};
                    bool all = true;
                    for (const auto& [name, ok] : have) {
                        cc_count.missing[k][name] += !ok;
                        all = all && ok;
                    }
                    cc_count.keyframes_complete[k] += all;
                }
                if (spool) {
                    const size_t before = spool->size();
                    spool->AddKeyframe(b.agent, kf.frame_id, kf.t_ns, b.frame_id_of_pose, b.T_WiC,
                                       spool->Camera(b.K.fx, b.K.fy, b.K.cx, b.K.cy, b.rgb.cols, b.rgb.rows),
                                       kv.baseline_m, b.rgb, z, runtime::MaskedDepth32(b.depth_sigma, b.mask));
                    cc_count.bundles_spooled[k] += spool->size() - before;
                }
                runs[k].T_B_C = kv.T_B_C;
                runs[k].kfs.push_back(std::move(f));
            };
            tools::RunStereoVo(src, a, argc, argv);
            std::printf("agent %u: %zu keyframes\n", k, runs[k].kfs.size());
        }

        // --- the exchange: agent 1 <-> broker (agent 0), all through the codec ----
        const core::WireCodec codec;
        core::ByteLedger ledger;
        collaboration::Inbox broker_in, agent1_in;
        collaboration::SequenceCounter seq0, seq1;
        LedgerLog ledger_log;
        ledger_log.csv.open(run_dir / "ledger.csv");
        ledger_log.csv << "seq,class,sender,receiver,payload_bytes,transmitted_bytes,wire_bytes,compressed,"
                          "wire_stamp_ns,causal_ns\n";
        Link to_broker(codec, ledger, faults, ledger_log), to_agent1(codec, ledger, faults, ledger_log);
        std::vector<int64_t> request_causal;   // by rank: later of the two keyframes (the 1.8 t_merge rule)
        using core::MessageClass;
        std::map<uint64_t, const KeyframeFeatures*> kf1;
        for (const auto& f : runs[1].kfs) kf1[f.id] = &f;

        for (const auto& f : runs[1].kfs) {
            core::GlobalDescriptorValue g{f.id, f.t_ns, f.global, float(f.keypoints.size() / 2)};
            to_broker.Send(codec.EncodeGlobalDescriptor(g, 1, 0, core::Timestamp{f.t_ns},
                                                        seq1.Next(0, MessageClass::GLOBAL_DESCRIPTOR)),
                           f.t_ns);
        }
        struct Candidate {
            double score;
            const KeyframeFeatures* a;
            uint64_t b;
        };
        // GT relative pose of two keyframes' rectified-left cameras, and the scene's
        // opportunity definition (worldgen closure gate: within 1.5 m / 30 deg).
        auto GtRel = [&](const KeyframeFeatures& a, const KeyframeFeatures& b) {
            const core::SE3 ca = runs[0].gt.at(a.t_ns) * runs[0].T_B_C, cb = runs[1].gt.at(b.t_ns) * runs[1].T_B_C;
            return ca.inverse() * cb;
        };
        auto ViewAngleDeg = [](const core::SE3& gt) {
            const core::Vec3 zb = gt.R * core::Vec3::UnitZ();
            return std::acos(std::clamp(core::Vec3::UnitZ().dot(zb), -1.0, 1.0)) * 180.0 / M_PI;
        };
        auto IsOpportunity = [&](const core::SE3& gt) {
            return gt.t.norm() <= cc.gt_max_distance_m && ViewAngleDeg(gt) <= cc.gt_max_angle_deg;
        };
        std::vector<Candidate> cands;
        std::vector<double> top1_scores;       // every query, before threshold and cap
        struct Top1 {
            double score;
            const KeyframeFeatures* a;
            const KeyframeFeatures* q;
            bool truth;
        };
        std::vector<Top1> top1s;
        int top1_true = 0, queries_with_opportunity = 0;
        for (const auto& m : to_broker.Flush(broker_in, /*drain=*/true)) {
            if (!m.has_global_descriptor) throw std::runtime_error("broker: unexpected message class");
            const KeyframeFeatures* best = nullptr;
            double bs = -2;
            for (const auto& a : runs[0].kfs) {
                const double s = collaboration::Cosine(a.global, m.global_descriptor.descriptor);
                if (s > bs) bs = s, best = &a;
            }
            if (!best) throw std::runtime_error("broker: agent 0 has no keyframes");
            // Retrieval quality over ALL queries (GT is evaluation-only, never fed back).
            const KeyframeFeatures& q = *kf1.at(m.global_descriptor.keyframe_id);
            top1_scores.push_back(bs);
            const bool t1 = IsOpportunity(GtRel(*best, q));
            top1_true += t1;
            top1s.push_back({bs, best, &q, t1});
            for (const auto& a : runs[0].kfs) {
                if (IsOpportunity(GtRel(a, q))) {
                    ++queries_with_opportunity;
                    break;
                }
            }
            if (bs >= cc.retrieval_threshold) cands.push_back({bs, best, m.global_descriptor.keyframe_id});
        }
        std::stable_sort(cands.begin(), cands.end(), [](const Candidate& x, const Candidate& y) { return x.score > y.score; });
        if (int(cands.size()) > cc.max_candidates) cands.resize(size_t(cc.max_candidates));

        // Per-candidate outcome, keyed by rank: every output below is built from this
        // map AFTER the exchange, so it cannot depend on message arrival order.
        struct Outcome {
            const KeyframeFeatures* b = nullptr;
            double gt_d = NAN, gt_a = NAN;
            bool truth = false;
            collaboration::VerifyResult v;
            double et = NAN, er = NAN;
        };
        std::map<uint64_t, Outcome> outcomes;
        size_t edges_received = 0;
        std::map<std::pair<uint64_t, uint64_t>, core::LoopEdgeValue> edges_out, edges_in;   // by (kf_a, kf_b)
        // Agent 1's receive side: every delivery is dispatched by payload; nothing is
        // silently dropped (a POSE_EDGE may arrive in the same flush as a request).
        auto agent1_receive = [&](const core::DecodedMessage& m) {
            if (m.has_match_request) {   // answer with this keyframe's features
                const KeyframeFeatures& f = *kf1.at(m.match_request.candidate_keyframe);
                core::FeatureResponseValue fr{m.match_request.request_id, f.id, f.keypoints, f.descriptors, f.points3d,
                                              f.camera};
                to_broker.Send(codec.EncodeFeatureResponse(fr, 1, 0, core::Timestamp{f.t_ns},
                                                           seq1.Next(0, MessageClass::MATCH_RESPONSE)),
                               request_causal.at(size_t(m.match_request.request_id)));
            } else if (m.has_edge) {
                ++edges_received;
                edges_in.emplace(std::make_pair(m.edge.keyframe_a, m.edge.keyframe_b), m.edge);
            } else {
                throw std::runtime_error("agent1: unexpected message class");
            }
        };
        // The broker's receive side: verify, record, and send the edge to agent 1.
        auto broker_receive = [&](const core::DecodedMessage& m) {
            if (!m.has_feature_response) throw std::runtime_error("broker: unexpected message class");
            const uint64_t rank = m.feature_response.request_id;
            if (rank >= cands.size() || outcomes.count(rank)) throw std::runtime_error("broker: bad request id");
            const Candidate& cm = cands[rank];
            Outcome o;
            o.b = kf1.at(m.feature_response.keyframe_id);
            const core::SE3 gt = GtRel(*cm.a, *o.b);
            o.gt_d = gt.t.norm();
            o.gt_a = ViewAngleDeg(gt);
            o.truth = IsOpportunity(gt);
            o.v = collaboration::VerifyEdge(*cm.a, m.feature_response, cc.verify, va.seed * 1000003ull + rank);
            if (o.v.ok) {
                o.et = (o.v.T_A_B.t - gt.t).norm();
                o.er = AngleDeg(o.v.T_A_B.R * gt.R.transpose());
                core::LoopEdgeValue e;
                e.agent_a = 0;
                e.keyframe_a = cm.a->id;
                e.agent_b = 1;
                e.keyframe_b = o.b->id;
                e.measurement = ToPoseValue(o.v.T_A_B);
                e.information = collaboration::ProxyInformation(o.v.pnp_inliers);
                e.frame_a = {0, 0};
                e.frame_b = {1, 0};
                e.status = wire::EdgeStatus::GEOMETRY_VERIFIED;
                e.inlier_count = uint32_t(o.v.pnp_inliers);
                e.retrieval_score = float(cm.score);
                edges_out.emplace(std::make_pair(e.keyframe_a, e.keyframe_b), e);
                ++cc_count.edges_encoded;
                to_agent1.Send(codec.EncodeLoopEdge(e, 0, 1, core::Timestamp{o.b->t_ns},
                                                    seq0.Next(1, MessageClass::POSE_EDGE)),
                               request_causal.at(size_t(rank)));
            }
            outcomes.emplace(rank, std::move(o));
        };
        for (const Candidate& c : cands) request_causal.push_back(std::max(c.a->t_ns, kf1.at(c.b)->t_ns));
        for (size_t r = 0; r < cands.size(); ++r) {
            const Candidate& c = cands[r];
            core::MatchRequestValue req{r, c.a->id, c.b, 0};
            to_agent1.Send(codec.EncodeMatchRequest(req, 0, 1, core::Timestamp{c.a->t_ns},
                                                    seq0.Next(1, MessageClass::MATCH_REQUEST)),
                           request_causal[r]);
            for (const auto& m : to_agent1.Flush(agent1_in)) agent1_receive(m);
            for (const auto& m : to_broker.Flush(broker_in)) broker_receive(m);
        }
        // Drain: deliver everything held back until both links are idle.
        for (int round = 0; !(to_agent1.Idle() && to_broker.Idle()); ++round) {
            if (round > 8) throw std::runtime_error("link: did not drain");
            for (const auto& m : to_agent1.Flush(agent1_in, true)) agent1_receive(m);
            for (const auto& m : to_broker.Flush(broker_in, true)) broker_receive(m);
        }
        if (outcomes.size() != cands.size()) throw std::runtime_error("broker: missing feature responses");
        if (broker_in.waiting() + agent1_in.waiting() != 0) throw std::runtime_error("link: messages left undelivered");
        if (faults && (broker_in.buffered() == 0 || agent1_in.buffered() == 0 || broker_in.duplicates() == 0 ||
                       agent1_in.duplicates() == 0)) {
            throw std::runtime_error("faults requested but not exercised on both links");
        }

        std::ofstream csv(run_dir / "candidates.csv");
        csv << "rank,score,kf_a,kf_b,gt_distance_m,gt_angle_deg,true_opportunity,result,matches,essential_inliers,"
               "pnp_inliers,edge_t_err_m,edge_r_err_deg,edge_correct,h_over_e,planarity\n";
        int n_true = 0, n_verified = 0, n_verified_true = 0, n_correct = 0;
        size_t edges_sent = 0;
        std::vector<std::pair<double, double>> edge_err;
        const KeyframeFeatures* merge_a = nullptr;   // the merge uses the best-RANKED verified edge
        const KeyframeFeatures* merge_b = nullptr;
        core::SE3 merge_T_A_B;
        for (const auto& [rank, o] : outcomes) {
            const Candidate& cm = cands[rank];
            const bool correct = o.v.ok && o.et <= cc.edge_correct_t_m && o.er <= cc.edge_correct_r_deg;
            n_true += o.truth;
            if (o.v.ok) {
                ++edges_sent;
                ++n_verified;
                n_verified_true += o.truth;
                n_correct += correct;
                edge_err.emplace_back(o.et, o.er);
                if (!merge_a) merge_a = cm.a, merge_b = o.b, merge_T_A_B = o.v.T_A_B;
            }
            csv << rank << "," << cm.score << "," << cm.a->id << "," << o.b->id << "," << o.gt_d << "," << o.gt_a << ","
                << o.truth << "," << o.v.reason << "," << o.v.matches << "," << o.v.essential_inliers << ","
                << o.v.pnp_inliers << "," << o.et << "," << o.er << "," << correct << "," << o.v.h_over_e << ","
                << o.v.planarity << "\n";
        }
        csv.close();
        if (edges_received != edges_sent) {
            throw std::runtime_error("link: " + std::to_string(edges_sent) + " edges sent, " +
                                     std::to_string(edges_received) + " received");
        }
        // §4.3 end to end: every edge agent 1 decoded is field-for-field the edge the broker encoded.
        if (edges_in.size() != edges_out.size()) throw std::runtime_error("link: distinct edges sent != received");
        for (const auto& [key, e] : edges_out) {
            const auto it = edges_in.find(key);
            if (it == edges_in.end() || !SameEdge(e, it->second)) {
                throw std::runtime_error("link: POSE_EDGE " + std::to_string(key.first) + "<-" +
                                         std::to_string(key.second) + " decoded differently from what was sent");
            }
            ++cc_count.edges_decoded_identical;
        }
        cc_count.merge_edge_received = merge_a && edges_in.count({merge_a->id, merge_b->id});

        // --- negative probe (evaluation only: no wire, no ledger, no edge, no merge) ---
        // Every exchanged pair was a real overlap, so the exchange alone never shows the
        // gate rejecting anything. Feed the verifier the highest-scoring top-1 pairs that
        // are NOT opportunities (appearance-similar, geometrically wrong: the hard case).
        // A false accept is a verified edge that is not correct vs GT (the cone does not
        // judge an edge: some out-of-cone pairs still overlap and verify correctly).
        std::vector<Top1> negs;
        for (const auto& t : top1s) {
            if (!t.truth) negs.push_back(t);
        }
        std::stable_sort(negs.begin(), negs.end(), [](const Top1& x, const Top1& y) { return x.score > y.score; });
        if (int(negs.size()) > cc.negative_probe) negs.resize(size_t(cc.negative_probe));
        int probe_rejected = 0, probe_correct = 0, probe_false = 0;
        double probe_false_t_max = 0;
        std::ofstream ncsv(run_dir / "negative_probe.csv");
        ncsv << "index,score,kf_a,kf_b,gt_distance_m,gt_angle_deg,result,matches,essential_inliers,pnp_inliers,"
                "edge_t_err_m,edge_r_err_deg,edge_correct,h_over_e,planarity\n";
        for (size_t i = 0; i < negs.size(); ++i) {
            const Top1& t = negs[i];
            const core::FeatureResponseValue fr{i, t.q->id, t.q->keypoints, t.q->descriptors, t.q->points3d, t.q->camera};
            const auto v = collaboration::VerifyEdge(*t.a, fr, cc.verify, va.seed * 1000003ull + 1000000ull + i);
            const core::SE3 gt = GtRel(*t.a, *t.q);
            double et = NAN, er = NAN;
            bool correct = false;
            if (!v.ok) {
                ++probe_rejected;
            } else {
                et = (v.T_A_B.t - gt.t).norm();
                er = AngleDeg(v.T_A_B.R * gt.R.transpose());
                correct = et <= cc.edge_correct_t_m && er <= cc.edge_correct_r_deg;
                if (correct) {
                    ++probe_correct;
                } else {
                    ++probe_false;
                    probe_false_t_max = std::max(probe_false_t_max, et);
                }
            }
            ncsv << i << "," << t.score << "," << t.a->id << "," << t.q->id << "," << gt.t.norm() << ","
                 << ViewAngleDeg(gt) << "," << v.reason << "," << v.matches << "," << v.essential_inliers << ","
                 << v.pnp_inliers << "," << et << "," << er << "," << correct << "," << v.h_over_e << "," << v.planarity
                 << "\n";
        }
        ncsv.close();

        // --- merge with the best verified edge; score both agents in one frame ------
        std::vector<core::TrajectorySample> e_all, g_all;
        double merged_rmse = NAN;
        core::SE3 T_O0_O1;
        if (merge_a) {
            T_O0_O1 = merge_a->T_O_C * merge_T_A_B * merge_b->T_O_C.inverse();
            for (uint32_t k = 0; k < 2; ++k) {
                for (const auto& p : runs[k].est_body) {
                    const auto it = runs[k].gt.find(p.t_ns);
                    if (it == runs[k].gt.end()) continue;
                    e_all.push_back({core::Timestamp{p.t_ns}, k, k == 0 ? p.T : T_O0_O1 * p.T});
                    g_all.push_back({core::Timestamp{p.t_ns}, k, it->second});
                }
            }
            merged_rmse = runtime::ComputeATE(e_all, g_all).rmse;
        }
        if (spool) {
            const std::string edge = merge_a ? "agent0 kf " + std::to_string(merge_a->id) + " <- agent1 kf " +
                                                   std::to_string(merge_b->id) + " (best-ranked verified, 1.6)"
                                             : "none";
            spool->Finish(run_dir.filename().string(), scene_id, merge_a != nullptr, T_O0_O1, edge);
            // Evaluation-only record (NOT in the spool): the one SE(3) alignment of the merged
            // trajectory to GT, i.e. T_W_O0, so the fused cloud can be compared with GT meshes.
            const auto al = runtime::AlignTrajectories(e_all, g_all, false);
            if (!al.valid) throw std::runtime_error("spool: no valid GT alignment for the evaluation record");
            YAML::Emitter y;
            y.SetDoublePrecision(17);
            std::vector<double> T;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) T.push_back(al.transform.R(i, j));
                T.push_back(al.transform.t(i));
            }
            y << YAML::BeginMap << YAML::Key << "T_W_O0" << YAML::Value << YAML::Flow << T << YAML::Key << "source"
              << YAML::Value << "Umeyama SE(3) of the merged two-agent trajectory onto GT (evaluation only)"
              << YAML::Key << "spool" << YAML::Value << (fs::path(spool_dir) / run_dir.filename()).string()
              << YAML::Key << "spool_entries" << YAML::Value << spool->size() << YAML::EndMap;
            std::ofstream(run_dir / "recon_eval.yaml") << "# Evaluation-only record for 1.7. Generated.\n" << y.c_str()
                                                       << "\n";
        }
        double et_max = 0, er_max = 0;
        for (const auto& [t, r] : edge_err) et_max = std::max(et_max, t), er_max = std::max(er_max, r);

        std::vector<double> sorted_scores = top1_scores;
        if (sorted_scores.empty()) throw std::runtime_error("broker: no global descriptors received");
        std::sort(sorted_scores.begin(), sorted_scores.end());
        std::ofstream res(run_dir / "results.yaml");
        res << "# ariadne_sim_collab results (design COLLAB_1.6). Generated; do not hand-edit.\n"
            << "scene_id: " << scene_id << "\ncollab_config: " << cc.name << "\ninject_faults: " << faults << "\n"
            << "keyframes: {agent0: " << runs[0].kfs.size() << ", agent1: " << runs[1].kfs.size() << "}\n"
            << "retrieval: {candidates: " << cands.size() << ", true_opportunities: " << n_true
            << ", verified: " << n_verified << ", verified_true: " << n_verified_true << "}\n"
            << "retrieval_all_queries: {queries: " << top1_scores.size()
            << ", with_gt_opportunity: " << queries_with_opportunity << ", top1_true: " << top1_true
            << ", top1_score_min: " << sorted_scores.front() << ", top1_score_median: "
            << sorted_scores[sorted_scores.size() / 2] << ", top1_score_max: " << sorted_scores.back() << "}\n"
            << "negative_probe: {probed: " << negs.size() << ", rejected: " << probe_rejected
            << ", accepted_correct: " << probe_correct << ", false_accepts: " << probe_false
            << ", max_false_t_err_m: " << probe_false_t_max << "}\n"
            << "edges: {sent: " << edges_sent << ", received_by_agent1: " << edges_received
            << ", correct: " << n_correct
            << ", max_t_err_m: " << et_max << ", max_r_err_deg: " << er_max << "}\n"
            << "merge: {ok: " << (merge_a != nullptr) << ", merged_two_agent_ate_rmse_m: " << merged_rmse
            << ", samples: " << e_all.size() << "}\n"
            << "protocol: {broker_duplicates: " << broker_in.duplicates() << ", broker_buffered: " << broker_in.buffered()
            << ", agent1_duplicates: " << agent1_in.duplicates() << ", agent1_buffered: " << agent1_in.buffered()
            << ", undelivered: " << broker_in.waiting() + agent1_in.waiting() << "}\n"
            << "bytes:  # wire bytes per class (ByteLedger, §15.7)\n";
        for (MessageClass cls : {MessageClass::GLOBAL_DESCRIPTOR, MessageClass::MATCH_REQUEST, MessageClass::MATCH_RESPONSE,
                                 MessageClass::POSE_EDGE}) {
            res << "  " << wire::EnumNameMessageClass(cls) << ": {messages: " << ledger.CountForClass(cls)
                << ", wire_bytes: " << ledger.TotalForClass(cls) << "}\n";
        }
        res << "  total_wire_bytes: " << ledger.Total() << "\n";
        res.close();

        // contracts.yaml (Phase 1 exit gate): counted at each hop; the declared-unset fields are the ones the
        // records' headers name (keyframe_records.hpp, stereo_source.hpp), never defaulted in this run.
        {
            int classes_used = 0;
            for (MessageClass cls : {MessageClass::GLOBAL_DESCRIPTOR, MessageClass::MATCH_REQUEST,
                                     MessageClass::MATCH_RESPONSE, MessageClass::POSE_EDGE}) {
                classes_used += ledger.CountForClass(cls) > 0;
            }
            std::ofstream c(run_dir / "contracts.yaml");
            c << "# §4 / §18.2 data contracts exercised by this run (design PHASE1_GATE). Generated; do not hand-edit.\n";
            for (uint32_t k = 0; k < 2; ++k) {
                c << "agent" << k << ":\n"
                  << "  frame_4_1: {delivered: " << cc_count.frames[k] << "}\n"
                  << "  keyframe_4_2: {records: " << cc_count.keyframes[k]
                  << ", complete: " << cc_count.keyframes_complete[k]
                  << ", gauge_zero_cov: " << cc_count.gauge_zero_cov[k] << ", missing: {";
                bool first = true;
                for (const auto& [name, n] : cc_count.missing[k]) {
                    c << (first ? "" : ", ") << name << ": " << n;
                    first = false;
                }
                c << "}}\n"
                  << "  keyframe_bundle_18_2: {built: " << cc_count.bundles[k]
                  << ", spooled: " << cc_count.bundles_spooled[k] << "}\n";
            }
            c << "loop_edge_4_3: {encoded: " << cc_count.edges_encoded
              << ", decoded_identical: " << cc_count.edges_decoded_identical
              << ", merge_edge_received: " << (cc_count.merge_edge_received ? "true" : "false") << "}\n"
              << "byte_record_4_4: {ledger_rows: " << ledger_log.seq << ", classes_used: " << classes_used
              << ", total_wire_bytes: " << ledger.Total() << "}\n"
              << "unset_by_design:\n"
              << "  frame_4_1: [imu_window, photo, illumination, gt]\n"
              << "  keyframe_4_2: [velocity_world, bias_gyro, bias_accel, gravity_dir, preint_from_prev, local_cloud, "
                 "motion_cost, energy_cost, timestamp_age, uncertainty]\n"
              << "  keyframe_bundle_18_2: [exposure_state, lamp_state]\n"
              << "  loop_edge_4_3: [pcm_score, gnc_weight, degeneracy_flag]   # carried at defaults until Phase 3\n"
              << "  message_class_4_4: [HEARTBEAT, LOCAL_DESCRIPTOR, POSE_GRAPH, SUBMAP_SUMMARY, SUBMAP_PATCH, "
                 "TASK_BID, TASK_CONSENSUS, HUMAN_REQUEST, ACK]   # no Phase 1 producer\n";
        }

        runtime::RunManifest m;
        m.run_id = run_dir.filename().string();
        m.created_utc = tools::UtcNow("%Y-%m-%dT%H:%M:%SZ");
        m.git_commit = std::getenv("ARIADNE_GIT_COMMIT") ? std::getenv("ARIADNE_GIT_COMMIT") : runtime::kUnknownField;
        m.git_dirty = std::getenv("ARIADNE_GIT_DIRTY") && std::string(std::getenv("ARIADNE_GIT_DIRTY")) == "true";
        m.image_digest = std::getenv("ARIADNE_IMAGE_DIGEST") ? std::getenv("ARIADNE_IMAGE_DIGEST") : runtime::kUnknownField;
        m.scene = "sim/" + scene_id;
        m.seed = va.seed;
        m.seed_recorded = true;
        m.agent_count = 2;
        m.execution_mode = "replay";
        m.communication_config = cc.name;
        m.perception_config = va.perception_config;
        m.slam_config = va.slam_config;
        m.degradation_config = "none";
        m.feature_engine = "orb";
        m.place_recognition = "mean_bit_orb_placeholder";
        m.loop_backend = "essential+pnp_placeholder";
        m.imu_mode = "none";
        runtime::RecordBuildIdentity(m);
        runtime::RecordOpenCvIdentity(m);
        runtime::RecordDeterminismDigests(m);
        std::string cmd;
        for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
        m.extra["command"] = cmd;
        std::ofstream(run_dir / "manifest.yaml") << runtime::ToYaml(m);
        std::printf("collab: %zu candidates, %d verified (%d true), merged ATE %.4f m; %llu wire bytes -> %s\n",
                    cands.size(), n_verified, n_verified_true, merged_rmse,
                    (unsigned long long)ledger.Total(), run_dir.c_str());
        return merge_a ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_sim_collab: %s\n", e.what());
        return 1;
    }
}
