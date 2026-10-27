#pragma once
// ---------------------------------------------------------------------------
// Intra-agent loop closure, the runtime adapter (D73 part 2b; design robust-backend.md §8). It lives in runtime
// because it joins perception-side features, collaboration's verifier and slam's backend (arch §3: slam and
// collaboration are peers; only runtime may include both).
//
// Per keyframe (data in -> data out):
//   rectified left image + masked SGBM depth  ->  ORB keypoints/descriptors + 3D points (B's stereo)  [stored, capped]
//   mean-bit global descriptor (1.6 PLACEHOLDER until 2.1's EigenPlaces) -> retrieval among older keyframes
//   each candidate B: edge_verifier (E gate, PnP) with the CURRENT keyframe as A (pixels) and B's 3D points
//   verified candidates wait until both keyframes' odometry is final (they left the BA window), then one event:
//   degeneracy gate -> PCM -> GNC (weight reuse) on the backend's mirror graph.
// Read-only towards the VO: the corrected trajectory is the mirror (plus VO-relative dead reckoning for keyframes
// still in the window). Outputs: loop_trace.jsonl (one line per event), loop_candidates.csv (every verified
// candidate with its ground-truth error, for the D71 calibration), and the funnel / recall summary.
// ---------------------------------------------------------------------------
#include "collaboration/broker/edge_verifier.hpp"
#include "collaboration/retrieval/mean_bit_descriptor.hpp"
#include "core/geometry/pinhole.hpp"
#include "slam/loop_closure/loop_backend.hpp"
#include "slam/loop_closure/retrieval.hpp"
#include "slam/slam_config.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace ariadne::runtime {

class IntraLoopClosure {
public:
    struct Funnel {
        int keyframes = 0, retrieved = 0, verified = 0, gate_passed = 0, events = 0, unprocessed_at_end = 0;
        int unmirrorable = 0;   // verified candidates dropped: a keyframe before the mirror's anchor (D87)
    };
    struct Proposal {   // every retrieval proposal, verified or not (recall)
        uint64_t a = 0, b = 0;
        double score = 0;
        bool verified = false;
        std::string verify_reason;
    };
    struct Event {
        uint64_t query = 0;
        int64_t t_query_ns = 0, t_event_ns = 0;   // latency = event - query (window wait + processing)
        slam::LoopEventRecord rec;
    };

    IntraLoopClosure(const slam::LoopClosureParams& p, const core::RectifiedStereo& rig, uint64_t seed)
        : p_(p), rig_(rig), seed_(seed), orb_(cv::ORB::create(p.orb_features)),
          retrieval_(slam::RetrievalParams{p.exclude_recent, p.exclude_seconds, p.min_score, p.top_k}),
          backend_(BackendParams(p)) {}

    /// One new keyframe: features, retrieval, verification, then sync the mirror and run every event now ready.
    void OnKeyframe(uint64_t id, int64_t t_ns, const cv::Mat& left_rect, const cv::Mat& depth32,
                    const slam::PoseGraph& vo) {
        ++funnel.keyframes;
        t_of_[id] = t_ns;
        collaboration::KeyframeFeatures f;
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        orb_->detectAndCompute(left_rect, cv::noArray(), kps, desc);
        f.id = id;
        f.t_ns = t_ns;
        f.camera = {float(rig_.cam.fx), float(rig_.cam.fy), float(rig_.cam.cx), float(rig_.cam.cy), float(rig_.baseline)};
        f.descriptors.assign(desc.data, desc.data + desc.total() * desc.elemSize());
        for (const auto& kp : kps) {
            f.keypoints.push_back(kp.pt.x);
            f.keypoints.push_back(kp.pt.y);
            const int x = std::clamp(cvRound(kp.pt.x), 0, depth32.cols - 1), y = std::clamp(cvRound(kp.pt.y), 0, depth32.rows - 1);
            const float z = depth32.at<float>(y, x);
            const bool ok = std::isfinite(z) && z > 0.0f;
            f.points3d.push_back(ok ? float((kp.pt.x - rig_.cam.cx) / rig_.cam.fx * z) : NAN);
            f.points3d.push_back(ok ? float((kp.pt.y - rig_.cam.cy) / rig_.cam.fy * z) : NAN);
            f.points3d.push_back(ok ? z : NAN);
        }
        const std::vector<float> g = collaboration::MeanBitDescriptor(f.descriptors);
        const auto cands = retrieval_.Query(id, t_ns, g);
        retrieval_.Add(id, t_ns, g);
        Pending pend{id, t_ns, {}};
        for (const auto& [b, score] : cands) {
            ++funnel.retrieved;
            const collaboration::KeyframeFeatures& fb = features_.at(b);
            const core::FeatureResponseValue fr{0, b, fb.keypoints, fb.descriptors, fb.points3d, fb.camera};
            const collaboration::VerifyResult v =
                collaboration::VerifyEdge(f, fr, collaboration::VerifyParams{}, seed_ * 1000003ull + id * 7919ull + b);
            proposals.push_back({id, b, score, v.ok, v.reason});
            if (!v.ok) continue;
            ++funnel.verified;
            slam::LoopCandidate c;
            c.a = id;
            c.b = b;
            c.retrieval_score = score;
            c.geometry.rig = rig_;
            for (const auto& q : v.inlier_points_B) c.geometry.X_B.push_back(core::Vec3(q.x, q.y, q.z));
            for (const auto& q : v.inlier_pixels_A) c.geometry.uv_A.push_back(core::Vec2(q.x, q.y));
            c.geometry.T_A_B = v.T_A_B;
            pend.cands.push_back(std::move(c));
        }
        features_[id] = std::move(f);
        if (!pend.cands.empty()) pending_.push_back(std::move(pend));
        backend_.Sync(vo);
        Drain(t_ns);
    }

    /// End of the run: last sync, process what became ready; the rest (still in the window) is counted, not run.
    void Finish(const slam::PoseGraph& vo, int64_t t_ns) {
        backend_.Sync(vo);
        Drain(t_ns);
        funnel.unprocessed_at_end = int(pending_.size());
    }

    /// The loop-corrected camera pose of keyframe `kf`: the mirror's, or (still in the BA window) the last mirrored
    /// keyframe's corrected pose composed with the VO's relative motion since.
    core::SE3 Corrected(uint64_t kf, const slam::PoseGraph& vo) const {
        const auto& m = backend_.graph().nodes();
        const auto it = m.find(kf);
        if (it != m.end()) return it->second.T_W_C;
        if (m.empty()) return vo.nodes().at(kf).T_W_C;
        auto last = m.lower_bound(kf);
        if (last == m.begin()) return vo.nodes().at(kf).T_W_C;
        --last;
        return last->second.T_W_C * vo.nodes().at(last->first).T_W_C.inverse() * vo.nodes().at(kf).T_W_C;
    }

    /// Trace + per-candidate ground-truth errors + summary. gt(t_ns) = ground-truth rectified-left camera pose.
    void WriteOutputs(const std::string& dir, const std::function<std::optional<core::SE3>(int64_t)>& gt,
                      std::ostream& summary) const {
        std::ofstream tr(dir + "/loop_trace.jsonl");
        for (const Event& e : events) {
            tr << "{\"query\":" << e.query << ",\"t_query_ns\":" << e.t_query_ns << ",\"t_event_ns\":" << e.t_event_ns
               << ",\"loops_total\":" << e.rec.loops_total << ",\"pcm_accepted\":" << e.rec.pcm_accepted
               << ",\"gnc_inliers\":" << e.rec.gnc_inliers << ",\"solved\":" << (e.rec.solved ? "true" : "false")
               << ",\"fast_path\":" << (e.rec.fast_path ? "true" : "false") << ",\"gnc_stop\":\"" << e.rec.gnc_stop
               << "\",\"innovation\":" << e.rec.innovation << ",\"innovation_bound\":" << e.rec.innovation_bound
               << ",\"ms\":{\"gate\":" << e.rec.ms_gate << ",\"pcm\":" << e.rec.ms_pcm << ",\"gnc\":" << e.rec.ms_gnc
               << "}";
            if (e.rec.certificate_ran)   // D74, only with loop_closure.certificate: flag-off traces are unchanged
                tr << ",\"certificate\":{\"certified\":" << (e.rec.certified ? "true" : "false") << ",\"verdict\":\""
                   << e.rec.certificate_verdict << "\",\"ms\":" << e.rec.ms_certificate << "}";
            tr << ",\"candidates\":[";
            for (size_t i = 0; i < e.rec.candidates.size(); ++i) {
                const auto& c = e.rec.candidates[i];
                std::string why;
                for (const auto& s : c.gate_reasons) why += (why.empty() ? "" : ",") + ("\"" + s + "\"");
                tr << (i ? "," : "") << "{\"a\":" << c.a << ",\"b\":" << c.b << ",\"score\":" << c.retrieval_score
                   << ",\"degenerate\":" << (c.degenerate ? "true" : "false") << ",\"reasons\":[" << why
                   << "],\"sigma_t_m\":" << c.sigma_t << ",\"sigma_r_rad\":" << c.sigma_r
                   << ",\"corridor\":" << c.corridor << ",\"edge\":" << c.edge
                   << ",\"pcm\":" << (c.pcm_accepted ? "true" : "false") << ",\"gnc_inlier\":"
                   << (c.gnc_inlier ? "true" : "false") << ",\"gnc_weight\":" << c.gnc_weight << "}";
            }
            tr << "]}\n";
        }
        // Every gated candidate with its error against ground truth (design §4 calibration protocol).
        std::ofstream cs(dir + "/loop_candidates.csv");
        // xi* = the left error of the edge (exp(xi) T_A_B = T_true, translation first); cov* = cov_left's upper
        // triangle row-major: every noise-model fit runs offline on this file (design §8).
        cs << "a,b,score,degenerate,reasons,sigma_t_m,sigma_r_rad,err_t_m,err_r_deg,nees6,label,pcm,gnc_inlier";
        for (int i = 0; i < 6; ++i) cs << ",xi" << i;
        for (int i = 0; i < 6; ++i)
            for (int j = i; j < 6; ++j) cs << ",cov" << i << j;
        cs << ",corridor\n";   // appended last (D148), so readers of the earlier columns are unchanged
        std::vector<std::pair<long, bool>> labelled_edges;   // (mirror edge, good) for the PCM diagnostic
        int good = 0, bad = 0, labelled = 0, gnc_in = 0, gnc_in_bad = 0;
        for (const Event& e : events)
            for (const auto& c : e.rec.candidates) {
                const auto ga = gt(t_of_.at(c.a)), gb = gt(t_of_.at(c.b));
                double et = NAN, er = NAN, nees = NAN;
                core::Vec6 xi = core::Vec6::Constant(NAN);
                std::string label = "no_gt";
                if (ga && gb) {
                    const core::Vec6 err = core::SE3Log(c.T_A_B.inverse() * (ga->inverse() * *gb));
                    et = err.head<3>().norm();
                    er = err.tail<3>().norm() * 180.0 / M_PI;
                    const Eigen::LDLT<core::Mat6> ldlt(c.cov_left);
                    if (c.cov_left.allFinite() && ldlt.info() == Eigen::Success && !c.cov_left.isZero()) {
                        // left error of the edge: exp(xi) T_A_B = T_true
                        nees = 0.0;
                    }
                    xi = core::SE3Log((ga->inverse() * *gb) * c.T_A_B.inverse());
                    if (std::isfinite(nees)) nees = xi.dot(ldlt.solve(xi));
                    // bad: beyond 0.2 m / 2 deg. The design §4 protocol also had "beyond 3 sigma of its own covariance";
                    // dropped after V1_01 measured the loop covariance ~100-1000x overconfident on real data (median
                    // NEES ~1e4), which made every candidate "bad". NEES stays in the CSV as a reported column.
                    const bool is_bad = et > 0.2 || er > 2.0;
                    label = is_bad ? "bad" : "good";
                    ++labelled;
                    (is_bad ? bad : good)++;
                    gnc_in += c.gnc_inlier;
                    gnc_in_bad += c.gnc_inlier && is_bad;
                    if (c.edge >= 0) labelled_edges.push_back({c.edge, !is_bad});
                }
                std::string why;
                for (const auto& s : c.gate_reasons) why += (why.empty() ? "" : "|") + s;
                cs << c.a << ',' << c.b << ',' << c.retrieval_score << ',' << c.degenerate << ',' << why << ','
                   << c.sigma_t << ',' << c.sigma_r << ',' << et << ',' << er << ',' << nees << ',' << label << ','
                   << c.pcm_accepted << ',' << c.gnc_inlier;
                for (int i = 0; i < 6; ++i) cs << ',' << xi(i);
                for (int i = 0; i < 6; ++i)
                    for (int j = i; j < 6; ++j) cs << ',' << c.cov_left(i, j);
                cs << ',' << c.corridor << '\n';
            }
        // PCM's other input (design §8): the pair statistic over pairs of ground-truth-GOOD loops. Honest edge and
        // odometry covariances give a mean ~6 (chi^2_6); far above means overconfidence somewhere.
        std::vector<double> pair_good;
        for (size_t i = 0; i < labelled_edges.size(); ++i)
            for (size_t j = i + 1; j < labelled_edges.size(); ++j)
                if (labelled_edges[i].second && labelled_edges[j].second) {
                    const double c = slam::PairConsistency(backend_.graph(), size_t(labelled_edges[i].first),
                                                           size_t(labelled_edges[j].first));
                    if (std::isfinite(c)) pair_good.push_back(c);
                }
        std::sort(pair_good.begin(), pair_good.end());
        double pair_mean = 0;
        for (double v : pair_good) pair_mean += v;
        pair_mean = pair_good.empty() ? NAN : pair_mean / double(pair_good.size());
        const double pair_median = pair_good.empty() ? NAN : pair_good[pair_good.size() / 2];
        // Retrieval recall against ground-truth revisits: a query with an eligible revisit (outside the exclusion
        // window, within 1 m and 30 deg) is a hit if any of its proposals is such a revisit.
        int queries_with_revisit = 0, hits = 0;
        {
            std::map<uint64_t, core::SE3> T;
            for (const auto& [id, t] : t_of_)
                if (const auto g = gt(t)) T[id] = *g;
            auto revisit = [&](uint64_t a, uint64_t b) {
                const auto ia = T.find(a), ib = T.find(b);
                if (ia == T.end() || ib == T.end()) return false;
                if (b + uint64_t(std::max(p_.exclude_recent, 0)) >= a) return false;
                if (std::fabs(double(t_of_.at(a) - t_of_.at(b))) * 1e-9 < p_.exclude_seconds) return false;
                const core::SE3 rel = ia->second.inverse() * ib->second;
                return rel.t.norm() < 1.0 && core::SO3Log(rel.R).norm() < 30.0 * M_PI / 180.0;
            };
            std::map<uint64_t, std::vector<uint64_t>> proposed;
            for (const auto& pr : proposals) proposed[pr.a].push_back(pr.b);
            for (const auto& [a, ta] : T) {
                bool any = false;
                for (const auto& [b, tb] : T) {
                    if (b >= a) break;
                    if (revisit(a, b)) { any = true; break; }
                }
                if (!any) continue;
                ++queries_with_revisit;
                for (uint64_t b : proposed[a])
                    if (revisit(a, b)) { ++hits; break; }
            }
        }
        double lat = 0, lat_max = 0;
        int fast = 0, solved = 0;
        for (const Event& e : events) {
            const double l = (e.t_event_ns - e.t_query_ns) * 1e-9 + (e.rec.ms_gate + e.rec.ms_pcm + e.rec.ms_gnc) * 1e-3;
            lat += l;
            lat_max = std::max(lat_max, l);
            fast += e.rec.fast_path;
            solved += e.rec.solved;
        }
        summary << "loop_closure: {keyframes: " << funnel.keyframes << ", queries_with_gt_revisit: "
                << queries_with_revisit << ", retrieval_hits: " << hits << ", retrieved: " << funnel.retrieved
                << ", verified: " << funnel.verified << ", gate_passed: " << funnel.gate_passed
                << ", events: " << funnel.events << ", solved_events: " << solved << ", fast_path: " << fast
                << ", unprocessed_at_end: " << funnel.unprocessed_at_end << ", unmirrorable: " << funnel.unmirrorable
                << ", mirror_anchor_kf: " << backend_.anchor() << ", labelled: " << labelled
                << ", labelled_good: " << good << ", labelled_bad: " << bad << ", gnc_inliers_final_events: " << gnc_in
                << ", gnc_inliers_bad: " << gnc_in_bad << ", latency_mean_s: " << (events.empty() ? 0.0 : lat / events.size())
                << ", latency_max_s: " << lat_max << ", pcm_pair_stat_good_pairs: " << pair_good.size()
                << ", pcm_pair_stat_good_median: " << pair_median << ", pcm_pair_stat_good_mean: " << pair_mean
                << "}  # descriptor: mean-bit PLACEHOLDER; pair stat ~6 if covariances are honest\n";
    }

    const slam::LoopBackend& backend() const { return backend_; }

    Funnel funnel;
    std::vector<Proposal> proposals;
    std::vector<Event> events;

private:
    static slam::LoopBackendParams BackendParams(const slam::LoopClosureParams& p) {
        slam::LoopBackendParams b;
        b.gate.noise_model = p.noise_model;
        b.gate.noise_k_t = p.noise_k_t;
        b.gate.noise_floor_t_m = p.noise_floor_t_m;
        b.gate.noise_k_r = p.noise_k_r;
        b.gate.noise_floor_r_rad = p.noise_floor_r_rad;
        b.gnc.lm.gravity.enabled = p.gravity_priors;
        b.certificate.enabled = p.certificate;
        return b;
    }
    struct Pending {
        uint64_t q;
        int64_t t_q;
        std::vector<slam::LoopCandidate> cands;
    };
    void Drain(int64_t now_ns) {
        while (!pending_.empty()) {
            Pending& p = pending_.front();
            // A candidate the mirror can never hold would block this FIFO for the rest of the run (D87: MH_04/MH_05
            // closed 0 loops). Drop those candidates only; the query's others still run.
            const auto dead = std::remove_if(p.cands.begin(), p.cands.end(), [&](const slam::LoopCandidate& c) {
                return backend_.Unreachable(c.a) || backend_.Unreachable(c.b);
            });
            funnel.unmirrorable += int(p.cands.end() - dead);
            p.cands.erase(dead, p.cands.end());
            if (p.cands.empty()) {
                pending_.pop_front();
                continue;
            }
            bool ready = true;
            for (const auto& c : p.cands) ready = ready && backend_.Ready(c.a, c.b);
            if (!ready) break;   // FIFO: events run in query order
            Event e;
            e.query = p.q;
            e.t_query_ns = p.t_q;
            e.t_event_ns = now_ns;
            e.rec = backend_.OnCandidates(p.cands, now_ns);
            for (const auto& c : e.rec.candidates) funnel.gate_passed += !c.degenerate;
            ++funnel.events;
            events.push_back(std::move(e));
            pending_.pop_front();
        }
    }

    slam::LoopClosureParams p_;
    core::RectifiedStereo rig_;
    uint64_t seed_;
    cv::Ptr<cv::ORB> orb_;
    slam::LoopRetrieval retrieval_;
    slam::LoopBackend backend_;
    std::map<uint64_t, collaboration::KeyframeFeatures> features_;
    std::map<uint64_t, int64_t> t_of_;
    std::deque<Pending> pending_;
};

}  // namespace ariadne::runtime
