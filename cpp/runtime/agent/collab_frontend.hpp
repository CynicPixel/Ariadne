#pragma once
// ---------------------------------------------------------------------------
// One agent's side of the 3.x collaboration, fed by its VO front end (design live-collaboration.md §1-§2, D174). Extracted from
// ariadne_collab_passa so pass A (replay) and the live processes run the same code. Three pieces:
//
//   CollabReleaseTracker   a LoopBackend mirror with no loop candidates, synced at every keyframe: a keyframe is
//                          RELEASED when the mirror takes it, i.e. its odometry edge became final (it left the BA
//                          window), at that frame's time (t_final). Keyframes before the mirror's anchor never become
//                          final (the D87 startup prefix). Cheap, no DNN: live, it runs in the estimator worker.
//   CollabFeatureExtractor EigenPlaces R50-256 at t_create; ALIKED-n16 local features with 3D points from the masked
//                          full-res SGBM depth, eagerly (pass A: the store carries them) or lazily on request (live:
//                          ~2.3 s/kf on CPU, and only budgeted, selected candidates are fetched). Live, it runs in
//                          ariadne_collab_worker, never in the estimator worker (its OpenCV pool would perturb VO,
//                          D152a/D168). Lazy and eager give identical features at the same OpenCV thread count.
//   CollabFrontend         both in one process, assembling CollabStoreRecords (pass A, and the lazy == eager check).
//
// A kept (lazy) image is dropped `keep_s` after its t_final (candidate_ttl_s, D170); a request after that is "gone".
// ---------------------------------------------------------------------------
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/place/place_recognizer.hpp"
#include "runtime/agent/collab_store.hpp"
#include "slam/loop_closure/loop_backend.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace ariadne::runtime {

struct CollabPinhole {
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

/// The keyframe as the front end hands it over (all in the rectified left camera).
struct CollabKeyframeInput {
    uint64_t id = 0;
    int64_t t_ns = 0;
    cv::Mat rgb;                  // rectified RGB, the size the ALIKED model expects
    cv::Mat depth32;              // full-res metric depth, NaN where unusable (MaskedDepth32)
    double fx = 0, fy = 0, cx = 0, cy = 0, baseline_m = 0;   // the rectified pinhole (double: the 3D points use it)
    core::SE3 T_B_C;
    std::optional<core::SE3> gt_T_W_B;   // EVALUATION ONLY
    const slam::PoseGraph* graph = nullptr;   // the VO graph right after this keyframe
};

struct CollabLocalFeatures {
    std::vector<float> keypoints, descriptors, points3d;
};

/// A released keyframe's odometry part (the tracker's output; live: the #crel message).
struct CollabRelease {
    uint64_t id = 0;
    int64_t t_final_ns = 0;
    core::SE3 T_W_C;
    std::optional<uint64_t> odo_from;
    core::SE3 odo_Z;
    std::array<double, 36> odo_information{};
};

class CollabReleaseTracker {
public:
    /// After keyframe `id` at `t_ns` entered `graph`: the keyframes whose odometry became final, in id order.
    std::vector<CollabRelease> OnKeyframe(uint64_t id, int64_t t_ns, const slam::PoseGraph& graph) {
        seen_.insert(id);
        std::vector<CollabRelease> out;
        mirror_.Sync(graph);
        const auto& nodes = mirror_.graph().nodes();
        for (auto it = last_released_ ? nodes.upper_bound(*last_released_) : nodes.begin(); it != nodes.end(); ++it) {
            if (!seen_.count(it->first)) throw std::logic_error("collab tracker: a final keyframe it never saw created");
            CollabRelease r;
            r.id = it->first;
            r.t_final_ns = t_ns;
            r.T_W_C = it->second.T_W_C;
            if (last_released_) {
                bool found = false;
                for (const auto& e : mirror_.graph().edges())
                    if (e.type == slam::EdgeType::kOdometry && e.from == *last_released_ && e.to == r.id) {
                        r.odo_from = e.from;
                        r.odo_Z = e.Z;
                        for (int q = 0; q < 36; ++q) r.odo_information[size_t(q)] = e.information(q / 6, q % 6);
                        found = true;
                    }
                if (!found) throw std::logic_error("collab tracker: no final odometry edge into a released keyframe");
            }
            if (!first_released_) first_released_ = r.id;
            last_released_ = it->first;
            seen_.erase(r.id);
            out.push_back(std::move(r));
        }
        return out;
    }
    /// Keyframes never released: the startup prefix (ids below the first released one) and the tail in the window.
    std::pair<uint64_t, uint64_t> Unreleased() const {
        uint64_t prefix = 0, tail = 0;
        for (uint64_t id : seen_) (!first_released_ || id < *first_released_ ? prefix : tail)++;
        return {prefix, tail};
    }

private:
    slam::LoopBackend mirror_;   // no candidates ever: it only says which odometry is final
    std::set<uint64_t> seen_;    // created, not yet released
    std::optional<uint64_t> last_released_, first_released_;
};

struct CollabExtractorStats {
    uint64_t described = 0, features_computed = 0, features_gone = 0;
    uint64_t evicted_early = 0;   // a timed request found its keyframe already evicted: a determinism hazard (> 0 = bug)
    double ms_aliked = 0, ms_place = 0;
};

class CollabFeatureExtractor {
public:
    CollabFeatureExtractor(const perception::PlaceRecognizer& place, const perception::HardPairMatcher& aliked,
                           double keep_s = 120.0, double evict_slack_s = 0.0)
        : place_(place), aliked_(aliked), keep_s_(keep_s), evict_slack_s_(evict_slack_s) {}

    std::vector<float> Describe(const cv::Mat& rgb) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> g = place_.DescribeRgb(rgb);
        stats_.ms_place += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++stats_.described;
        return g;
    }

    /// ALIKED on the RGB, each keypoint's 3D point from the depth at its rounded pixel (NaN = none).
    CollabLocalFeatures ExtractLocal(const cv::Mat& rgb, const cv::Mat& depth32, const CollabPinhole& cam) {
        const auto t0 = std::chrono::steady_clock::now();
        const perception::HardPairFeatures f = aliked_.ExtractRgb(rgb);
        stats_.ms_aliked += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++stats_.features_computed;
        CollabLocalFeatures o;
        for (int i = 0; i < int(f.keypoints.size()); ++i) {
            const cv::Point2f& q = f.keypoints[size_t(i)].pt;
            o.keypoints.push_back(q.x);
            o.keypoints.push_back(q.y);
            const int u = int(std::lround(q.x)), v = int(std::lround(q.y));
            float z = NAN;
            if (u >= 0 && v >= 0 && u < depth32.cols && v < depth32.rows) z = depth32.at<float>(v, u);
            const bool ok = std::isfinite(z) && z > 0.0f;
            o.points3d.push_back(ok ? float((q.x - cam.cx) / cam.fx * z) : NAN);
            o.points3d.push_back(ok ? float((q.y - cam.cy) / cam.fy * z) : NAN);
            o.points3d.push_back(ok ? z : NAN);
            const float* d = f.descriptors.ptr<float>(i);
            o.descriptors.insert(o.descriptors.end(), d, d + f.descriptors.cols);
        }
        return o;
    }

    /// Lazy: keep a keyframe's inputs (from its creation) for a later Features(id).
    void Keep(uint64_t id, cv::Mat rgb, cv::Mat depth32, const CollabPinhole& cam) {
        kept_[id] = Kept{std::nullopt, std::move(rgb), std::move(depth32), cam};
    }
    /// Lazy: the keyframe was released at t_final (its keep window starts); `now_ns` expires older ones.
    void Released(uint64_t id, int64_t t_final_ns) {
        const auto it = kept_.find(id);
        if (it != kept_.end()) it->second.t_final_ns = t_final_ns;
        Expire(t_final_ns);
    }
    /// Frees memory: drops keyframes older than keep_s + evict_slack_s at `now_ns`. With timed requests (live) the
    /// "gone" ANSWER never depends on this (see Features(id, t_req)); the slack covers the parent's lag behind `now`.
    void Expire(int64_t now_ns) {
        for (auto it = kept_.begin(); it != kept_.end();) {
            if (it->second.t_final_ns && (now_ns - *it->second.t_final_ns) * 1e-9 > keep_s_ + evict_slack_s_) {
                evicted_.insert(it->first);
                it = kept_.erase(it);
            } else {
                ++it;
            }
        }
    }
    /// Lazy: the local features of a released, kept keyframe, computed now; nullopt if never released or expired.
    std::optional<CollabLocalFeatures> Features(uint64_t id) {
        const auto it = kept_.find(id);
        if (it == kept_.end() || !it->second.t_final_ns) {
            ++stats_.features_gone;
            return std::nullopt;
        }
        return ExtractLocal(it->second.rgb, it->second.depth32, it->second.pinhole);
    }
    /// Live (D179): the answer is a function of SIM time only - gone iff never released or released more than keep_s
    /// before the request's tick - so it does not depend on how the estimator and parent streams interleave here.
    std::optional<CollabLocalFeatures> Features(uint64_t id, int64_t t_req_ns) {
        const auto it = kept_.find(id);
        if (it == kept_.end()) {
            stats_.evicted_early += evicted_.count(id);
            ++stats_.features_gone;
            return std::nullopt;
        }
        if (!it->second.t_final_ns || (t_req_ns - *it->second.t_final_ns) * 1e-9 > keep_s_) {
            ++stats_.features_gone;
            return std::nullopt;
        }
        return ExtractLocal(it->second.rgb, it->second.depth32, it->second.pinhole);
    }

    const CollabExtractorStats& stats() const { return stats_; }
    const perception::PlaceRecognizer& place() const { return place_; }
    const perception::HardPairMatcher& aliked() const { return aliked_; }

private:
    struct Kept {
        std::optional<int64_t> t_final_ns;   // unset until released
        cv::Mat rgb, depth32;
        CollabPinhole pinhole;
    };
    const perception::PlaceRecognizer& place_;
    const perception::HardPairMatcher& aliked_;
    double keep_s_, evict_slack_s_;
    std::map<uint64_t, Kept> kept_;
    std::set<uint64_t> evicted_;
    CollabExtractorStats stats_;
};

struct CollabFrontendParams {
    bool eager = true;            // pass A: features at t_create; false: on request (Features)
    double keep_s = 120.0;        // lazy: images kept this long after t_final (candidate_ttl_s, D170)
};

struct CollabFrontendStats {
    uint64_t keyframes = 0, released = 0;
    double ms_aliked = 0, ms_place = 0;
};

/// Tracker + extractor in one process: assembles whole CollabStoreRecords (pass A).
class CollabFrontend {
public:
    CollabFrontend(const perception::PlaceRecognizer& place, const perception::HardPairMatcher& aliked,
                   CollabFrontendParams p = {})
        : x_(place, aliked, p.keep_s), p_(p) {}

    /// One keyframe at its creation; returns the records released by it (id order, t_final = this keyframe's time).
    std::vector<CollabStoreRecord> OnKeyframe(const CollabKeyframeInput& kf) {
        ++keyframes_;
        Pending p;
        p.t_create_ns = kf.t_ns;
        p.T_B_C = kf.T_B_C;
        p.gt_T_W_B = kf.gt_T_W_B;
        const CollabPinhole pin{kf.fx, kf.fy, kf.cx, kf.cy};
        p.camera = {float(kf.fx), float(kf.fy), float(kf.cx), float(kf.cy), float(kf.baseline_m)};
        if (p_.eager) p.local = x_.ExtractLocal(kf.rgb, kf.depth32, pin);
        else x_.Keep(kf.id, kf.rgb.clone(), kf.depth32.clone(), pin);
        p.global = x_.Describe(kf.rgb);
        pending_[kf.id] = std::move(p);

        std::vector<CollabStoreRecord> out;
        for (CollabRelease& rel : tracker_.OnKeyframe(kf.id, kf.t_ns, *kf.graph)) {
            const auto pit = pending_.find(rel.id);
            if (pit == pending_.end()) throw std::logic_error("collab frontend: a final keyframe without its features");
            CollabStoreRecord r = Assemble(rel, pit->second);
            if (!p_.eager) x_.Released(r.id, r.t_final_ns);
            out.push_back(std::move(r));
            pending_.erase(pit);
        }
        released_ += out.size();
        return out;
    }

    std::optional<CollabLocalFeatures> Features(uint64_t id) { return x_.Features(id); }
    std::pair<uint64_t, uint64_t> Unreleased() const { return tracker_.Unreleased(); }
    CollabFrontendStats stats() const {
        return {keyframes_, released_, x_.stats().ms_aliked, x_.stats().ms_place};
    }

private:
    struct Pending {   // waiting for the keyframe's odometry to become final
        int64_t t_create_ns = 0;
        std::vector<float> global;
        CollabLocalFeatures local;   // eager
        std::array<float, 5> camera{};
        std::optional<core::SE3> gt_T_W_B;
        core::SE3 T_B_C;
    };
    static CollabStoreRecord Assemble(const CollabRelease& rel, Pending& p) {
        CollabStoreRecord r;
        r.id = rel.id;
        r.t_create_ns = p.t_create_ns;
        r.t_final_ns = rel.t_final_ns;
        r.T_W_C = rel.T_W_C;
        r.odo_from = rel.odo_from;
        r.odo_Z = rel.odo_Z;
        r.odo_information = rel.odo_information;
        r.global = std::move(p.global);
        r.keypoints = std::move(p.local.keypoints);
        r.descriptors = std::move(p.local.descriptors);
        r.points3d = std::move(p.local.points3d);
        r.camera = p.camera;
        r.has_gt = p.gt_T_W_B.has_value();
        if (r.has_gt) r.gt_T_W_B = *p.gt_T_W_B;
        r.T_B_C = p.T_B_C;
        return r;
    }
    CollabFeatureExtractor x_;
    CollabReleaseTracker tracker_;
    CollabFrontendParams p_;
    std::map<uint64_t, Pending> pending_;
    uint64_t keyframes_ = 0, released_ = 0;
};

}  // namespace ariadne::runtime
