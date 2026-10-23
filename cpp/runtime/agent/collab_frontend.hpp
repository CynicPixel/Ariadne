#pragma once
// ---------------------------------------------------------------------------
// CollabFrontend: one agent's side of the 3.x collaboration, fed by its VO front end (design LIVE_COLLAB §1-§2, D174).
// Extracted from ariadne_collab_passa so pass A (replay) and the live estimator worker run the same code.
//
// Per keyframe at its creation (t_create): the EigenPlaces R50-256 place descriptor of the rectified RGB, and its
// ALIKED-n16 local features with 3D points from the masked full-res SGBM depth. Then the keyframe waits: a LoopBackend
// mirror with no loop candidates is synced at every keyframe, and a keyframe is RELEASED (a CollabStoreRecord) when the
// mirror takes it, i.e. its odometry edge became final (it left the BA window), at that frame's time (t_final).
// Keyframes before the mirror's anchor never become final (the D87 startup prefix).
//
// Local features: `eager` computes them at t_create (pass A: the store carries them). Lazy (live, LIVE_COLLAB §2)
// keeps each keyframe's RGB and masked depth and computes them only when CollabNode asks (`Features`), because ALIKED
// costs ~2.3 s/kf on CPU and only budgeted, selected candidates are ever fetched. The model is deterministic, so lazy
// and eager give identical features for the same keyframe. A kept image is dropped `keep_s` after its t_final
// (candidate_ttl_s, D170); a request after that is answered "gone" (nullopt).
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
#include <stdexcept>
#include <vector>

namespace ariadne::runtime {

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

struct CollabPinhole {
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

struct CollabLocalFeatures {
    std::vector<float> keypoints, descriptors, points3d;
};

struct CollabFrontendParams {
    bool eager = true;            // pass A: features at t_create; live: false (on request)
    double keep_s = 120.0;        // lazy: images kept this long after t_final (candidate_ttl_s, D170)
};

struct CollabFrontendStats {
    uint64_t keyframes = 0, released = 0, features_computed = 0, features_gone = 0;
    double ms_aliked = 0, ms_place = 0;
};

class CollabFrontend {
public:
    CollabFrontend(const perception::PlaceRecognizer& place, const perception::HardPairMatcher& aliked,
                   CollabFrontendParams p = {})
        : place_(place), aliked_(aliked), p_(p) {}

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

    /// One keyframe at its creation; returns the records released by it (id order, t_final = this keyframe's time).
    std::vector<CollabStoreRecord> OnKeyframe(const CollabKeyframeInput& kf) {
        ++stats_.keyframes;
        Pending p;
        p.t_create_ns = kf.t_ns;
        p.T_B_C = kf.T_B_C;
        p.gt_T_W_B = kf.gt_T_W_B;
        p.pinhole = {kf.fx, kf.fy, kf.cx, kf.cy};
        p.camera = {float(kf.fx), float(kf.fy), float(kf.cx), float(kf.cy), float(kf.baseline_m)};
        if (p_.eager) {
            p.local = ExtractLocal(kf.rgb, kf.depth32, p.pinhole);
        } else {
            p.rgb = kf.rgb.clone();
            p.depth32 = kf.depth32.clone();
        }
        const auto t1 = std::chrono::steady_clock::now();
        p.global = place_.DescribeRgb(kf.rgb);
        stats_.ms_place += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        pending_[kf.id] = std::move(p);

        // Release whatever the mirror now holds as final, in id order, at this frame's time.
        std::vector<CollabStoreRecord> out;
        mirror_.Sync(*kf.graph);
        const auto& nodes = mirror_.graph().nodes();
        for (auto it = last_released_ ? nodes.upper_bound(*last_released_) : nodes.begin(); it != nodes.end(); ++it) {
            const auto pit = pending_.find(it->first);
            if (pit == pending_.end()) throw std::logic_error("collab frontend: a final keyframe without its features");
            CollabStoreRecord r;
            r.id = it->first;
            r.t_create_ns = pit->second.t_create_ns;
            r.t_final_ns = kf.t_ns;
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
                if (!found) throw std::logic_error("collab frontend: no final odometry edge into a released keyframe");
            }
            r.global = std::move(pit->second.global);
            r.keypoints = std::move(pit->second.local.keypoints);
            r.descriptors = std::move(pit->second.local.descriptors);
            r.points3d = std::move(pit->second.local.points3d);
            r.camera = pit->second.camera;
            r.has_gt = pit->second.gt_T_W_B.has_value();
            if (r.has_gt) r.gt_T_W_B = *pit->second.gt_T_W_B;
            r.T_B_C = pit->second.T_B_C;
            if (!p_.eager) kept_[r.id] = Kept{kf.t_ns, std::move(pit->second.rgb), std::move(pit->second.depth32), pit->second.pinhole};
            out.push_back(std::move(r));
            pending_.erase(pit);
            last_released_ = it->first;
            ++stats_.released;
        }
        // Lazy: forget images whose keep window has passed (sim time of this keyframe).
        for (auto it = kept_.begin(); it != kept_.end();)
            it = (kf.t_ns - it->second.t_final_ns) * 1e-9 > p_.keep_s ? kept_.erase(it) : std::next(it);
        return out;
    }

    /// Lazy: the local features of a released keyframe, computed now; nullopt if never released or past keep_s.
    std::optional<CollabLocalFeatures> Features(uint64_t id) {
        const auto it = kept_.find(id);
        if (it == kept_.end()) {
            ++stats_.features_gone;
            return std::nullopt;
        }
        return ExtractLocal(it->second.rgb, it->second.depth32, it->second.pinhole);
    }

    /// Keyframes never released: the startup prefix (ids below the first released one) and the tail in the window.
    std::pair<uint64_t, uint64_t> Unreleased(const std::optional<uint64_t>& first_released) const {
        uint64_t prefix = 0, tail = 0;
        for (const auto& [id, p] : pending_) (!first_released || id < *first_released ? prefix : tail)++;
        return {prefix, tail};
    }

    const CollabFrontendStats& stats() const { return stats_; }
    const perception::PlaceRecognizer& place() const { return place_; }
    const perception::HardPairMatcher& aliked() const { return aliked_; }

private:
    struct Pending {   // waiting for the keyframe's odometry to become final
        int64_t t_create_ns = 0;
        std::vector<float> global;
        CollabLocalFeatures local;   // eager
        cv::Mat rgb, depth32;        // lazy
        CollabPinhole pinhole;
        std::array<float, 5> camera{};
        std::optional<core::SE3> gt_T_W_B;
        core::SE3 T_B_C;
    };
    struct Kept {   // lazy: a released keyframe's inputs, until keep_s after t_final
        int64_t t_final_ns = 0;
        cv::Mat rgb, depth32;
        CollabPinhole pinhole;
    };
    const perception::PlaceRecognizer& place_;
    const perception::HardPairMatcher& aliked_;
    CollabFrontendParams p_;
    std::map<uint64_t, Pending> pending_;
    std::map<uint64_t, Kept> kept_;
    slam::LoopBackend mirror_;   // no candidates ever: it only says which odometry is final
    std::optional<uint64_t> last_released_;
    CollabFrontendStats stats_;
};

}  // namespace ariadne::runtime
