#pragma once
// ---------------------------------------------------------------------------
// 2.1 re-association glue (PERCEPTION_2.1.md §4.3): between the tracker and
// StereoVo, owned by the runtime because it reads both. Kept out of
// stereo_vo_runner.hpp so that shared file only gains call sites.
//
// State: one descriptor per live landmark, taken from its latest KEYFRAME
// observation (the pixel its track held there), refreshed at every keyframe
// and pruned when StereoVo retires the landmark. ORB-SLAM's representative
// (medoid) descriptor is ablation A5.
//
// Per frame, BEFORE StereoVo::Track:
//   - count the live tracks that hold a landmark (what PnP will see);
//   - if fewer than `trigger_correspondences`: describe the tracks that hold NO
//     landmark at their pixel, and match them against the healthy landmarks no
//     live track holds (perception::ReassociateLost: mutual ratio test, then a
//     seeded PnP-RANSAC with the healthy correspondences as anchors);
//   - accepted renames go to the tracker (next frame) and into this frame's
//     VoFrame (StereoVo then finds them by id).
// Healthy = created at a keyframe whose pose was MEASURED (D56 PoseSource),
// unless admit_bridged (ablation A1b). Window-only: StereoVo's own retirement
// decides which landmarks exist (§4.0 window expiry; the out-of-window store is
// ablation A1s, built only if needed).
//
// Disabled (no `reassociation:` section): nothing is called, nothing is written.
// ---------------------------------------------------------------------------
#include "perception/association/landmark_reassociation.hpp"
#include "perception/features/feature_engine.hpp"
#include "perception/perception_config.hpp"
#include "perception/tracking/klt_tracker.hpp"
#include "slam/odometry/stereo_vo.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ariadne::tools {

class LostReassociator {
public:
    LostReassociator(const perception::ReassociationParams& params, const perception::LocalFeatureEngine& engine,
                     double fx, double fy, double cx, double cy)
        : p_(params), engine_(engine), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
        trace_ << "frame,correspondences,anchors,orphans,candidates,matched,inliers,match_inliers,accepted,why\n";
    }

    bool enabled() const { return p_.enabled; }

    /// Before vo.Track(vf). May rename tracks in `tracker` and in `vf.track_ids`.
    void BeforeTrack(const cv::Mat& gray, const slam::StereoVo& vo, perception::KltTracker& tracker,
                     slam::VoFrame& vf, uint64_t seed) {
        if (!p_.enabled || vo.graph().nodes().empty()) return;
        const std::vector<slam::StereoVo::LandmarkInfo> lms = vo.Landmarks();
        std::map<uint64_t, const slam::StereoVo::LandmarkInfo*> by_id;
        for (const auto& l : lms) by_id.emplace(l.id, &l);
        const auto healthy = [&](const slam::StereoVo::LandmarkInfo& l) {
            return l.born_source == slam::PoseSource::kMeasured ||
                   (p_.admit_bridged && l.born_source == slam::PoseSource::kBridged);
        };

        int correspondences = 0;
        perception::AnchorCorrespondences anchors;
        std::vector<size_t> orphan_idx;
        std::set<uint64_t> held;
        for (size_t i = 0; i < vf.track_ids.size(); ++i) {
            const auto it = by_id.find(vf.track_ids[i]);
            if (it == by_id.end()) {
                orphan_idx.push_back(i);
                continue;
            }
            ++correspondences;
            held.insert(it->first);
            if (healthy(*it->second)) {
                const core::Vec3& X = it->second->X_w;
                anchors.X_w.emplace_back(X.x(), X.y(), X.z());
                anchors.points.push_back(vf.points[i]);
            }
        }
        if (correspondences >= p_.trigger_correspondences) return;
        ++runs_;
        const auto t0 = std::chrono::steady_clock::now();

        perception::CandidateLandmarks cand;
        std::vector<cv::Mat> cand_rows;
        for (const auto& l : lms) {
            if (held.count(l.id) || !healthy(l)) continue;
            const auto d = desc_.find(l.id);
            if (d == desc_.end()) continue;
            cand.landmark_ids.push_back(l.id);
            cand.X_w.emplace_back(l.X_w.x(), l.X_w.y(), l.X_w.z());
            cand_rows.push_back(d->second);
        }
        perception::OrphanTracks orph;
        std::vector<cv::Point2f> pts;
        for (size_t i : orphan_idx) pts.push_back(vf.points[i]);
        std::vector<uchar> ok;
        const cv::Mat d = engine_.Describe(gray, pts, ok);
        std::vector<cv::Mat> orph_rows;
        for (size_t k = 0; k < orphan_idx.size(); ++k) {
            if (!ok[k]) continue;
            orph.track_ids.push_back(vf.track_ids[orphan_idx[k]]);
            orph.points.push_back(pts[k]);
            orph_rows.push_back(d.row(static_cast<int>(k)));
        }
        if (!cand_rows.empty()) cv::vconcat(cand_rows, cand.descriptors);
        if (!orph_rows.empty()) cv::vconcat(orph_rows, orph.descriptors);

        perception::ReassociationResult r;
        if (cand_rows.empty() || orph_rows.empty()) {
            r.orphans = static_cast<int>(orph.track_ids.size());
            r.candidates = static_cast<int>(cand.landmark_ids.size());
            r.anchors = static_cast<int>(anchors.X_w.size());
            r.why = cand_rows.empty() ? "no healthy candidate" : "no describable orphan";
        } else {
            r = perception::ReassociateLost(orph, cand, anchors, fx_, fy_, cx_, cy_, p_, seed, vf.frame_id);
        }
        if (r.accepted) {
            tracker.Relabel(r.from, r.to);
            std::map<uint64_t, uint64_t> ren;
            for (size_t k = 0; k < r.from.size(); ++k) ren.emplace(r.from[k], r.to[k]);
            for (uint64_t& id : vf.track_ids) {
                const auto it = ren.find(id);
                if (it != ren.end()) id = it->second;
            }
            ++accepted_;
            relabeled_ += static_cast<long>(r.from.size());
        }
        matched_ += r.matched;
        run_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r.candidates == 0) ++no_candidates_;
        trace_ << vf.frame_id << ',' << correspondences << ',' << r.anchors << ',' << r.orphans << ','
               << r.candidates << ',' << r.matched << ',' << r.inliers << ',' << r.match_inliers << ','
               << int(r.accepted) << ',' << (r.why.empty() ? "-" : r.why) << '\n';
    }

    /// After vo.AddKeyframe() for the frame `vf` (ids as renamed): refresh the descriptor of every landmark a live
    /// track holds, at its pixel in this keyframe, and forget retired landmarks.
    void AfterKeyframe(const cv::Mat& gray, const slam::StereoVo& vo, const slam::VoFrame& vf) {
        if (!p_.enabled) return;
        const auto t0 = std::chrono::steady_clock::now();
        std::set<uint64_t> live;
        for (const auto& l : vo.Landmarks()) live.insert(l.id);
        for (auto it = desc_.begin(); it != desc_.end();) it = live.count(it->first) ? std::next(it) : desc_.erase(it);
        std::vector<cv::Point2f> pts;
        std::vector<uint64_t> ids;
        for (size_t i = 0; i < vf.track_ids.size(); ++i) {
            if (!live.count(vf.track_ids[i])) continue;
            ids.push_back(vf.track_ids[i]);
            pts.push_back(vf.points[i]);
        }
        std::vector<uchar> ok;
        const cv::Mat d = engine_.Describe(gray, pts, ok);
        for (size_t k = 0; k < ids.size(); ++k) {
            if (ok[k]) desc_[ids[k]] = d.row(static_cast<int>(k)).clone();
        }
        ++keyframes_;
        kf_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    /// results.yaml line (written only when enabled).
    std::string ResultsLine() const {
        std::ostringstream o;
        o << "reassociation: {mode: lost_window, descriptor: latest_keyframe, admit_bridged: "
          << (p_.admit_bridged ? "true" : "false") << ", trigger_correspondences: " << p_.trigger_correspondences
          << ", runs: " << runs_ << ", accepted: " << accepted_ << ", relabeled: " << relabeled_
          << ", matched: " << matched_ << ", runs_without_candidates: " << no_candidates_
          << ", run_ms_mean: " << (runs_ ? run_ms_ / runs_ : -1.0)
          << ", keyframe_describe_ms_mean: " << (keyframes_ ? kf_ms_ / keyframes_ : -1.0) << "}\n";
        return o.str();
    }
    /// Manifest value.
    std::string ManifestValue() const {
        if (!p_.enabled) return "off";
        return std::string("lost_window/latest_keyframe") + (p_.admit_bridged ? "/admit_bridged" : "");
    }
    std::string Trace() const { return trace_.str(); }

private:
    perception::ReassociationParams p_;
    const perception::LocalFeatureEngine& engine_;
    double fx_, fy_, cx_, cy_;
    std::map<uint64_t, cv::Mat> desc_;   // landmark id -> descriptor (one row)
    long runs_ = 0, accepted_ = 0, relabeled_ = 0, matched_ = 0, no_candidates_ = 0, keyframes_ = 0;
    double run_ms_ = 0.0, kf_ms_ = 0.0;   // wall clock of BeforeTrack runs / AfterKeyframe
    std::ostringstream trace_;
};

}  // namespace ariadne::tools
