#include "perception/tracking/klt_tracker.hpp"

#include "core/random/philox.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ariadne::perception {

namespace {

// Philox site reserved for the tracker's robust estimator. Pixel-keyed users
// (the degradation operator) draw from sites far below this, so the streams
// never overlap. Reserved sites count down from the top of the 48-bit domain.
constexpr uint64_t kSiteTrackerUsac = core::PhiloxStream::kMaxSite - 1;
// Draws per frame before the epipolar check is skipped (OpenCV 5.0.0 USAC defect, SMOKE §4q). At the measured
// 31% per-draw failure rate on the worst frame, 3 draws all fail with probability ~3%.
constexpr int kUsacAttempts = 3;

bool InBorder(const cv::Point2f& p, cv::Size size, int border) {
    return p.x >= border && p.y >= border && p.x <= size.width - 1 - border &&
           p.y <= size.height - 1 - border;
}

cv::TermCriteria LkCriteria(const TrackerParams& p) {
    return {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, p.term_max_iterations,
            p.term_epsilon};
}

}  // namespace

EpipolarResult EpipolarInliers(const std::vector<cv::Point2f>& p0,
                               const std::vector<cv::Point2f>& p1,
                               const TrackerParams& params, uint64_t seed, uint64_t frame_id) {
    EpipolarResult out;
    if (p0.size() != p1.size() || static_cast<int>(p0.size()) < params.f_min_points) return out;
    // Every static point's two views satisfy x1^T F x0 = 0. A robust fit of F
    // to all correspondences flags those inconsistent with a single rigid
    // motion: mis-tracks, and points on moving objects. The threshold is in
    // pixels, valid because the input is rectified pinhole (no distortion).
    //
    // The UsacParams overload, NOT the FM_RANSAC one: that path samples from
    // cv::theRNG(), per-thread state that any earlier OpenCV call can advance,
    // so the same frame could reject different tracks depending on what ran
    // before it. Here the sampler is seeded from (seed, frame_id) and runs
    // single-threaded, so a replay reproduces the decision exactly.
    //
    // OpenCV 5.0.0 defect (SMOKE §4q): on some inputs USAC's F-degeneracy handling hands an EMPTY model to the
    // Sampson error and CV_Assert(!model.empty()) throws -- seed-dependent (V1_01 frame 1009: 62/200 seeds), not
    // the polisher, not LO. Only that throw is handled, so every frame that never threw is byte-identical: retry
    // with the next draws of the SAME Philox stream (still a pure function of seed, frame_id), and after
    // kUsacAttempts skip the check for this frame. Both are counted; nothing else is swallowed.
    core::PhiloxStream rng(seed, frame_id, kSiteTrackerUsac);
    cv::UsacParams usac;
    usac.threshold = params.f_ransac_threshold_px;
    usac.confidence = params.f_confidence;
    usac.maxIterations = params.f_max_iterations;
    usac.isParallel = false;
    std::vector<uchar> inlier;
    cv::Mat F;
    for (int attempt = 0; attempt < kUsacAttempts; ++attempt) {
        usac.randomGeneratorState = static_cast<int>(rng.NextBits32() & 0x7FFFFFFFu);
        try {
            F = cv::findFundamentalMat(p0, p1, inlier, usac);
            break;
        } catch (const cv::Exception& e) {
            if (e.code != cv::Error::StsAssert || e.err.find("!model.empty()") == std::string::npos) throw;
            ++out.usac_defect_retries;
            if (attempt + 1 == kUsacAttempts) {
                out.usac_defect_skipped = true;
                return out;
            }
        }
    }
    if (F.empty() || inlier.size() != p0.size()) return out;
    out.inlier = std::move(inlier);
    out.ran = true;
    return out;
}

KltTracker::KltTracker(const TrackerParams& params, const LocalFeatureEngine& replenisher)
    : params_(params), replenisher_(replenisher) {}

LkPyramid KltTracker::BuildPyramid(const cv::Mat& gray, const TrackerParams& params) {
    LkPyramid pyr;
    // withDerivatives=true precomputes the Scharr gradients LK needs at every
    // level, so the forward pass, backward pass and next frame's forward pass
    // all reuse them instead of each recomputing.
    pyr.max_level = cv::buildOpticalFlowPyramid(gray, pyr.images,
                                                cv::Size(params.win_size, params.win_size),
                                                params.max_level, /*withDerivatives=*/true);
    return pyr;
}

TrackedFrame KltTracker::Process(const cv::Mat& gray, uint64_t frame_id, uint64_t seed, const cv::Matx33d* predicted) {
    if (gray.empty() || gray.type() != CV_8UC1) {
        throw std::invalid_argument("KltTracker: expected a non-empty CV_8UC1 image");
    }
    const cv::Size size = gray.size();
    const LkPyramid pyramid = BuildPyramid(gray, params_);

    TrackedFrame out;
    out.frame_id = frame_id;
    TrackStats& stats = out.stats;

    // --- 1. Carry existing tracks forward ---------------------------------
    std::vector<size_t> kept;           // indices into points_
    std::vector<cv::Point2f> fwd;
    const bool have_previous = !prev_pyramid_.empty();
    if (have_previous && !points_.empty()) {
        stats.attempted = static_cast<int>(points_.size());
        const cv::Size win(params_.win_size, params_.win_size);
        const int levels = std::min(prev_pyramid_.max_level, pyramid.max_level);

        std::vector<uchar> st_fwd, st_bwd;
        std::vector<float> err;
        int flags = 0;
        if (predicted) {   // seed the search at the predicted position (2.4: gyro rotation homography)
            fwd.resize(points_.size());
            for (size_t i = 0; i < points_.size(); ++i) {
                const cv::Vec3d p = (*predicted) * cv::Vec3d(points_[i].x, points_[i].y, 1.0);
                fwd[i] = std::abs(p[2]) > 1e-12 ? cv::Point2f(float(p[0] / p[2]), float(p[1] / p[2])) : points_[i];
            }
            flags = cv::OPTFLOW_USE_INITIAL_FLOW;
        }
        cv::calcOpticalFlowPyrLK(prev_pyramid_.images, pyramid.images, points_, fwd, st_fwd,
                                 err, win, levels, LkCriteria(params_), flags,
                                 params_.min_eig_threshold);
        // Backward pass (Kalal et al. 2010, forward-backward error): track the
        // result back into the previous frame. A correct track returns to its
        // start; one that slid along an edge or jumped to a lookalike does
        // not. The previous positions are the initial guess, as in VINS-Fusion.
        std::vector<cv::Point2f> bwd = points_;
        cv::calcOpticalFlowPyrLK(pyramid.images, prev_pyramid_.images, fwd, bwd, st_bwd, err,
                                 win, levels, LkCriteria(params_),
                                 cv::OPTFLOW_USE_INITIAL_FLOW, params_.min_eig_threshold);

        const double fb2 = params_.fb_threshold_px * params_.fb_threshold_px;
        for (size_t i = 0; i < points_.size(); ++i) {
            if (!st_fwd[i] || !st_bwd[i]) {
                ++stats.lk_lost;
            } else if (cv::Point2d(bwd[i] - points_[i]).ddot(cv::Point2d(bwd[i] - points_[i])) >
                       fb2) {
                ++stats.fb_rejected;
            } else if (!InBorder(fwd[i], size, params_.border_px)) {
                ++stats.border_rejected;
            } else {
                kept.push_back(i);
            }
        }

        // --- 2. Epipolar outlier rejection --------------------------------
        std::vector<cv::Point2f> p0, p1;
        p0.reserve(kept.size());
        p1.reserve(kept.size());
        for (size_t i : kept) {
            p0.push_back(points_[i]);
            p1.push_back(fwd[i]);
        }
        const EpipolarResult epi = EpipolarInliers(p0, p1, params_, seed, frame_id);
        stats.usac_defect_retries = epi.usac_defect_retries;
        stats.usac_defect_skipped = epi.usac_defect_skipped;
        if (epi.ran) {
            stats.epipolar_check_ran = true;
            std::vector<size_t> survivors;
            for (size_t k = 0; k < kept.size(); ++k) {
                if (epi.inlier[k]) survivors.push_back(kept[k]);
                else ++stats.epipolar_rejected;
            }
            kept.swap(survivors);
        }
    }

    // --- 3. Spacing: older tracks win (VINS-Fusion's setMask) ---------------
    // Tracks that drift within min distance of an older one are dropped, and
    // the mask left behind tells the detector where NOT to start new tracks.
    std::stable_sort(kept.begin(), kept.end(), [this](size_t a, size_t b) {
        if (ages_[a] != ages_[b]) return ages_[a] > ages_[b];
        return ids_[a] < ids_[b];
    });
    cv::Mat mask(size, CV_8U, cv::Scalar(255));
    const int radius = static_cast<int>(std::lround(params_.min_track_distance_px));
    for (size_t i : kept) {
        const cv::Point2f& p = fwd[i];
        if (mask.at<uchar>(cvRound(p.y), cvRound(p.x)) == 0) {
            ++stats.crowded_out;
            continue;
        }
        out.ids.push_back(ids_[i]);
        out.points.push_back(p);
        out.prev_points.push_back(points_[i]);
        out.ages.push_back(ages_[i] + 1);
        cv::circle(mask, cv::Point(cvRound(p.x), cvRound(p.y)), radius, cv::Scalar(0), cv::FILLED);
    }
    const int survivors = static_cast<int>(out.ids.size());

    // --- 4. Replenish from the feature engine ------------------------------
    const int need = params_.target_tracks - survivors;
    if (need > 0) {
        const std::vector<cv::KeyPoint> fresh =
            replenisher_.Detect(gray, mask, need * params_.replenish_request_factor);
        for (const cv::KeyPoint& k : fresh) {
            if (static_cast<int>(out.ids.size()) >= params_.target_tracks) break;
            const cv::Point2f p = k.pt;
            if (!InBorder(p, size, params_.border_px)) continue;
            // The engine applies the mask at detection, but two new keypoints
            // can still sit next to each other; the mask is updated as they
            // are accepted so the min distance holds among them too.
            if (mask.at<uchar>(cvRound(p.y), cvRound(p.x)) == 0) continue;
            out.ids.push_back(next_id_++);
            out.points.push_back(p);
            out.prev_points.push_back(p);
            out.ages.push_back(0);
            cv::circle(mask, cv::Point(cvRound(p.x), cvRound(p.y)), radius, cv::Scalar(0), cv::FILLED);
            ++stats.replenished;
        }
    }

    if (!have_previous) {
        out.state = TrackingState::kInitializing;
    } else {
        out.state = survivors < params_.degraded_min_tracks ? TrackingState::kDegraded
                                                            : TrackingState::kTracking;
    }

    prev_pyramid_ = pyramid;
    ids_ = out.ids;
    points_ = out.points;
    ages_ = out.ages;
    return out;
}

}  // namespace ariadne::perception
