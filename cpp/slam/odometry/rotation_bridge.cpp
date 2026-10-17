#include "slam/odometry/rotation_bridge.hpp"

#include "core/random/philox.hpp"

#include <opencv2/geometry.hpp>

#include <Eigen/SVD>
#include <cmath>
#include <functional>

namespace ariadne::slam {

namespace {

// Philox sites, counted down from the top of the 48-bit domain (tracker kMaxSite - 1, PnP kMaxSite - 2): H and E,
// frame-to-frame (- 3, - 4) and from the anchor (- 5, - 6).
constexpr uint64_t kSiteBridgeH = core::PhiloxStream::kMaxSite - 3;
constexpr uint64_t kSiteBridgeE = core::PhiloxStream::kMaxSite - 4;
constexpr uint64_t kSiteAnchorH = core::PhiloxStream::kMaxSite - 5;
constexpr uint64_t kSiteAnchorE = core::PhiloxStream::kMaxSite - 6;
constexpr int kUsacAttempts = 3;
constexpr double kNoDistanceCut = 1e12;   // recoverPose distanceThresh, in baselines: count every point

// Run one seeded USAC estimate, absorbing ONLY the OpenCV 5.0.0 `!model.empty()` assertion by redrawing from the same
// Philox stream (klt_tracker.cpp has the full story). Returns an empty Mat when every attempt hit the defect.
cv::Mat SeededUsac(const std::function<cv::Mat(const cv::UsacParams&)>& estimate, cv::UsacParams usac,
                   core::PhiloxStream rng, int& retries) {
    for (int attempt = 0; attempt < kUsacAttempts; ++attempt) {
        usac.randomGeneratorState = static_cast<int>(rng.NextBits32() & 0x7FFFFFFFu);
        try {
            return estimate(usac);
        } catch (const cv::Exception& e) {
            if (e.code != cv::Error::StsAssert || e.err.find("!model.empty()") == std::string::npos) throw;
            ++retries;
        }
    }
    return cv::Mat();
}

core::Mat3 ToMat3(const cv::Mat& m) {
    core::Mat3 R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R(r, c) = m.at<double>(r, c);
    return R;
}

// The rotation nearest (Frobenius) to M: U V^T, with the determinant fix.
core::Mat3 NearestRotation(const core::Mat3& M) {
    Eigen::JacobiSVD<core::Mat3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    core::Mat3 S = core::Mat3::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) S(2, 2) = -1.0;
    return svd.matrixU() * S * svd.matrixV().transpose();
}

int Count(const cv::Mat& mask) { return mask.empty() ? 0 : cv::countNonZero(mask); }

// ORB-SLAM's model scores (Mur-Artal et al., T-RO 2015, §IV-B; Initializer::CheckHomography / CheckFundamental):
// over every correspondence and BOTH images, a squared error e = d^2 / sigma^2 adds (Gamma - e) when e < T_M. Gamma =
// 5.991 for both models so the scores are comparable; T_H = 5.991 (chi^2, 2 dof: transfer error), T_F = 3.841 (1 dof:
// distance to the epipolar line); sigma = 1 px. Distances are measured in normalised coordinates and scaled by fx.
constexpr double kGamma = 5.991, kThH = 5.991, kThF = 3.841;

double Contribution(double e, double th) { return e < th ? kGamma - e : 0.0; }

double ScoreH(const core::Mat3& H, const std::vector<cv::Point2d>& n0, const std::vector<cv::Point2d>& n1, double s2) {
    const core::Mat3 Hi = H.inverse();
    double score = 0.0;
    for (size_t i = 0; i < n0.size(); ++i) {
        const core::Vec3 a(n0[i].x, n0[i].y, 1.0), b(n1[i].x, n1[i].y, 1.0);
        const core::Vec3 ab = H * a, ba = Hi * b;
        score += Contribution((b.head<2>() - ab.head<2>() / ab.z()).squaredNorm() * s2, kThH);
        score += Contribution((a.head<2>() - ba.head<2>() / ba.z()).squaredNorm() * s2, kThH);
    }
    return score;
}

double ScoreF(const core::Mat3& E, const std::vector<cv::Point2d>& n0, const std::vector<cv::Point2d>& n1, double s2) {
    double score = 0.0;
    for (size_t i = 0; i < n0.size(); ++i) {
        const core::Vec3 a(n0[i].x, n0[i].y, 1.0), b(n1[i].x, n1[i].y, 1.0);
        const core::Vec3 l1 = E * a, l0 = E.transpose() * b;   // epipolar lines in image 1 and image 0
        const double r = b.dot(l1);                            // = a . l0
        score += Contribution(r * r / l1.head<2>().squaredNorm() * s2, kThF);
        score += Contribution(r * r / l0.head<2>().squaredNorm() * s2, kThF);
    }
    return score;
}

}  // namespace

FrameRotation EstimateFrameRotation(const std::vector<cv::Point2f>& p0, const std::vector<cv::Point2f>& p1,
                                    const core::PinholeCamera& cam, const RotationBridgeParams& params, uint64_t seed,
                                    uint64_t frame_id, BridgeStream stream) {
    const bool anchor = stream == BridgeStream::kFromAnchor;
    FrameRotation out;
    if (p0.size() != p1.size()) throw std::invalid_argument("EstimateFrameRotation: p0/p1 sizes differ");
    out.tracks = static_cast<int>(p0.size());
    if (out.tracks < params.min_tracks) {
        out.why = "too few tracks";
        return out;
    }
    // Normalised coordinates: K = I, so H is R itself (pure rotation) and E needs no camera matrix.
    std::vector<cv::Point2d> n0, n1;
    n0.reserve(p0.size());
    n1.reserve(p1.size());
    for (size_t i = 0; i < p0.size(); ++i) {
        n0.emplace_back((p0[i].x - cam.cx) / cam.fx, (p0[i].y - cam.cy) / cam.fy);
        n1.emplace_back((p1[i].x - cam.cx) / cam.fx, (p1[i].y - cam.cy) / cam.fy);
    }
    cv::UsacParams usac;
    usac.threshold = params.threshold_px / cam.fx;
    usac.confidence = params.confidence;
    usac.maxIterations = params.max_iterations;
    usac.isParallel = false;

    // x1 ~ R_10 x0 under pure rotation: H (det normalised to 1) projected to SO(3) is R_10.
    cv::Mat mask_h, mask_e;
    const cv::Mat H = SeededUsac([&](const cv::UsacParams& u) { return cv::findHomography(n0, n1, mask_h, u); }, usac,
                                 core::PhiloxStream(seed, frame_id, anchor ? kSiteAnchorH : kSiteBridgeH), out.usac_retries);
    const double s2 = cam.fx * cam.fx / (params.sigma_px * params.sigma_px);   // normalised^2 -> (px / sigma)^2
    bool have_h = false;
    core::Mat3 R_h = core::Mat3::Identity();
    if (H.rows == 3 && H.cols == 3) {
        core::Mat3 Hm = ToMat3(H);
        const double det = Hm.determinant();
        if (std::isfinite(det) && std::abs(det) > 1e-12) {
            R_h = NearestRotation(Hm / std::cbrt(det));
            out.inliers_h = Count(mask_h);
            out.score_h = ScoreH(Hm, n0, n1, s2);
            have_h = true;
        }
    }
    const cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat E = SeededUsac(
        [&](const cv::UsacParams& u) { return cv::findEssentialMat(n0, n1, I, I, cv::noArray(), cv::noArray(), mask_e, u); },
        usac, core::PhiloxStream(seed, frame_id, anchor ? kSiteAnchorE : kSiteBridgeE), out.usac_retries);
    bool have_e = false;
    core::Mat3 R_e = core::Mat3::Identity();
    if (E.rows >= 3 && E.cols == 3) {
        cv::Mat R, t, m = mask_e.clone();
        // recoverPose keeps the decomposition with the most points in front of both cameras; X1 = R X0 + t.
        // recoverPose picks the decomposition with the most points in front of both cameras. Its default overload
        // counts only points within 50 baselines -- NONE when consecutive frames are 1-2 cm apart in a 1-6 m room, so
        // it returned 0 and E was lost (synthetic forward motion): every point votes here. The count is a diagnostic.
        out.cheiral_e = cv::recoverPose(E.rowRange(0, 3), n0, n1, I, R, t, kNoDistanceCut, m);
        out.inliers_e = Count(mask_e);
        if (out.cheiral_e > 0) {
            R_e = ToMat3(R);
            out.score_f = ScoreF(ToMat3(E.rowRange(0, 3)), n0, n1, s2);
            have_e = true;
        }
    }
    // Physical bound: consecutive frames rotated by more than half the horizontal FOV share no tracks.
    const double max_angle = std::atan2(cam.cx, cam.fx);
    const auto plausible = [&](const core::Mat3& R) { return core::SO3Log(R).norm() <= max_angle; };
    have_h = have_h && out.inliers_h >= 8 && plausible(R_h);
    have_e = have_e && out.inliers_e >= 8 && plausible(R_e);
    if (have_e && have_h) out.disagreement_deg = core::SO3Log(R_e.transpose() * R_h).norm() * 180.0 / M_PI;
    const bool prefer_e = have_e && (!have_h || out.score_h / (out.score_h + out.score_f) <= params.model_ratio);
    if (prefer_e) {
        out.R_prev_cur = R_e.transpose();
        out.model = 'E';
    } else if (have_h) {
        out.R_prev_cur = R_h.transpose();
        out.model = 'H';
    } else {
        out.why = "no plausible H or E";
        return out;
    }
    out.ok = true;
    return out;
}

}  // namespace ariadne::slam
