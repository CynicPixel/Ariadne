#pragma once
// ---------------------------------------------------------------------------
// Trajectory error metrics (arch §30.1, §30.2).
//
// An estimated trajectory lives in its own arbitrary reference frame, so
// comparing it to ground truth without alignment measures the frame offset
// rather than the estimator. Umeyama (TPAMI 1991) supplies the optimal
// alignment, and is what the `evo` toolchain uses, which is why `evo` can serve
// as the external oracle §30.1 asks for.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "core/trajectory/synthetic.hpp"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <vector>

namespace ariadne::runtime {

using core::Mat3;
using core::SE3;
using core::TrajectorySample;
using core::Vec3;

struct AlignmentResult {
    SE3 transform;          // maps source into the target frame
    double scale = 1.0;     // 1.0 unless a Sim(3) alignment was requested
    bool valid = false;
};

/// Umeyama alignment of `src` onto `dst`.
///
/// `with_scale` selects Sim(3) instead of SE(3). Stereo makes scale observable
/// per agent (C2), so SE(3) is correct for the production estimator; the scale
/// variant exists for the monocular ablation, where scale is not observable and
/// leaving it out would report an error that is really a units mismatch.
inline AlignmentResult UmeyamaAlign(const std::vector<Vec3>& src,
                                    const std::vector<Vec3>& dst,
                                    bool with_scale = false) {
    AlignmentResult result;
    if (src.size() != dst.size() || src.size() < 3) return result;
    const int n = static_cast<int>(src.size());
    const double inv_n = 1.0 / static_cast<double>(n);

    Vec3 mu_src = Vec3::Zero(), mu_dst = Vec3::Zero();
    for (int i = 0; i < n; ++i) { mu_src += src[i]; mu_dst += dst[i]; }
    mu_src *= inv_n;
    mu_dst *= inv_n;

    Mat3 sigma = Mat3::Zero();
    double var_src = 0.0;
    for (int i = 0; i < n; ++i) {
        const Vec3 a = src[i] - mu_src;
        const Vec3 b = dst[i] - mu_dst;
        sigma += b * a.transpose();
        var_src += a.squaredNorm();
    }
    sigma *= inv_n;
    var_src *= inv_n;

    Eigen::JacobiSVD<Mat3> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // THE DETERMINANT FIX. An unconstrained SVD solution can be a REFLECTION,
    // which fits the points at least as well as any rotation but is not a
    // physical rigid motion. Flipping the sign on the least-significant
    // singular direction yields the best proper rotation. Omitting this gives
    // a mirrored trajectory with a deceptively small residual.
    Mat3 s = Mat3::Identity();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
        s(2, 2) = -1.0;
    }

    result.transform.R = svd.matrixU() * s * svd.matrixV().transpose();
    if (with_scale && var_src > 0.0) {
        result.scale = (svd.singularValues().asDiagonal() * s).trace() / var_src;
    }
    result.transform.t = mu_dst - result.scale * result.transform.R * mu_src;
    result.valid = true;
    return result;
}

struct ErrorStats {
    double rmse = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double max = 0.0;
    size_t count = 0;
};

inline ErrorStats Summarize(std::vector<double> errors) {
    ErrorStats s;
    if (errors.empty()) return s;
    s.count = errors.size();
    double sum = 0.0, sq = 0.0;
    for (double e : errors) { sum += e; sq += e * e; s.max = std::max(s.max, e); }
    s.mean = sum / static_cast<double>(errors.size());
    s.rmse = std::sqrt(sq / static_cast<double>(errors.size()));
    std::sort(errors.begin(), errors.end());
    s.median = errors[errors.size() / 2];
    return s;
}

/// Align an estimated trajectory onto ground truth by its POSITIONS.
///
/// Both ATE components below use this one alignment, so the translational and
/// rotational numbers always describe the same estimate-to-GT registration and
/// can never be computed against two different frames.
inline AlignmentResult AlignTrajectories(const std::vector<TrajectorySample>& est,
                                         const std::vector<TrajectorySample>& gt,
                                         bool with_scale) {
    if (est.size() != gt.size() || est.empty()) return AlignmentResult{};
    std::vector<Vec3> src, dst;
    src.reserve(est.size());
    dst.reserve(gt.size());
    for (size_t i = 0; i < est.size(); ++i) {
        src.push_back(est[i].pose.t);
        dst.push_back(gt[i].pose.t);
    }
    return UmeyamaAlign(src, dst, with_scale);
}

/// Absolute Trajectory Error: align, then take per-pose translation residuals.
/// Measures global consistency, so drift accumulated over the run shows up here.
inline ErrorStats ComputeATE(const std::vector<TrajectorySample>& est,
                             const std::vector<TrajectorySample>& gt,
                             bool with_scale = false) {
    const AlignmentResult a = AlignTrajectories(est, gt, with_scale);
    if (!a.valid) return ErrorStats{};

    std::vector<double> errors;
    errors.reserve(est.size());
    for (size_t i = 0; i < est.size(); ++i) {
        const Vec3 mapped = a.scale * (a.transform.R * est[i].pose.t) + a.transform.t;
        errors.push_back((mapped - gt[i].pose.t).norm());
    }
    return Summarize(std::move(errors));
}

/// Rotational part of the absolute error (arch §30.2 "rotation error").
///
/// After the SAME position-based alignment as ComputeATE, the per-pose error is
/// the angle of R_gt^T * (R_align * R_est). This is the convention of evo's APE
/// "rotation part". Translation-only ATE cannot see an estimator whose
/// positions are right but whose orientation is wrong, and orientation is what
/// the pose graph, the NBV planner and the splat renderer all consume.
///
/// Scale does not enter: a Sim(3) alignment scales positions, not rotations.
inline ErrorStats ComputeATERotation(const std::vector<TrajectorySample>& est,
                                     const std::vector<TrajectorySample>& gt,
                                     bool with_scale = false) {
    const AlignmentResult a = AlignTrajectories(est, gt, with_scale);
    if (!a.valid) return ErrorStats{};

    std::vector<double> errors;
    errors.reserve(est.size());
    for (size_t i = 0; i < est.size(); ++i) {
        const Mat3 aligned = a.transform.R * est[i].pose.R;
        errors.push_back(core::SO3Log(gt[i].pose.R.transpose() * aligned).norm());
    }
    return Summarize(std::move(errors));
}

struct RelativeError {
    ErrorStats translation;   // metres per delta
    ErrorStats rotation;      // radians per delta
};

/// Relative Pose Error over a fixed keyframe delta.
///
/// RPE needs no alignment: it compares relative motions, which are already
/// frame-independent. It measures local consistency, so a locally accurate
/// estimator that drifts globally scores well on RPE and poorly on ATE. The
/// two together separate drift from local error, which is why §30.2 wants both.
inline RelativeError ComputeRPE(const std::vector<TrajectorySample>& est,
                                const std::vector<TrajectorySample>& gt,
                                size_t delta = 1) {
    RelativeError out;
    if (est.size() != gt.size() || est.size() <= delta) return out;

    std::vector<double> trans, rot;
    for (size_t i = 0; i + delta < est.size(); ++i) {
        const SE3 est_rel = est[i].pose.inverse() * est[i + delta].pose;
        const SE3 gt_rel = gt[i].pose.inverse() * gt[i + delta].pose;
        const SE3 error = gt_rel.inverse() * est_rel;
        trans.push_back(error.t.norm());
        rot.push_back(core::SO3Log(error.R).norm());
    }
    out.translation = Summarize(std::move(trans));
    out.rotation = Summarize(std::move(rot));
    return out;
}

}  // namespace ariadne::runtime
