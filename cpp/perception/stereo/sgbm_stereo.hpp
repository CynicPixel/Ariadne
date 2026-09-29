#pragma once
// ---------------------------------------------------------------------------
// Dense stereo depth with semi-global matching (arch §9.3-9.6).
//
// SGM (Hirschmueller, TPAMI 2008) in one paragraph: for every pixel and every
// candidate disparity d, a matching cost says how different the left pixel is
// from the right pixel d columns over. Picking the cheapest d per pixel alone
// is noisy. SGM adds a smoothness prior: changing disparity by 1 between
// neighbours costs P1, by more costs P2 > P1 (small steps on slanted surfaces
// are cheap, jumps only at object edges). Minimising that over the whole 2D
// image is NP-hard, so SGM aggregates the cost along 8 (or fewer) 1D scanline
// directions with dynamic programming and sums them. MODE_SGBM_3WAY is
// OpenCV's faster 3-direction variant (§9.3 picks it for speed).
//
// Output contract (§9.6) with one recorded deviation (DECISIONS D34):
// `disparity` stays in SGBM's native fixed point (CV_16S, units of 1/16 px),
// NOT fp16. fp16 spacing is 0.125 px above 128 px, coarser than SGBM's 1/16,
// and at f=640 px, B=0.10 m that is every depth under 0.5 m: the confined-space
// range the project exists for. CV_16S is exact and still 2 bytes/pixel.
// Z and sigma_Z are computed in float32 and stored as fp16 (relative precision
// ~5e-4, far below sigma_Z itself).
//
// `confidence` is a [PLACEHOLDER]: all zero, and `confidence_is_placeholder`
// says so. §9.3 requires the DisparityWLSFilter confidence map (Phase 2) and
// forbids inventing a heuristic; zero rather than 255 so that a consumer that
// reads it by mistake discards the depth instead of trusting it.
// ---------------------------------------------------------------------------
#include "perception/perception_config.hpp"

#include <opencv2/core.hpp>
#include <opencv2/stereo.hpp>

namespace ariadne::perception {

struct StereoDepth {
    cv::Mat disparity;    // CV_16S, 1/16 px (D34); invalid pixels hold the SGBM invalid value
    cv::Mat Z;            // CV_16F, metres; 0 where invalid
    cv::Mat sigma_Z;      // CV_16F, metres; 0 where invalid (§9.5)
    cv::Mat confidence;   // CV_8U  [PLACEHOLDER] zeros until WLS (Phase 2)
    cv::Mat valid;        // CV_8U  255 = valid depth
    bool confidence_is_placeholder = true;
    double valid_fraction = 0.0;
    /// Columns [0, blind_columns) can never be valid: their match would lie
    /// left of the right image's first column. Reported, not hidden.
    int blind_columns = 0;
};

/// Depths with dense local support (D34). SGBM reports surfaces nearer than
/// its search range as sparse blobs of phantom FAR depth, indistinguishable
/// by value; they are distinguishable by density. Measured on synthetic
/// planes: true depth has 31x31 valid density >= 0.84 (1st percentile), the
/// aliases <= 0.42. A consumer that carves free space along depth rays
/// (occupancy, 1.8) must carve only through SUPPORTED depth, or one alias
/// carves free space through a real obstacle.
/// Returns CV_8U, 255 = valid AND supported.
cv::Mat DenseSupportMask(const StereoDepth& depth, const StereoParams& params);

class SgbmStereo {
public:
    /// Validates the params against the rig (§9.4) and throws ConfigError on a
    /// combination with a blind zone past required_min_range_m.
    SgbmStereo(const StereoParams& params, const RectifiedStereoRig& rig);

    /// `left`, `right`: rectified CV_8UC1 images of the rig's size.
    StereoDepth Compute(const cv::Mat& left, const cv::Mat& right) const;

    /// §9.5: sigma_Z = Z^2 / (f B) * sigma_d.
    static double DepthSigma(double z, double fx, double baseline, double sigma_d) {
        return z * z / (fx * baseline) * sigma_d;
    }

private:
    StereoParams params_;
    RectifiedStereoRig rig_;
    cv::Ptr<cv::StereoSGBM> sgbm_;
};

}  // namespace ariadne::perception
