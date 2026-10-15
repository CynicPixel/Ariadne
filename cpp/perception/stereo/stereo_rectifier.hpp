#pragma once
// ---------------------------------------------------------------------------
// Stereo rectification (arch §9.2): turn a calibrated, distorted stereo pair
// into the ideal pinhole pair that SGBM and the RectifiedStereoRig assume.
//
// WHAT RECTIFICATION DOES. Two real cameras are never perfectly parallel and
// their lenses bend straight lines. Rectification computes one rotation per
// camera (R1, R2) that makes both image planes coplanar with rows parallel to
// the baseline, plus a new shared pinhole projection (P1, P2). Undistortion
// and rotation are folded into one per-pixel lookup map, so at run time each
// frame costs a single cv::remap. Afterwards a scene point appears on the SAME
// row in both images, which is what reduces stereo matching to a 1D search
// along rows and makes d = x_left - x_right meaningful.
//
// Pattern as in ORB-SLAM3 (Settings::precomputeRectificationMaps) and
// Kimera-VIO (StereoCamera):
//   stereoRectify(K1, D1, K2, D2, size, R, T, ..., STEREO_ZERO_DISPARITY, alpha)
//   initUndistortRectifyMap(K_i, D_i, R_i, P_i(0:3, 0:3), size, ...)  once
//   remap(image, map_i)                                               per frame
// Choices:
//   alpha = 0: crop to pixels that are valid in both images. Kimera-VIO uses 0
//     because the default (-1) leaves black borders that produce spurious
//     keypoints (opencv/opencv#7240); for us they would also produce spurious
//     SGBM matches.
//   STEREO_ZERO_DISPARITY: both principal points at the same pixel, so zero
//     disparity means infinite depth (OpenCV 5's name; CALIB_ZERO_DISPARITY is
//     its deprecated synonym).
//   CV_16SC2 maps: fixed-point, OpenCV's fast remap path, and integer
//     arithmetic, so the rectified pixels are deterministic.
// remap is also one of the COOL benchmark's always-on per-frame stages (C4).
// ---------------------------------------------------------------------------
#include "perception/perception_config.hpp"
#include "perception/stereo/camera_model.hpp"

#include <opencv2/core.hpp>

namespace ariadne::perception {

class StereoRectifier {
public:
    /// alpha in [0, 1]: 0 keeps only valid pixels, 1 keeps every source pixel.
    explicit StereoRectifier(const RawStereoCalibration& calib, double alpha = 0.0);

    /// The ideal pinhole rig after rectification: fx, cx, cy from P1, and the
    /// baseline from P2 (P2(0,3) = -fx * B).
    const RectifiedStereoRig& Rig() const { return rig_; }

    /// Rectify one raw pair (CV_8UC1 or CV_8UC3, the calibration's size).
    void Rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& left_out,
                 cv::Mat& right_out) const;

    /// Rectify a left image only, with the same maps (e.g. the colour frame
    /// the reconstruction service needs alongside the gray VO input).
    void RectifyLeft(const cv::Mat& left, cv::Mat& left_out) const;

    /// Rectification outputs, for tests and the manifest.
    const cv::Mat& R1() const { return R1_; }
    const cv::Mat& R2() const { return R2_; }
    const cv::Mat& P1() const { return P1_; }
    const cv::Mat& P2() const { return P2_; }

private:
    RectifiedStereoRig rig_;
    cv::Size size_;
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat map_left_[2], map_right_[2];
};

}  // namespace ariadne::perception
