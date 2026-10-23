// 5.5 (D148): the depth_eval block -- GT depth on the raw grid, rectified, converted to the rectified camera's Z, and
// the SGBM error binned by GT range.
#include "runtime/tools/depth_eval.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>

using ariadne::tools::KeyframeDepthEval;

TEST(DepthEval, RectifiedZFromRawDepthAndBinnedRelativeError) {
    // A plane at raw Z = 1.5 m seen through a rectification rotated 3 degrees about y: the rectified Z is
    // Z_raw / (R1^T d).z per pixel. SGBM reads it 1 % long on the left half and is invalid on the right half.
    const int w = 64, h = 48;
    const cv::Matx33d K(40, 0, 31.5, 0, 40, 23.5, 0, 0, 1);
    const double a = 3.0 * M_PI / 180.0;
    const cv::Matx33d R1(std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a));
    cv::Mat gt(h, w, CV_32F, cv::Scalar(1.5f)), Z(h, w, CV_32F), valid(h, w, CV_8U, cv::Scalar(0));
    for (int v = 0; v < h; ++v)
        for (int u = 0; u < w; ++u) {
            const cv::Vec3d d((u - K(0, 2)) / K(0, 0), (v - K(1, 2)) / K(1, 1), 1.0);
            Z.at<float>(v, u) = float(1.01 * 1.5 / (R1.t() * d)[2]);
            if (u < w / 2) valid.at<uchar>(v, u) = 255;
        }
    gt.at<float>(0, 0) = std::nanf("");   // no GT there: skipped
    KeyframeDepthEval e;
    e.Add(gt, Z, valid, K, R1);
    std::ostringstream os;
    e.WriteResults(os);
    const std::string r = os.str();
    // Everything lands in [1, 2): 767 GT pixels (32x24 sampled, minus the NaN one, which sat in the valid half: 383
    // valid = 0.4993), 1 % error up to float rounding (0.00999999).
    EXPECT_NE(r.find("{z_m: [1, 2], gt_pixels: 767, valid_fraction: 0.4993"), std::string::npos) << r;
    EXPECT_NE(r.find("rel_err_median: 0.0099999"), std::string::npos) << r;
    EXPECT_NE(r.find("{z_m: [0.3, 1], gt_pixels: 0,"), std::string::npos) << r;
    EXPECT_EQ(KeyframeDepthEval::Bin(0.29), -1);
    EXPECT_EQ(KeyframeDepthEval::Bin(7.99), 3);
    std::ostringstream none;
    KeyframeDepthEval().WriteResults(none);
    EXPECT_TRUE(none.str().empty());   // off: no block
}
