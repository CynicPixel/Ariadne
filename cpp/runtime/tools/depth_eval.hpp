#pragma once
// ---------------------------------------------------------------------------
// 5.5 (D148): stereo depth against ground truth at keyframes, binned by range -- the "range/depth error" of the
// §9.1 baseline sweep. Evaluation only: the GT depth comes from the recording's depth camera (co-located with the
// left camera, never shown to perception) and is read here, after the keyframe's SGBM depth exists.
//
//   in   the GT depth already on the RAW left camera grid (SimSource: the render's depth through the same lens,
//        perception::ApplyLensDepth), rectified with nearest-neighbour; the keyframe's SGBM Z and valid mask on
//        the rectified grid; K_rect and R1 (rectified <- raw)
//   GT   a rectified pixel's ray d = K_rect^-1 [u v 1]^T (d.z = 1) is (R1^T d) in the raw camera, so the raw depth
//        Z_raw at that pixel means Z_rect = Z_raw / (R1^T d).z
//   out  per range bin [0.3,1) [1,2) [2,4) [4,8) m of GT depth: GT pixels, the fraction SGBM made valid, and the
//        median / p90 of |Z - Z_gt| / Z_gt over the valid ones (every 2nd pixel in both directions)
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>
#include <vector>

namespace ariadne::tools {

class KeyframeDepthEval {
public:
    static constexpr std::array<double, 5> kEdges{0.3, 1.0, 2.0, 4.0, 8.0};

    /// gt_raw_on_rect: CV_32F raw-camera depth sampled at each rectified pixel (0 / NaN = none). Z: CV_32F or CV_16F.
    void Add(const cv::Mat& gt_raw_on_rect, const cv::Mat& Z, const cv::Mat& valid, const cv::Matx33d& K_rect,
             const cv::Matx33d& R1) {
        cv::Mat z;
        Z.convertTo(z, CV_32F);
        const cv::Matx33d Rt = R1.t();
        for (int v = 0; v < z.rows; v += 2) {
            for (int u = 0; u < z.cols; u += 2) {
                const float g = gt_raw_on_rect.at<float>(v, u);
                if (!(g > 0) || !std::isfinite(g)) continue;
                const cv::Vec3d d((u - K_rect(0, 2)) / K_rect(0, 0), (v - K_rect(1, 2)) / K_rect(1, 1), 1.0);
                const double zr = (Rt * d)[2];
                if (!(zr > 0)) continue;
                const double zg = g / zr;
                const int b = Bin(zg);
                if (b < 0) continue;
                ++gt_[b];
                const float zs = z.at<float>(v, u);
                if (valid.at<uchar>(v, u) && zs > 0 && std::isfinite(zs)) rel_[b].push_back(std::abs(zs - zg) / zg);
            }
        }
        ++keyframes_;
    }

    static int Bin(double z) {
        for (size_t i = 0; i + 1 < kEdges.size(); ++i)
            if (z >= kEdges[i] && z < kEdges[i + 1]) return int(i);
        return -1;
    }

    /// results.yaml block (nothing when no keyframe was evaluated).
    void WriteResults(std::ostream& out) {
        if (!keyframes_) return;
        out << "depth_eval: {keyframes: " << keyframes_ << ", pixel_stride: 2, bins: [";
        for (size_t b = 0; b + 1 < kEdges.size(); ++b) {
            auto& r = rel_[b];
            double med = NAN, p90 = NAN;
            if (!r.empty()) {
                std::sort(r.begin(), r.end());
                med = r[r.size() / 2];
                p90 = r[std::min(r.size() - 1, size_t(0.9 * double(r.size())))];
            }
            out << (b ? ", " : "") << "{z_m: [" << kEdges[b] << ", " << kEdges[b + 1] << "], gt_pixels: " << gt_[b]
                << ", valid_fraction: " << (gt_[b] ? double(r.size()) / double(gt_[b]) : 0.0)
                << ", rel_err_median: " << med << ", rel_err_p90: " << p90 << "}";
        }
        out << "]}\n";
    }

private:
    std::array<long, 4> gt_{};
    std::array<std::vector<float>, 4> rel_;
    long keyframes_ = 0;
};

}  // namespace ariadne::tools
