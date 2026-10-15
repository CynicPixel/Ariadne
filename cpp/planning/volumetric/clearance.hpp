#pragma once
// ---------------------------------------------------------------------------
// 2D clearance (arch §20.3/§20.5 placeholder; design MISSION_1.8): distance from
// every cell CENTRE to the nearest EDGE of a blocked square, the convention of the
// scene ground truth (python/dataset_tools/worldgen/raster.py `_edt_to` / `esdf` /
// `traversable`), so a perfect map reproduces GT traversability cell for cell:
//
//   per axis an offset of k cells contributes max(|k| - 0.5, 0) (0 inside),
//   d = sqrt(dx_edge^2 + dy_edge^2), min over blocked squares, capped at cap cells;
//   traversable(x)  <=>  free(x) and d(x) * cell >= rho - 1e-9.
//
// Exact within the cap: a vertical pass finds the nearest blocked row per column,
// then a horizontal sweep over offsets |k| <= cap + 1 combines them (the same
// two-pass structure as worldgen). cv::distanceTransform measures centre to
// centre, which would differ by up to half a cell along every wall.
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ariadne::planning {

constexpr double kTravEpsM = 1e-9;   // worldgen TRAV_EPS_M: equality must not hinge on one ulp

/// Distance in metres from each cell centre to the nearest blocked square (0 for blocked cells),
/// capped at cap_m. `blocked` is CV_8U (nonzero = blocked). Returns CV_64F.
inline cv::Mat EdgeClearance(const cv::Mat& blocked, double cell_m, double cap_m) {
    if (blocked.type() != CV_8U) throw std::invalid_argument("EdgeClearance: CV_8U mask");
    const int H = blocked.rows, W = blocked.cols;
    const int cap = int(std::ceil(cap_m / cell_m - 1e-12));
    const double big = std::numeric_limits<double>::infinity();
    // Vertical pass: edge distance (cells) to the nearest blocked cell in the same column.
    cv::Mat dy(H, W, CV_64F, cv::Scalar(big));
    for (int x = 0; x < W; ++x) {
        int last = -1;
        std::vector<int> up(size_t(H), -1);
        for (int y = 0; y < H; ++y) {
            if (blocked.at<uint8_t>(y, x)) last = y;
            up[size_t(y)] = last;
        }
        int next = -1;
        for (int y = H - 1; y >= 0; --y) {
            if (blocked.at<uint8_t>(y, x)) next = y;
            double d = big;
            if (up[size_t(y)] >= 0) d = std::min(d, double(y - up[size_t(y)]));
            if (next >= 0) d = std::min(d, double(next - y));
            const double edge = d == 0 ? 0.0 : (std::isinf(d) ? big : std::max(d - 0.5, 0.0));
            dy.at<double>(y, x) = std::min(edge, double(cap) + 1.0);
        }
    }
    // Horizontal sweep over column offsets.
    cv::Mat out(H, W, CV_64F);
    const int kmax = std::min(cap + 1, W - 1);
    for (int y = 0; y < H; ++y) {
        const double* r = dy.ptr<double>(y);
        double* o = out.ptr<double>(y);
        for (int x = 0; x < W; ++x) {
            double best = big;
            for (int k = -kmax; k <= kmax; ++k) {
                const int xx = x + k;
                if (xx < 0 || xx >= W || std::isinf(r[xx])) continue;
                const double dx = k == 0 ? 0.0 : std::abs(k) - 0.5;
                best = std::min(best, dx * dx + r[xx] * r[xx]);
            }
            o[x] = std::min(std::sqrt(best), double(cap)) * cell_m;
        }
    }
    return out;
}

/// traversable = free & clearance >= rho (worldgen's rule). CV_8U 0/1.
inline cv::Mat Traversable(const cv::Mat& free_mask, const cv::Mat& clearance_m, double rho_m) {
    cv::Mat t(free_mask.size(), CV_8U);
    for (int y = 0; y < t.rows; ++y) {
        for (int x = 0; x < t.cols; ++x) {
            t.at<uint8_t>(y, x) =
                uint8_t(free_mask.at<uint8_t>(y, x) && clearance_m.at<double>(y, x) >= rho_m - kTravEpsM);
        }
    }
    return t;
}

}  // namespace ariadne::planning
