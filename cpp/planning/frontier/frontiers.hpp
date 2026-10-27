#pragma once
// ---------------------------------------------------------------------------
// Frontiers (arch §20.4 placeholder; Yamauchi 1997, design mission.md): FREE
// cells 4-adjacent to UNKNOWN, grouped by 8-connected components
// (cv::connectedComponentsWithStats). Clusters smaller than `min_cells` are
// dropped (pinholes left by the ray stride) and counted. The nearest-frontier
// policy (§21 placeholder) picks the cluster with the shortest grid path from the
// agent; in replay its choice is logged, never executed.
// ---------------------------------------------------------------------------
#include "planning/route/astar.hpp"
#include "planning/volumetric/occupancy_grid.hpp"

#include <opencv2/imgproc.hpp>

#include <vector>

namespace ariadne::planning {

struct FrontierCluster {
    int id = 0;
    int cells = 0;
    double centroid_r = 0, centroid_c = 0;   // cell units
    std::vector<Cell> members;
};

struct FrontierSet {
    std::vector<FrontierCluster> clusters;   // kept, by component label order (deterministic)
    int rejected_small = 0;
    cv::Mat mask;                            // CV_8U 1 = frontier cell (all, before the size gate)
};

inline FrontierSet Frontiers(const cv::Mat& slab, int min_cells) {
    FrontierSet fs;
    fs.mask = cv::Mat::zeros(slab.size(), CV_8U);
    const int H = slab.rows, W = slab.cols;
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (slab.at<uint8_t>(r, c) != kFree) continue;
            const bool fr = (r > 0 && slab.at<uint8_t>(r - 1, c) == kUnknown) ||
                            (r + 1 < H && slab.at<uint8_t>(r + 1, c) == kUnknown) ||
                            (c > 0 && slab.at<uint8_t>(r, c - 1) == kUnknown) ||
                            (c + 1 < W && slab.at<uint8_t>(r, c + 1) == kUnknown);
            fs.mask.at<uint8_t>(r, c) = fr;
        }
    }
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(fs.mask, labels, stats, centroids, 8, CV_32S);
    std::vector<FrontierCluster> all(size_t(std::max(n, 1)));
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            const int l = labels.at<int>(r, c);
            if (l > 0) all[size_t(l)].members.push_back({r, c});
        }
    }
    for (int l = 1; l < n; ++l) {
        FrontierCluster& k = all[size_t(l)];
        k.id = l;
        k.cells = stats.at<int>(l, cv::CC_STAT_AREA);
        k.centroid_c = centroids.at<double>(l, 0);
        k.centroid_r = centroids.at<double>(l, 1);
        if (k.cells < min_cells) {
            ++fs.rejected_small;
            continue;
        }
        fs.clusters.push_back(std::move(k));
    }
    return fs;
}

}  // namespace ariadne::planning
