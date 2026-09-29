#pragma once
// ---------------------------------------------------------------------------
// §7.3 keypoint selection: "Do not keep the top N points globally. Use spatial
// bins: per cell, retain top-k by score."
//
// Why: a global top-N concentrates every keypoint on the one textured patch in
// view (a poster, a rebar mesh), and pose estimation from a single small patch
// is poorly conditioned. Spreading the budget over a grid trades a little
// per-point quality for geometry that constrains all six degrees of freedom.
//
// Two passes:
//   1. each cell keeps its best  floor(budget / cells)  keypoints;
//   2. budget left over (from cells with too few candidates) goes to the best
//      remaining keypoints anywhere, so an empty wall does not waste budget.
//
// DETERMINISM: ties in response are broken by (y, x, octave, angle), so the
// result depends only on the keypoint set, never on its input order or on the
// sort implementation. std::nth_element / KeyPointsFilter::retainBest make no
// such promise.
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>

#include <algorithm>
#include <vector>

namespace ariadne::perception {

/// Strict weak order: strongest first, then a total positional tie-break.
inline bool StrongerKeypoint(const cv::KeyPoint& a, const cv::KeyPoint& b) {
    if (a.response != b.response) return a.response > b.response;
    if (a.pt.y != b.pt.y) return a.pt.y < b.pt.y;
    if (a.pt.x != b.pt.x) return a.pt.x < b.pt.x;
    if (a.octave != b.octave) return a.octave < b.octave;
    return a.angle < b.angle;
}

/// Index of the grid cell containing `pt`, row-major.
inline int GridCell(const cv::Point2f& pt, cv::Size image, int cols, int rows) {
    const int cx = std::clamp(static_cast<int>(pt.x * cols / image.width), 0, cols - 1);
    const int cy = std::clamp(static_cast<int>(pt.y * rows / image.height), 0, rows - 1);
    return cy * cols + cx;
}

/// Select at most `budget` keypoints spread over a cols x rows grid.
/// Output is sorted strongest-first (StrongerKeypoint).
inline std::vector<cv::KeyPoint> GridSelect(std::vector<cv::KeyPoint> candidates,
                                            cv::Size image, int cols, int rows, int budget) {
    std::vector<cv::KeyPoint> out;
    if (budget <= 0 || candidates.empty()) return out;
    std::sort(candidates.begin(), candidates.end(), StrongerKeypoint);

    const int cells = cols * rows;
    const int per_cell = budget / cells;
    std::vector<int> taken(static_cast<size_t>(cells), 0);
    std::vector<char> used(candidates.size(), 0);

    // Pass 1: best per_cell keypoints of each cell. Candidates are already
    // strongest-first, so the first per_cell seen in a cell are its best.
    for (size_t i = 0; i < candidates.size(); ++i) {
        const int c = GridCell(candidates[i].pt, image, cols, rows);
        if (taken[static_cast<size_t>(c)] < per_cell) {
            ++taken[static_cast<size_t>(c)];
            used[i] = 1;
            out.push_back(candidates[i]);
        }
    }
    // Pass 2: spend what empty or sparse cells could not.
    for (size_t i = 0; i < candidates.size() && static_cast<int>(out.size()) < budget; ++i) {
        if (!used[i]) out.push_back(candidates[i]);
    }
    std::sort(out.begin(), out.end(), StrongerKeypoint);
    return out;
}

}  // namespace ariadne::perception
