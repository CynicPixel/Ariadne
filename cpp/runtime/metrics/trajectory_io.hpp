#pragma once
// ---------------------------------------------------------------------------
// Trajectory association and TUM export, matching evo (the 1.9 oracle):
// nearest-timestamp association with max |dt| = 0.01 s (evo.core.sync
// associate_trajectories default), and the TUM format
// "timestamp tx ty tz qx qy qz qw", space-separated, no trailing delimiter
// (evo.tools.file_interface.read_tum_trajectory_file).
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <Eigen/Geometry>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ariadne::runtime {

inline constexpr int64_t kEvoDefaultMaxDiffNs = 10'000'000;   // 0.01 s

struct TimedPose {
    int64_t t_ns = 0;
    core::SE3 T;
};

/// For each `est` pose, the index of the nearest `ref` pose within max_diff,
/// or -1. `ref` must be sorted by time.
inline std::vector<int> AssociateNearest(const std::vector<TimedPose>& est, const std::vector<TimedPose>& ref,
                                         int64_t max_diff_ns = kEvoDefaultMaxDiffNs) {
    std::vector<int> idx(est.size(), -1);
    size_t j = 0;
    for (size_t i = 0; i < est.size(); ++i) {
        while (j + 1 < ref.size() && ref[j + 1].t_ns <= est[i].t_ns) ++j;
        int best = -1;
        int64_t best_d = max_diff_ns + 1;
        for (size_t k = j; k < std::min(ref.size(), j + 2); ++k) {
            const int64_t d = std::llabs(ref[k].t_ns - est[i].t_ns);
            if (d < best_d) { best_d = d; best = static_cast<int>(k); }
        }
        if (best >= 0 && best_d <= max_diff_ns) idx[i] = best;
    }
    return idx;
}

/// Write a TUM trajectory (seconds with ns precision).
inline bool WriteTum(const std::string& path, const std::vector<TimedPose>& poses) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fprintf(f, "# timestamp tx ty tz qx qy qz qw\n");
    for (const auto& p : poses) {
        const Eigen::Quaterniond q(p.T.R);
        std::fprintf(f, "%lld.%09lld %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n",
                     static_cast<long long>(p.t_ns / 1000000000LL), static_cast<long long>(p.t_ns % 1000000000LL),
                     p.T.t.x(), p.T.t.y(), p.T.t.z(), q.x(), q.y(), q.z(), q.w());
    }
    return std::fclose(f) == 0;
}

}  // namespace ariadne::runtime
