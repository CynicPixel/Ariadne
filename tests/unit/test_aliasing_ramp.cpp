// Perceptual-aliasing outlier ramp (D75; design ROBUST_BACKEND_2.5.md §6; PHASE2_PLAN §3 "outlier ramp").
// Protocol, fixed before the first measured run:
//   - world: 3 laps of a 20 m x 10 m rectangular corridor at 1 m spacing (60 keyframes per lap, 180 in all), laps
//     offset 0.2 m sideways; odometry left noise 0.02 m / 0.005 rad per step; every run starts from dead reckoning;
//   - 60 loops per graph: (1 - rate) x 60 TRUE loops (same corridor position on two different laps, sampled without
//     replacement) and rate x 60 FALSE loops in aliasing groups (aliasing_generator.hpp; segment starts >= 3 m apart,
//     apparent offset J up to 0.3 m / 0.05 rad); every loop has left noise 0.05 m / 0.01 rad;
//   - two group policies: "size10" = groups of 10 (Sünderhauf & Protzel 2013's grouped policy; the last one smaller)
//     and "count2" = two groups splitting the false loops evenly, so the largest false group grows with the rate;
//   - two arms: "gnc" = GNC-TLS on every loop; "pcm_gnc" = PCM (alpha_loop 0.001), then GNC on PCM's accepted loops;
//   - success = loop precision >= 0.99 and recall >= 0.95 (final GNC weight > 0.5), as D73's ramp;
//   - rates 0.1 .. 0.9 step 0.1, 10 seeds each; BREAKPOINT = the lowest rate whose success probability is < 0.5
//     ("none" if no rate falls below);
//   - ATE = RMS translation error against the truth (anchor at the true first pose, no alignment); runtime = wall
//     time of PCM and GNC.
// The full ramp writes raw CSV when ARIADNE_RAMP_OUT is set (evidence run); otherwise one seed at three rates runs.
#include "slam/robust/aliasing_generator.hpp"
#include "slam/robust/gnc.hpp"
#include "slam/robust/pcm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>

using namespace ariadne;
using namespace ariadne::slam;
using core::SE3;
using core::Vec3;

namespace {

Mat6 Diag(double st, double sr) {
    Vec6 d;
    d << st * st, st * st, st * st, sr * sr, sr * sr, sr * sr;
    return d.asDiagonal();
}

const Mat6 kOdo = Diag(0.02, 0.005);
const Mat6 kLoop = Diag(0.05, 0.01);
constexpr int kPerLap = 60, kLaps = 3, kLoops = 60;

/// Corridor position p in [0, 60) on lap l: the rectangle's perimeter at 1 m spacing, heading along the corridor.
SE3 CorridorPose(int lap, int p) {
    const double w = 20, h = 10, s = p;
    Vec3 x;
    double yaw;
    if (s < w) x = Vec3(s, 0, 0), yaw = 0;
    else if (s < w + h) x = Vec3(w, s - w, 0), yaw = M_PI / 2;
    else if (s < 2 * w + h) x = Vec3(w - (s - w - h), h, 0), yaw = M_PI;
    else x = Vec3(0, h - (s - 2 * w - h), 0), yaw = -M_PI / 2;
    SE3 T;
    T.R = Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix();
    T.t = x + T.R * Vec3(0, 0.2 * lap, 0);   // laps offset sideways
    return T;
}

struct World {
    std::map<uint64_t, SE3> truth;
    std::vector<uint64_t> chain;
    PoseGraph g;
    std::set<size_t> true_loops, false_loops;
    int groups = 0, largest_group = 0;
};

World Make(std::mt19937& gen, double rate, bool count2) {
    World w;
    for (int l = 0; l < kLaps; ++l)
        for (int p = 0; p < kPerLap; ++p) {
            const uint64_t id = uint64_t(l * kPerLap + p);
            w.truth[id] = CorridorPose(l, p);
            w.chain.push_back(id);
        }
    SE3 est = w.truth[0];
    for (uint64_t id : w.chain) {
        if (id > 0) est = est * (core::SE3Exp(DrawLeftNoise(gen, kOdo)) * (w.truth[id - 1].inverse() * w.truth[id]));
        PoseNode n;
        n.keyframe = id;
        n.T_W_C = est;
        w.g.AddNode(n);
        if (id > 0) w.g.AddEdge(OdometryEdge(id - 1, id, w.g.nodes().at(id - 1).T_W_C.inverse() * est, kOdo));
    }
    const int n_false = int(std::lround(rate * kLoops)), n_true = kLoops - n_false;
    std::vector<std::pair<uint64_t, uint64_t>> cand;
    for (int p = 0; p < kPerLap; ++p)
        for (int a = 0; a < kLaps; ++a)
            for (int b = a + 1; b < kLaps; ++b) cand.push_back({uint64_t(a * kPerLap + p), uint64_t(b * kPerLap + p)});
    std::shuffle(cand.begin(), cand.end(), gen);
    for (int k = 0; k < n_true; ++k)
        w.true_loops.insert(AddTrueLoop(w.g, w.truth, cand[size_t(k)].first, cand[size_t(k)].second, kLoop,
                                        EdgeType::kLocalLoop, gen));
    std::vector<int> sizes;
    if (count2) {
        if (n_false > 0) sizes.push_back((n_false + 1) / 2);
        if (n_false > 1) sizes.push_back(n_false / 2);
    } else {
        for (int left = n_false; left > 0; left -= 10) sizes.push_back(std::min(10, left));
    }
    AliasingGroupSpec spec;
    spec.loop_cov = kLoop;
    for (int sz : sizes) {
        spec.size = sz;
        const auto p = PlantAliasingGroup(w.g, w.truth, {w.chain}, spec, gen);
        if (!p) throw std::runtime_error("ramp: no admissible segment pair");
        w.false_loops.insert(p->edges.begin(), p->edges.end());
        ++w.groups;
        w.largest_group = std::max(w.largest_group, sz);
    }
    return w;
}

struct Outcome {
    double precision = 1, recall = 1, ate = 0, ms_pcm = 0, ms_gnc = 0;
    int pcm_accepted = -1, inliers = 0;
    bool success = false;
};

Outcome RunArm(World w, bool pcm) {
    Outcome o;
    std::vector<bool> active(w.g.edges().size(), true);
    auto t0 = std::chrono::steady_clock::now();
    if (pcm) {
        const PcmReport r = Pcm(w.g, PcmParams{});
        std::fill(active.begin(), active.end(), false);
        for (size_t l : r.accepted) active[l] = true;
        o.pcm_accepted = int(r.accepted.size());
    }
    auto t1 = std::chrono::steady_clock::now();
    const GncReport r = SolveGnc(w.g, 0, &active);
    auto t2 = std::chrono::steady_clock::now();
    o.ms_pcm = std::chrono::duration<double, std::milli>(t1 - t0).count();
    o.ms_gnc = std::chrono::duration<double, std::milli>(t2 - t1).count();
    int tp = 0;
    for (size_t l : r.inliers) tp += int(w.true_loops.count(l));
    o.inliers = int(r.inliers.size());
    o.precision = r.inliers.empty() ? 1.0 : double(tp) / double(r.inliers.size());
    o.recall = w.true_loops.empty() ? 1.0 : double(tp) / double(w.true_loops.size());
    o.success = o.precision >= 0.99 && o.recall >= 0.95;
    double s = 0;
    for (const auto& [id, T] : w.truth) s += (w.g.nodes().at(id).T_W_C.t - T.t).squaredNorm();
    o.ate = std::sqrt(s / double(w.truth.size()));
    return o;
}

}  // namespace

// The generator's own property: a planted group is mutually consistent, so PCM accepts it whole (otherwise the
// generator would have rebuilt random outliers).
TEST(AliasingGenerator, PcmAcceptsAPlantedGroupAsOneClique) {
    for (int seed = 0; seed < 5; ++seed) {
        std::mt19937 gen(10u + unsigned(seed));
        World w = Make(gen, 0.0, false);
        PoseGraph g = w.g;
        std::vector<PoseEdge> keep;
        for (const auto& e : g.edges())
            if (!IsLoopType(e.type)) keep.push_back(e);
        PoseGraph clean;
        for (const auto& [id, n] : g.nodes()) clean.AddNode(n);
        for (const auto& e : keep) clean.AddEdge(e);
        AliasingGroupSpec spec;
        spec.loop_cov = kLoop;
        spec.size = 10;
        const auto p = PlantAliasingGroup(clean, w.truth, {w.chain}, spec, gen);
        ASSERT_TRUE(p);
        const PcmReport r = Pcm(clean, PcmParams{});
        EXPECT_EQ(r.accepted.size(), 10u) << "seed " << seed;
        EXPECT_GE((w.truth.at(w.chain[p->start_a]).t - w.truth.at(w.chain[p->start_b]).t).norm(), 3.0);
    }
}

TEST(AliasingRamp, BreakpointReported) {
    const char* out = std::getenv("ARIADNE_RAMP_OUT");
    std::vector<double> rates{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    int seeds = 10;
    if (!out) rates = {0.3, 0.7, 0.9}, seeds = 1;
    std::ofstream csv;
    if (out) {
        csv.open(std::string(out) + "/ramp_raw.csv");
        csv << "policy,arm,rate,seed,n_true,n_false,groups,largest_group,pcm_accepted,gnc_inliers,precision,recall,"
               "success,ate_m,ms_pcm,ms_gnc\n";
    }
    for (const bool count2 : {false, true}) {
        const char* policy = count2 ? "count2" : "size10";
        for (const bool pcm : {false, true}) {
            const char* arm = pcm ? "pcm_gnc" : "gnc";
            std::string breakpoint = "none";
            for (const double rate : rates) {
                int ok = 0;
                double ate = 0, ms = 0;
                for (int seed = 0; seed < seeds; ++seed) {
                    std::mt19937 gen(75000u + 1000u * unsigned(std::lround(rate * 10)) + unsigned(seed));
                    const World w = Make(gen, rate, count2);
                    const Outcome o = RunArm(w, pcm);
                    ok += o.success;
                    ate += o.ate;
                    ms += o.ms_pcm + o.ms_gnc;
                    if (out)
                        csv << policy << ',' << arm << ',' << rate << ',' << seed << ',' << w.true_loops.size() << ','
                            << w.false_loops.size() << ',' << w.groups << ',' << w.largest_group << ',' << o.pcm_accepted
                            << ',' << o.inliers << ',' << o.precision << ',' << o.recall << ',' << int(o.success) << ','
                            << o.ate << ',' << o.ms_pcm << ',' << o.ms_gnc << '\n';
                }
                if (breakpoint == "none" && 2 * ok < seeds) breakpoint = std::to_string(rate).substr(0, 3);
                std::printf("ramp %-6s %-7s false %2.0f%%: success %2d / %d, ATE mean %.3f m, %.0f ms mean\n", policy,
                            arm, rate * 100, ok, seeds, ate / seeds, ms / seeds);
            }
            std::printf("ramp %-6s %-7s BREAKPOINT %s\n", policy, arm, breakpoint.c_str());
        }
    }
}
