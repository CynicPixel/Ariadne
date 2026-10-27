// The intra-agent loop backend and retrieval (design robust-backend.md §8, D73 part 2b), synthetic, no runtime:
//   - edge convention: a noise-free loop from the gate has ~zero residual at the true poses;
//   - Sync: late (window-exit) odometry edges, Ready() false until both ends are mirrored, zero added energy,
//     a broken chain throws;
//   - a two-lap square with revisit loops plus one self-consistent WRONG loop (the gate passes it on geometry;
//     PCM/GNC must reject it): trace fields filled, mirror ATE better than odometry alone;
//   - retrieval: temporal exclusion, threshold, ordering.
#include "slam/loop_closure/loop_backend.hpp"
#include "slam/loop_closure/retrieval.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::Vec2;
using core::Vec3;
using core::Vec6;

namespace {

core::RectifiedStereo Rig() {
    core::RectifiedStereo r;
    r.cam = {700.0, 700.0, 640.0, 360.0};
    r.baseline = 0.12;
    return r;
}

/// Camera looking along the travel direction (z forward, y down), world z up.
SE3 CameraAt(const Vec3& p, double yaw) {
    SE3 T;
    const double c = std::cos(yaw), s = std::sin(yaw);
    T.R.col(0) = Vec3(s, -c, 0);
    T.R.col(1) = Vec3(0, 0, -1);
    T.R.col(2) = Vec3(c, s, 0);
    T.t = p;
    return T;
}

Vec6 Draw(std::mt19937& g, const Mat6& cov) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec6 z;
    for (int i = 0; i < 6; ++i) z(i) = n(g);
    return Eigen::LLT<Mat6>(cov).matrixL() * z;
}

Mat6 Diag(double st, double sr) {
    Vec6 d;
    d << st * st, st * st, st * st, sr * sr, sr * sr, sr * sr;
    return d.asDiagonal();
}

/// Points in front of camera A (world), seen by B's stereo (noisy disparity) and by A (noisy pixels); the
/// measured geometry is consistent with `T_A_B_geometry` (the truth, or a wrong pose for a self-consistent outlier).
LoopGeometry Geometry(std::mt19937& g, const SE3& T_W_A, const SE3& T_A_B_geometry, double noise) {
    LoopGeometry lg;
    lg.rig = Rig();
    std::uniform_real_distribution<double> u(20, 1260), v(20, 700), z(1.5, 6.0);
    std::normal_distribution<double> nz(0.0, 1.0);
    const SE3 T_B_A = T_A_B_geometry.inverse();
    while (lg.X_B.size() < 100) {
        const Vec2 uv{u(g), v(g)};   // pixel, then depth: sequenced (D123)
        const Vec3 XA = core::Unproject(lg.rig.cam, uv, z(g));
        const Vec3 XB = T_B_A * XA;
        if (XB.z() < 0.5) continue;
        const Vec2 uvB = core::Project(lg.rig.cam, XB);
        if (uvB.x() < 0 || uvB.x() > 1280 || uvB.y() < 0 || uvB.y() > 720) continue;
        const double d = lg.rig.cam.fx * lg.rig.baseline / XB.z() + noise * 0.65 * nz(g);
        lg.X_B.push_back(core::Unproject(lg.rig.cam, uvB + noise * 0.46 * Vec2{nz(g), nz(g)}, lg.rig.cam.fx * lg.rig.baseline / d));
        lg.uv_A.push_back(core::Project(lg.rig.cam, XA) + noise * 0.46 * Vec2{nz(g), nz(g)});
    }
    (void)T_W_A;
    Vec6 j;
    j << 0.02, -0.01, 0.015, 0.005, -0.004, 0.006;
    lg.T_A_B = core::SE3Exp(j) * T_A_B_geometry;
    return lg;
}

/// Two laps of a 6 m square, 0.4 m steps (60 keyframes per lap): node i and i - 60 are the same place.
std::vector<SE3> TwoLaps() {
    std::vector<SE3> t;
    for (int lap = 0; lap < 2; ++lap)
        for (int side = 0; side < 4; ++side)
            for (int k = 0; k < 15; ++k) {
                const double yaw = side * M_PI / 2;
                const Vec3 corner[4] = {{0, 0, 0}, {6, 0, 0}, {6, 6, 0}, {0, 6, 0}};
                const Vec3 p = corner[side] + 0.4 * k * Vec3(std::cos(yaw), std::sin(yaw), 0);
                t.push_back(CameraAt(p + Vec3(0, 0, 1.0), yaw));
            }
    return t;
}

}  // namespace

TEST(LoopBackend, NoiseFreeLoopEdgeHasZeroResidualAtTheTruth) {
    std::mt19937 g(1);
    const SE3 A = CameraAt(Vec3(1, 2, 1), 0.3), B = CameraAt(Vec3(1.4, 2.1, 1.05), 0.5);
    const DegeneracyReport d = AssessLoop(Geometry(g, A, A.inverse() * B, 0.0));
    ASSERT_FALSE(d.degenerate);
    PoseGraph pg;
    PoseNode na, nb;
    na.keyframe = 0, na.T_W_C = A;
    nb.keyframe = 1, nb.T_W_C = B;
    pg.AddNode(na);
    pg.AddNode(nb);
    PoseEdge e;
    e.type = EdgeType::kLocalLoop, e.from = 0, e.to = 1, e.Z = d.T_A_B, e.information = d.information;
    pg.AddEdge(e);
    EXPECT_LT(pg.Residual(pg.edges()[0]).norm(), 1e-8);   // A -> B, Z = T_A_B: the convention, pinned
}

TEST(LoopBackend, TwoLapsWithLateOdometryAndOneWrongLoop) {
    std::mt19937 g(2);
    const std::vector<SE3> truth = TwoLaps();
    const int N = int(truth.size()), W = 10;   // window: an odometry edge appears W keyframes late
    const Mat6 odo = Diag(0.01, 0.004);
    PoseGraph vo;
    LoopBackend be;
    std::vector<SE3> vo_pose;
    std::vector<std::pair<int, SE3>> pending;   // (to, Z)
    int events = 0, wrong_event_edge = -1, ready_checks = 0;
    bool energy_ok = true;
    std::vector<LoopEventRecord> recs;
    for (int i = 0; i < N; ++i) {
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        const SE3 Z = i > 0 ? core::SE3Exp(Draw(g, odo)) * (truth[size_t(i - 1)].inverse() * truth[size_t(i)]) : SE3();
        vo_pose.push_back(i > 0 ? vo_pose.back() * Z : truth[0]);
        nd.T_W_C = vo_pose.back();
        vo.AddNode(nd);
        if (i > 0) pending.push_back({i, Z});
        if (i >= W) {   // the edge into keyframe i - W + 1 becomes final now
            const auto [to, Zf] = pending.front();
            pending.erase(pending.begin());
            vo.AddEdge(OdometryEdge(uint64_t(to - 1), uint64_t(to), Zf, odo));
        }
        const double before = be.graph().Cost();
        be.Sync(vo);
        energy_ok = energy_ok && std::fabs(be.graph().Cost() - before) <= 1e-9 * std::max(1.0, before);
        // Loop candidates: the query keyframe q = i - W (now final) revisits q - 60 on the second lap.
        const int q = i - W;
        if (q >= 60 && q % 5 == 0) {
            EXPECT_FALSE(be.Ready(uint64_t(i), uint64_t(i - 60))) << "keyframe " << i << " is still in the window";
            ++ready_checks;
            ASSERT_TRUE(be.Ready(uint64_t(q), uint64_t(q - 60)));
            std::vector<LoopCandidate> c(1);
            c[0].a = uint64_t(q), c[0].b = uint64_t(q - 60);
            c[0].geometry = Geometry(g, truth[size_t(q)], truth[size_t(q)].inverse() * truth[size_t(q - 60)], 1.0);
            c[0].retrieval_score = 0.9;
            if (q == 90) {   // a self-consistent WRONG loop to a different place, in the same event
                Vec6 off;
                off << 1.5, -0.8, 0.0, 0.0, 0.0, 0.4;
                LoopCandidate w;
                w.a = uint64_t(q), w.b = 20;
                w.geometry = Geometry(g, truth[size_t(q)], core::SE3Exp(off) * (truth[size_t(q)].inverse() * truth[20]), 1.0);
                c.push_back(w);
            }
            recs.push_back(be.OnCandidates(c, int64_t(q) * 100000000));
            ++events;
            if (q == 90) wrong_event_edge = int(recs.back().candidates.at(1).edge);
        }
    }
    EXPECT_TRUE(energy_ok) << "Sync changed the energy (new nodes must add zero odometry residual)";
    EXPECT_GT(ready_checks, 0);
    ASSERT_GE(wrong_event_edge, 0) << "the wrong loop should pass the GATE (its geometry is good)";
    const auto& last = recs.back();
    EXPECT_TRUE(last.solved);
    EXPECT_GT(last.ms_gnc, 0.0);
    EXPECT_FALSE(be.graph().edges()[size_t(wrong_event_edge)].gnc_weight > 0.5) << "wrong loop accepted";
    int inliers = 0, true_loops = 0;
    for (const auto& r : recs)
        for (const auto& c : r.candidates) {
            if (c.edge != wrong_event_edge) ++true_loops;   // identify the wrong loop by its edge, not its keyframes
            // the gate's corridor score reaches the record (D148): lambda_max / trace lies in [1/3, 1]
            EXPECT_GE(c.corridor, 1.0 / 3.0 - 1e-9);
            EXPECT_LE(c.corridor, 1.0 + 1e-9);
        }
    for (size_t e = 0; e < be.graph().edges().size(); ++e)
        if (IsLoopType(be.graph().edges()[e].type) && int(e) != wrong_event_edge) inliers += be.graph().edges()[e].gnc_weight > 0.5;
    double ate_vo = 0, ate_m = 0;
    int n = 0;
    for (const auto& [id, node] : be.graph().nodes()) {
        ate_vo += (vo_pose[size_t(id)].t - truth[size_t(id)].t).squaredNorm();
        ate_m += (node.T_W_C.t - truth[size_t(id)].t).squaredNorm();
        ++n;
    }
    ate_vo = std::sqrt(ate_vo / n);
    ate_m = std::sqrt(ate_m / n);
    int fast = 0;
    for (const auto& r : recs) fast += r.fast_path;
    std::printf("two laps: %d events, %d / %d true loops GNC inliers, wrong loop rejected; fast path %d / %d | ATE "
                "odometry %.3f m -> mirror %.3f m (%d keyframes mirrored of %d; %d wait in the window)\n", events, inliers,
                true_loops, fast, events, ate_vo, ate_m, n, N, N - n);
    EXPECT_EQ(inliers, true_loops);
    EXPECT_LT(ate_m, ate_vo);
}

TEST(LoopBackend, BrokenChainThrows) {
    PoseGraph vo;
    for (uint64_t k : {0, 1, 5, 6}) {
        PoseNode n;
        n.keyframe = k;
        vo.AddNode(n);
    }
    const Mat6 odo = Diag(0.01, 0.004);
    vo.AddEdge(OdometryEdge(0, 1, SE3(), odo));
    vo.AddEdge(OdometryEdge(5, 6, SE3(), odo));   // nothing links 1 and 5
    LoopBackend be;
    EXPECT_THROW(be.Sync(vo), std::logic_error);
}

// D87 (EuRoC MH_04/MH_05 on AWS): the VO emits no exit edge for keyframes whose window solve was skipped before
// inertial init, so the mirror's chain starts later than keyframe 0. A loop to one of those keyframes is never Ready;
// the runtime's FIFO used to wait on it forever (0 loop events on both flights). Unreachable() names them.
TEST(LoopBackend, KeyframesBeforeTheAnchorAreUnreachable) {
    PoseGraph vo;
    for (uint64_t k = 0; k < 12; ++k) {
        PoseNode n;
        n.keyframe = k;
        vo.AddNode(n);
    }
    const Mat6 odo = Diag(0.01, 0.004);
    LoopBackend be;
    EXPECT_FALSE(be.Unreachable(2)) << "nothing is final before the first exit edge";
    for (uint64_t k = 5; k < 9; ++k) vo.AddEdge(OdometryEdge(k, k + 1, SE3(), odo));   // 0-4 never got an exit edge
    be.Sync(vo);
    EXPECT_EQ(be.anchor(), 5u);
    for (uint64_t k : {0, 2, 4}) EXPECT_TRUE(be.Unreachable(k)) << k;
    for (uint64_t k : {5, 9}) EXPECT_FALSE(be.Unreachable(k)) << k << " is mirrored";
    EXPECT_FALSE(be.Unreachable(10)) << "10 is still in the window: it will become Ready";
    EXPECT_FALSE(be.Ready(9, 2));
}

TEST(LoopRetrieval, ExclusionThresholdAndOrder) {
    RetrievalParams p;
    p.exclude_recent = 5;
    p.exclude_seconds = 1.0;
    p.min_score = 0.5;
    p.top_k = 2;
    LoopRetrieval r(p);
    auto d = [](float a, float b) { return std::vector<float>{a, b}; };
    r.Add(0, 0, d(1, 0));
    r.Add(1, int64_t(0.5e9), d(0.9f, 0.1f));
    r.Add(2, int64_t(5e9), d(0.8f, 0.6f));
    r.Add(3, int64_t(6e9), d(0, 1));
    r.Add(10, int64_t(9.5e9), d(1, 0));   // recent by id
    const auto out = r.Query(12, int64_t(10e9), d(1, 0));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].first, 0u);   // score 1
    EXPECT_EQ(out[1].first, 1u);   // ~0.994; id 2 (0.8) is third, cut by top_k; id 3 (0) under threshold
    const auto near = r.Query(12, int64_t(0.6e9), d(1, 0));   // ids 0 and 1 are within 1 s of the query now
    ASSERT_EQ(near.size(), 1u);
    EXPECT_EQ(near[0].first, 2u);
}

// D74 behind its flag: certificate on vs off, the same event sequence in lockstep. The mirrors stay bit-identical
// (the certificate is read-only), and only the flagged backend records a verdict.
TEST(LoopBackend, CertificateFlagIsReadOnly) {
    std::mt19937 g(2);
    const std::vector<SE3> truth = TwoLaps();
    const int N = int(truth.size());
    const Mat6 odo = Diag(0.01, 0.004);
    PoseGraph vo;
    LoopBackendParams on;
    on.certificate.enabled = true;
    LoopBackend off_be, on_be(on);
    SE3 pose = truth[0];
    int certified = 0, ran = 0;
    for (int i = 0; i < N; ++i) {
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        const SE3 Z = i > 0 ? core::SE3Exp(Draw(g, odo)) * (truth[size_t(i - 1)].inverse() * truth[size_t(i)]) : SE3();
        if (i > 0) pose = pose * Z;
        nd.T_W_C = pose;
        vo.AddNode(nd);
        if (i > 0) vo.AddEdge(OdometryEdge(uint64_t(i - 1), uint64_t(i), Z, odo));
        off_be.Sync(vo);
        on_be.Sync(vo);
        if (i >= 60 && i % 5 == 0) {
            std::vector<LoopCandidate> c(1);
            c[0].a = uint64_t(i), c[0].b = uint64_t(i - 60);
            c[0].geometry = Geometry(g, truth[size_t(i)], truth[size_t(i)].inverse() * truth[size_t(i - 60)], 1.0);
            const LoopEventRecord a = off_be.OnCandidates(c, i), b = on_be.OnCandidates(c, i);
            EXPECT_FALSE(a.certificate_ran);
            EXPECT_EQ(b.certificate_ran, b.solved);
            ran += b.certificate_ran;
            certified += b.certified;
        }
    }
    for (const auto& [id, n] : off_be.graph().nodes()) {
        const SE3& T = on_be.graph().nodes().at(id).T_W_C;
        EXPECT_EQ(std::memcmp(n.T_W_C.R.data(), T.R.data(), sizeof(double) * 9), 0);
        EXPECT_EQ(std::memcmp(n.T_W_C.t.data(), T.t.data(), sizeof(double) * 3), 0);
    }
    std::printf("certificate flag: %d / %d solved events certified (chordal surrogate of the mirror)\n", certified, ran);
    EXPECT_GT(ran, 0);
}
