#pragma once
// ---------------------------------------------------------------------------
// PGO assembly digest (design pose-graph.md §5): FNV-1a over the raw IEEE
// bits of the energy, H and g of ONE fixed pose graph, assembled at a fixed
// point. Measured into every run manifest (D21 pattern), so a result produced
// by drifted pose-graph numerics is identifiable after the fact.
//
// SCOPE, stated honestly: the SE(3) maths uses libm (std::sin/cos/sqrt), so
// this digest is reproducible for the same binary on the same architecture,
// NOT across architectures. The cross-architecture guarantee stays where
// Phase 0 put it: core::DetMathDigest (det_math). The full LevMarqBase solve
// is checked bit-identical twice in-process by test_pgo_levmarq.cpp.
// ---------------------------------------------------------------------------
#include "core/determinism/digests.hpp"
#include "slam/pose_graph/pgo_problem.hpp"

#include <cstring>

namespace ariadne::slam {

inline uint64_t PgoAssemblyDigest() {
    PoseGraph g;
    const int n = 8;
    for (int i = 0; i < n; ++i) {
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        core::Vec6 xi;
        xi << 0.7 * i, -0.3 * i, 0.1 * i * i, 0.05 * i, -0.11 * i, 0.31 + 0.2 * i;
        nd.T_W_C = core::SE3Exp(xi);
        g.AddNode(nd);
    }
    auto edge = [&](int a, int b, double k) {
        PoseEdge e;
        e.type = EdgeType::kLocalLoop;
        e.from = uint64_t(a);
        e.to = uint64_t(b);
        core::Vec6 d;
        d << 0.01 * k, -0.02 * k, 0.015, 0.003 * k, -0.002, 0.004 * k;
        e.Z = core::SE3Exp(d) * g.nodes().at(e.from).T_W_C.inverse() * g.nodes().at(e.to).T_W_C;
        e.information = Mat6::Identity() * (10.0 + k);
        e.information(0, 4) = e.information(4, 0) = 0.5 * k;
        g.AddEdge(e);
    };
    for (int i = 0; i + 1 < n; ++i) edge(i, i + 1, double(i));
    edge(0, 5, 1.5);
    edge(2, 7, 2.5);
    edge(3, 3, 0.5);   // a unary edge
    const PgoProblem p(g, 0);
    SparseMat H;
    Eigen::VectorXd grad;
    const double E = p.Linearize(p.poses(), H, grad);
    core::Fnv1a64 h;
    auto bits = [&](double v) {
        uint64_t u;
        std::memcpy(&u, &v, 8);
        h.AbsorbU64(u);
    };
    bits(E);
    const Eigen::MatrixXd Hd(H);
    for (Eigen::Index i = 0; i < Hd.size(); ++i) bits(Hd.data()[i]);
    for (Eigen::Index i = 0; i < grad.size(); ++i) bits(grad(i));
    return h.value();
}

}  // namespace ariadne::slam
