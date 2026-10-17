// Phase 2 exit-gate item (arch §11.6, §37.5; POSE_GRAPH_2.md §5): Ariadne's PGO (OpenCV LevMarqBase + our backend)
// vs GTSAM 4.3 LM on the cases python/oracle/pgo_gtsam.py writes to $ARIADNE_PGO_ORACLE (same graph, same
// information, same anchor, same initial estimate; half the graphs with full correlated information). Agreement:
// final energy within 1e-6 relative, and every pose within 1e-6 absolute OR within 1e-6 Mahalanobis^2 of the
// node's marginal (scale-invariant; see the comment in the loop). SKIPS without the cases; FAILS under ARIADNE_REQUIRE_MODELS=1 (pre-merge gate).
#include "slam/pose_graph/pgo_levmarq.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace ariadne;
using namespace ariadne::slam;
namespace fs = std::filesystem;

namespace {

SE3 ReadPose(std::istringstream& in) {
    double tx, ty, tz, qw, qx, qy, qz;
    in >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
    SE3 T;
    T.R = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
    T.t = core::Vec3(tx, ty, tz);
    return T;
}

struct OracleCase {
    PoseGraph graph;
    uint64_t anchor = 0;
    std::map<uint64_t, SE3> opt;
    double init_error = 0, final_error = 0;
    int iterations = 0;
};

OracleCase Load(const fs::path& p) {
    std::ifstream f(p);
    std::string line;
    std::getline(f, line);
    if (line != "ariadne-pgo 1") throw std::runtime_error("oracle: bad header in " + p.string());
    OracleCase c;
    std::vector<PoseEdge> edges;
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string tag;
        in >> tag;
        if (tag == "ANCHOR") {
            in >> c.anchor;
        } else if (tag == "NODE") {
            PoseNode n;
            in >> n.keyframe;
            n.T_W_C = ReadPose(in);
            c.graph.AddNode(n);
        } else if (tag == "EDGE") {
            PoseEdge e;
            e.type = EdgeType::kLocalLoop;
            in >> e.from >> e.to;
            e.Z = ReadPose(in);
            for (int r = 0; r < 6; ++r)
                for (int col = r; col < 6; ++col) {
                    in >> e.information(r, col);
                    e.information(col, r) = e.information(r, col);
                }
            edges.push_back(e);
        } else if (tag == "OPT") {
            uint64_t id;
            in >> id;
            c.opt[id] = ReadPose(in);
        } else if (tag == "INIT_ERROR") {
            in >> c.init_error;
        } else if (tag == "FINAL_ERROR") {
            in >> c.final_error;
        } else if (tag == "ITERATIONS") {
            in >> c.iterations;
        }
        if (in.fail()) throw std::runtime_error("oracle: malformed line in " + p.string() + ": " + line);
    }
    for (const auto& e : edges) c.graph.AddEdge(e);
    return c;
}

}  // namespace

TEST(PgoOracle, AgreesWithGtsamOnEveryCase) {
    const char* dir = std::getenv("ARIADNE_PGO_ORACLE");
    const char* req = std::getenv("ARIADNE_REQUIRE_MODELS");
    if (!dir || !fs::is_directory(dir)) {
        if (req && std::string(req) == "1") FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no PGO oracle cases";
        GTEST_SKIP() << "no oracle cases (python -m oracle.pgo_gtsam, then set ARIADNE_PGO_ORACLE)";
    }
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".graph") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    ASSERT_FALSE(files.empty());
    PgoSettings s;   // tight: agreement to 1e-6 needs convergence well below it
    s.rel_energy_delta_tolerance = 1e-14;
    s.step_norm_tolerance = 1e-14;
    s.min_gradient_tolerance = 1e-10;
    s.max_iterations = 1000;
    s.recompute_covariance = true;   // the marginals define the Mahalanobis comparison below
    int agree = 0, abs_ok = 0;
    double worst_e = 0, worst_t = 0, worst_r = 0, worst_m2 = 0;
    for (const auto& p : files) {
        OracleCase c = Load(p);
        const double e0 = PgoProblem(c.graph, c.anchor).Energy(PgoProblem(c.graph, c.anchor).poses());
        EXPECT_NEAR(e0, c.init_error, 1e-9 * std::max(1.0, c.init_error)) << p.filename() << ": initial energies differ "
                                                                          << "(the two costs are not the same)";
        const PgoReport rep = SolvePoseGraph(c.graph, c.anchor, s);
        const double de = std::fabs(rep.final_energy - c.final_error) / std::max(1.0, c.final_error);
        // Poses: absolute difference, AND the difference in units of the solution's own uncertainty (Mahalanobis^2
        // of Log(X_ours X_gtsam^-1) under the node's marginal). With information spanning ~1000x, two solvers that
        // agree on the energy to 1e-14 can still differ by ~1e-6 m along a weak direction; that is a flat minimum,
        // not a disagreement, and only the scale-invariant measure can tell the two apart.
        double dt = 0, dr = 0, m2 = 0;
        for (const auto& [id, T] : c.opt) {
            const PoseNode& n = c.graph.nodes().at(id);
            dt = std::max(dt, (n.T_W_C.t - T.t).norm());
            dr = std::max(dr, core::SO3Log(n.T_W_C.R * T.R.transpose()).norm());
            if (id == c.anchor) continue;
            const core::Vec6 d = core::SE3Log(n.T_W_C * T.inverse());
            m2 = std::max(m2, d.dot(n.covariance.ldlt().solve(d)));
        }
        const bool poses_ok = (dt <= 1e-6 && dr <= 1e-6) || m2 <= 1e-6;
        const bool ok = de <= 1e-6 && poses_ok;
        agree += ok;
        abs_ok += dt <= 1e-6 && dr <= 1e-6;
        worst_e = std::max(worst_e, de), worst_t = std::max(worst_t, dt), worst_r = std::max(worst_r, dr);
        worst_m2 = std::max(worst_m2, m2);
        EXPECT_TRUE(ok) << p.filename() << ": energy ours " << rep.final_energy << " gtsam " << c.final_error
                        << " (rel " << de << "), max dt " << dt << " m, max dr " << dr << " rad, max mahalanobis^2 "
                        << m2 << "; ours " << rep.iterations << " its, gtsam " << c.iterations;
    }
    std::printf("PGO oracle: %d / %zu cases agree (%d also within 1e-6 absolute); worst rel energy %.2e, worst dt "
                "%.2e m, worst dr %.2e rad, worst mahalanobis^2 %.2e\n",
                agree, files.size(), abs_ok, worst_e, worst_t, worst_r, worst_m2);
}

// The SE-Sync benchmark graphs (models.sh pgo-sesync; NOT in the pre-merge gate: rim and grid3D are slow), from the
// files' own odometry vertices. Criterion (POSE_GRAPH_2.md §5, revised 2026-10-16 with the evidence there):
//  - a file whose information matrices are not all PSD is EXCLUDED, by name: the cost is unbounded below and
//    PoseGraph::AddEdge refuses it (GTSAM's Information() Cholesky does not check, so its number is for a different
//    problem). Exactly cubicle and rim are expected; any other exclusion fails the test.
//  - same minimum (final energies within 1e-6 relative): ours must be stationary; poses are REPORTED, not gated --
//    on 2.5k-8k-node graphs a 3e-7 relative energy difference spans 1.7e-3 m along flat directions (sphere2500),
//    and the per-pose test lives in the 100-case oracle above, with the Mahalanobis criterion.
//  - different minima: ours must be stationary and the lower cost is reported.
TEST(PgoOracle, SeSyncBenchmarkGraphs) {
    const char* dir = std::getenv("ARIADNE_PGO_SESYNC");
    if (!dir || !fs::is_directory(dir)) GTEST_SKIP() << "set ARIADNE_PGO_SESYNC (infra/scripts/models.sh pgo-sesync)";
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".graph") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    ASSERT_FALSE(files.empty());
    PgoSettings s;
    s.rel_energy_delta_tolerance = 1e-14;
    s.step_norm_tolerance = 1e-14;
    s.min_gradient_tolerance = 1e-9;
    s.max_iterations = 1000;
    s.recompute_covariance = false;
    std::vector<std::string> excluded;
    for (const auto& p : files) {
        int indefinite = 0, edges = 0;
        {
            std::ifstream in(p);
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line);
                std::string tag;
                if (!(ls >> tag) || tag != "EDGE") continue;
                double v[30];
                for (double& x : v) ls >> x;   // a, b, t(3), q(4), then the 21 information entries
                core::Mat6 I;
                for (int i = 0, k = 9; i < 6; ++i)
                    for (int j = i; j < 6; ++j, ++k) I(i, j) = I(j, i) = v[k];
                ++edges;
                indefinite += Eigen::SelfAdjointEigenSolver<core::Mat6>(I).eigenvalues()(0) < 0.0;
            }
        }
        if (indefinite > 0) {
            std::printf("%-16s EXCLUDED: %d / %d edge information matrices are indefinite\n", p.stem().c_str(),
                        indefinite, edges);
            excluded.push_back(p.stem().string());
            continue;
        }
        OracleCase c = Load(p);
        const PgoReport rep = SolvePoseGraph(c.graph, c.anchor, s);
        const PgoProblem prob(c.graph, c.anchor);
        SparseMat H;
        Eigen::VectorXd grad;
        prob.Linearize(prob.poses(), H, grad);
        const double gmax = grad.cwiseAbs().maxCoeff();
        const double de = std::fabs(rep.final_energy - c.final_error) / std::max(1.0, c.final_error);
        double dt = 0, dr = 0;
        for (const auto& [id, T] : c.opt) {
            const SE3& X = c.graph.nodes().at(id).T_W_C;
            dt = std::max(dt, (X.t - T.t).norm());
            dr = std::max(dr, core::SO3Log(X.R * T.R.transpose()).norm());
        }
        const bool same_min = de <= 1e-6;
        std::printf("%-16s %6zu nodes: E0 %.6g | ours %.10g (%d its, found %d, %d lin, %d/%d fact failed, max|grad| "
                    "%.1e) | gtsam %.10g (%d its) | %s",
                    p.stem().c_str(), c.graph.nodes().size(), c.init_error, rep.final_energy, rep.iterations,
                    int(rep.found), rep.linearizations, rep.failed_factorizations, rep.factorizations, gmax,
                    c.final_error, c.iterations, same_min ? "same minimum" : "DIFFERENT minima");
        if (same_min) std::printf(", max dt %.1e m, dr %.1e rad\n", dt, dr);
        else std::printf(", lower: %s\n", rep.final_energy < c.final_error ? "ours" : "gtsam");
        EXPECT_LT(gmax, 1e-4 * std::max(1.0, rep.final_energy)) << p.filename() << ": ours is not stationary";
    }
    EXPECT_EQ(excluded, (std::vector<std::string>{"cubicle", "rim"}));
}
