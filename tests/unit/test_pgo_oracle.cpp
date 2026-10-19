// Phase 2 exit-gate item (arch §11.6, §37.5; POSE_GRAPH_2.md §5): Ariadne's PGO (OpenCV LevMarqBase + our backend)
// vs GTSAM 4.3 LM on the cases python/oracle/pgo_gtsam.py writes to $ARIADNE_PGO_ORACLE (same graph, same
// information, same anchor, same initial estimate; half the graphs with full correlated information). Agreement:
// final energy within 1e-6 relative, and every pose within 1e-6 absolute OR within 1e-6 Mahalanobis^2 of the
// node's marginal (scale-invariant; see the comment in the loop). SKIPS without the cases; FAILS under ARIADNE_REQUIRE_MODELS=1 (pre-merge gate).
#include "slam/pose_graph/chordal_init.hpp"
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
    std::map<uint64_t, core::Mat3> chordal;   // GTSAM's chordal rotations (CHORDAL_R, 2.5 oracle files only)
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
        } else if (tag == "CHORDAL_R") {
            uint64_t id;
            double qw, qx, qy, qz;
            in >> id >> qw >> qx >> qy >> qz;
            c.chordal[id] = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
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
    PgoSettings s = PgoV2();   // the pose graph's settings, tight: agreement to 1e-6 needs convergence well below it
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
    PgoSettings s = PgoV2();
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
        // Same problem first: GTSAM's error at the shared start must be ours. (Missing until 2.5: the oracle fed GTSAM
        // unnormalised g2o quaternions, a slightly different problem, and garage read as "different minima"; D70.)
        const double e0 = PgoProblem(c.graph, c.anchor).Energy(PgoProblem(c.graph, c.anchor).poses());
        EXPECT_NEAR(e0, c.init_error, 1e-9 * std::max(1.0, c.init_error)) << p.filename() << ": initial energies "
                                                                          << "differ (the two costs are not the same)";
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

// 2.5 open item (b), D70: why does parking-garage take 1000 LM iterations where GTSAM takes 8? One setting changed at
// a time, same dead-reckoning start, same tolerances as the test above. A diagnosis, not a gate: it reports and checks
// only that every run is stationary. ARIADNE_PGO_DAMPING_GRAPHS = comma-separated stems (default parking-garage).
TEST(PgoOracle, DampingAblation) {
    const char* dir = std::getenv("ARIADNE_PGO_SESYNC");
    if (!dir || !fs::is_directory(dir)) GTEST_SKIP() << "set ARIADNE_PGO_SESYNC (infra/scripts/models.sh pgo-sesync)";
    const char* which = std::getenv("ARIADNE_PGO_DAMPING_GRAPHS");
    std::vector<std::string> stems;
    std::stringstream list(which ? which : "parking-garage");
    for (std::string s; std::getline(list, s, ',');) stems.push_back(s);
    struct Variant {
        const char* name;
        bool clamp, jacobi;
    };
    const Variant variants[] = {{"baseline (clamp on, jacobi off)", true, false},
                                {"clamp off", false, false},
                                {"jacobi on", true, true},
                                {"clamp off + jacobi on", false, true}};
    for (const auto& stem : stems) {
        const fs::path p = fs::path(dir) / (stem + ".graph");
        ASSERT_TRUE(fs::exists(p)) << p;
        {   // the clamp floor is an absolute 1e-6 on lambda*diag(H): how big is diag(H) at the start?
            const OracleCase c = Load(p);
            const PgoProblem prob(c.graph, c.anchor);
            SparseMat H;
            Eigen::VectorXd grad;
            prob.Linearize(prob.poses(), H, grad);
            const Eigen::VectorXd d = H.diagonal();
            int below = 0;
            for (int i = 0; i < d.size(); ++i) below += 1e-4 * d(i) < 1e-6;
            std::printf("%-16s diag(H) at start: min %.3e max %.3e; %d / %ld entries have 1e-4*diag < 1e-6\n",
                        stem.c_str(), d.minCoeff(), d.maxCoeff(), below, long(d.size()));
            // Is GTSAM's answer a lower minimum of OUR cost, or a different cost? Evaluate ours at GTSAM's poses.
            OracleCase g = Load(p);
            for (const auto& [id, T] : g.opt) g.graph.UpdateNode(id, T, Mat6::Zero());
            const PgoProblem at_gtsam(g.graph, g.anchor);
            SparseMat Hg;
            Eigen::VectorXd gg;
            at_gtsam.Linearize(at_gtsam.poses(), Hg, gg);
            std::printf("%-16s our energy at GTSAM's optimum %.10g (GTSAM reports %.10g), max|grad| there %.1e\n",
                        stem.c_str(), at_gtsam.Energy(at_gtsam.poses()), g.final_error, gg.cwiseAbs().maxCoeff());
        }
        // Energy / gradient after N iterations, clamp on vs off: slow convergence, or converged but not terminating?
        for (const bool clamp : {true, false})
            for (const int n : {5, 10, 20, 50, 200}) {
                OracleCase c = Load(p);
                PgoSettings s;
                s.check_rel_energy_change = s.check_min_gradient = s.check_step_norm = false;
                s.max_iterations = n;
                s.recompute_covariance = false;
                s.clamp_diagonal = clamp;
                const PgoReport rep = SolvePoseGraph(c.graph, c.anchor, s);
                const PgoProblem prob(c.graph, c.anchor);
                SparseMat H;
                Eigen::VectorXd grad;
                prob.Linearize(prob.poses(), H, grad);
                std::printf("%-16s clamp %-3s after %3d its: energy %.12g max|grad| %.1e (%d lin)\n", stem.c_str(),
                            clamp ? "on" : "off", n, rep.final_energy, grad.cwiseAbs().maxCoeff(), rep.linearizations);
            }
        for (const bool clamp : {true, false}) {   // production tolerances (pgo_v1 defaults)
            OracleCase c = Load(p);
            PgoSettings s;
            s.recompute_covariance = false;
            s.clamp_diagonal = clamp;
            const PgoReport rep = SolvePoseGraph(c.graph, c.anchor, s);
            std::printf("%-16s pgo_v1 defaults, clamp %-3s: energy %.12g in %d its (found %d)\n", stem.c_str(),
                        clamp ? "on" : "off", rep.final_energy, rep.iterations, int(rep.found));
        }
        for (const auto& v : variants) {
            OracleCase c = Load(p);
            PgoSettings s;
            s.rel_energy_delta_tolerance = 1e-14;
            s.step_norm_tolerance = 1e-14;
            s.min_gradient_tolerance = 1e-9;
            s.max_iterations = 1000;
            s.recompute_covariance = false;
            s.clamp_diagonal = v.clamp;
            s.jacobi_scaling = v.jacobi;
            const PgoReport rep = SolvePoseGraph(c.graph, c.anchor, s);
            const PgoProblem prob(c.graph, c.anchor);
            SparseMat H;
            Eigen::VectorXd grad;
            prob.Linearize(prob.poses(), H, grad);
            const double gmax = grad.cwiseAbs().maxCoeff();
            const double de = (rep.final_energy - c.final_error) / std::max(1.0, c.final_error);
            std::printf("%-16s %-32s ours %.10g (%4d its, found %d, %4d lin, %4d/%4d fact failed, max|grad| %.1e) | "
                        "gtsam %.10g (%d its) | rel diff %+.1e\n",
                        stem.c_str(), v.name, rep.final_energy, rep.iterations, int(rep.found), rep.linearizations,
                        rep.failed_factorizations, rep.factorizations, gmax, c.final_error, c.iterations, de);
            EXPECT_LT(gmax, 1e-4 * std::max(1.0, rep.final_energy)) << stem << " / " << v.name << ": not stationary";
        }
    }
}

// The 2.5 init gate (design ROBUST_BACKEND_2.5.md §9, D70), fixed before measuring:
//  - start from DEAD RECKONING ($ARIADNE_PGO_SESYNC_DR: pgo_gtsam.py --init dr, NODE lines = the chain composition);
//  - reference minimum = GTSAM LM from GTSAM's chordal init ($ARIADNE_PGO_SESYNC_CHORDAL: --init chordal);
//  - failure set S = graphs where OUR LM from dead reckoning ends stationary but > 1e-6 relative above the reference;
//  - gate on every graph: chordal init + LM (pgo_v2: clamp_diagonal off) lands within 1e-6 relative of the reference
//    (or below it), stationary, in <= 50 iterations. Dead-reckoning and MST starts are reported as the ablation.
TEST(PgoInit, SeSyncDeadReckoningGate) {
    const char* dr_dir = std::getenv("ARIADNE_PGO_SESYNC_DR");
    const char* ch_dir = std::getenv("ARIADNE_PGO_SESYNC_CHORDAL");
    if (!dr_dir || !ch_dir || !fs::is_directory(dr_dir) || !fs::is_directory(ch_dir))
        GTEST_SKIP() << "set ARIADNE_PGO_SESYNC_DR and ARIADNE_PGO_SESYNC_CHORDAL (pgo_gtsam.py --init dr|chordal)";
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dr_dir))
        if (e.path().extension() == ".graph" && fs::exists(fs::path(ch_dir) / e.path().filename()))
            files.push_back(e.path());
    std::sort(files.begin(), files.end());
    ASSERT_FALSE(files.empty());
    PgoSettings v2 = PgoV2();   // production: pgo_v1 tolerances, OpenCV's absolute damping floor off (§2)
    v2.recompute_covariance = false;
    std::vector<std::string> failure_set;
    for (const auto& p : files) {
        const OracleCase ref = Load(fs::path(ch_dir) / p.filename());
        const double e_ref = ref.final_error;
        for (const PgoInit init : {PgoInit::kCurrent, PgoInit::kMst, PgoInit::kChordal}) {
            OracleCase c = Load(p);
            if (init == PgoInit::kCurrent) {
                const double e0 = PgoProblem(c.graph, c.anchor).Energy(PgoProblem(c.graph, c.anchor).poses());
                EXPECT_NEAR(e0, c.init_error, 1e-9 * std::max(1.0, c.init_error)) << p.filename() << ": not the "
                                                                                  << "same problem as GTSAM's";
            }
            InitReport ir;
            ApplyPoses(c.graph, InitialPoses(c.graph, c.anchor, init, &ir));
            const double e_start = PgoProblem(c.graph, c.anchor).Energy(PgoProblem(c.graph, c.anchor).poses());
            const PgoReport rep = SolvePoseGraph(c.graph, c.anchor, v2);
            // pgo_v2 stops on a 1e-6 relative energy change, so its end point need not satisfy a tight gradient test.
            // Polish with the tight tolerances of SeSyncBenchmarkGraphs: the production stop was AT the minimum iff the
            // polish lowers the energy by <= 1e-6 relative, and the polished point is the stationarity witness.
            PgoSettings tight = v2;
            tight.rel_energy_delta_tolerance = tight.step_norm_tolerance = 1e-14;
            tight.min_gradient_tolerance = 1e-9;
            tight.max_iterations = 1000;
            const PgoReport pol = SolvePoseGraph(c.graph, c.anchor, tight);
            const PgoProblem prob(c.graph, c.anchor);
            SparseMat H;
            Eigen::VectorXd grad;
            prob.Linearize(prob.poses(), H, grad);
            const double gmax = grad.cwiseAbs().maxCoeff();
            const bool stationary = gmax < 1e-4 * std::max(1.0, pol.final_energy);   // the polished point
            const bool stopped_at_min = rep.final_energy - pol.final_energy <= 1e-6 * std::max(1.0, pol.final_energy);
            const double rel = (rep.final_energy - e_ref) / std::max(1.0, e_ref);
            const double rel_pol = (pol.final_energy - e_ref) / std::max(1.0, e_ref);
            const char* name = init == PgoInit::kCurrent ? "dead reckoning" : init == PgoInit::kMst ? "MST" : "chordal";
            std::printf("%-15s %-14s start %.6e -> %.10g in %3d its (found %d; polished max|grad| %.1e) | ref %.10g | rel "
                        "%+.1e (polished %.10g)%s\n",
                        p.stem().c_str(), name, e_start, rep.final_energy, rep.iterations, int(rep.found), gmax, e_ref,
                        rel, pol.final_energy, init == PgoInit::kChordal ? (ir.det_fixes ? " | det fixes > 0" : "") : "");
            if (init == PgoInit::kCurrent) {
                std::printf("%-15s (GTSAM LM from dead reckoning: %.10g in %d its)\n", p.stem().c_str(),
                            c.final_error, c.iterations);
                // S: the minimum dead reckoning leads to (polished, stationary) is above the reference.
                if (stationary && rel_pol > 1e-6) failure_set.push_back(p.stem().string());
            }
            if (init == PgoInit::kChordal) {
                EXPECT_TRUE(stationary && stopped_at_min) << p.filename() << ": chordal + LM did not stop at a "
                                                          << "stationary point (polish moved it "
                                                          << rep.final_energy - pol.final_energy << ")";
                EXPECT_LE(rel, 1e-6) << p.filename() << ": chordal + LM misses the reference minimum";
                EXPECT_LE(rep.iterations, 50) << p.filename() << ": chordal + LM needs more than 50 iterations";
            }
        }
    }
    std::string s;
    for (const auto& f : failure_set) s += " " + f;
    std::printf("failure set S (dead reckoning ends > 1e-6 above the reference):%s\n", s.empty() ? " EMPTY" : s.c_str());
}

// Our chordal rotation stage vs GTSAM's computeOrientationsChordal (design ROBUST_BACKEND_2.5.md §3). GTSAM weights an
// edge by whiten(e1)[0] (sqrt of ONE axis's precision), we by SE-Sync's kappa: the two least-squares problems coincide
// only when every edge carries the same isotropic rotation noise, i.e. the even (diagonal-information) cases, which are
// gated; the odd (full correlated) cases are reported. Needs the 2.5 oracle files (CHORDAL_R lines).
TEST(PgoInit, ChordalRotationsAgreeWithGtsam) {
    const char* dir = std::getenv("ARIADNE_PGO_ORACLE");
    if (!dir || !fs::is_directory(dir)) GTEST_SKIP() << "set ARIADNE_PGO_ORACLE";
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".graph") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    int gated = 0;
    double worst_even = 0, worst_odd = 0;
    for (const auto& p : files) {
        const OracleCase c = Load(p);
        if (c.chordal.empty()) GTEST_SKIP() << p.filename() << " has no CHORDAL_R lines (pre-2.5 oracle files)";
        const int k = std::stoi(p.stem().string().substr(p.stem().string().rfind('_') + 1));
        const auto ours = InitialPoses(c.graph, c.anchor, PgoInit::kChordal);
        double dr = 0;
        for (const auto& [id, R] : c.chordal) dr = std::max(dr, core::SO3Log(ours.at(id).R * R.transpose()).norm());
        if (k % 2 == 0) {
            ++gated;
            worst_even = std::max(worst_even, dr);
            EXPECT_LT(dr, 1e-8) << p.filename() << ": chordal rotations differ from GTSAM's by " << dr << " rad";
        } else {
            worst_odd = std::max(worst_odd, dr);
        }
    }
    std::printf("chordal vs GTSAM: %d isotropic cases gated, worst %.1e rad; correlated cases (different weights, "
                "reported) worst %.1e rad\n", gated, worst_even, worst_odd);
}
