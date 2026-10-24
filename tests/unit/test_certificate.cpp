// The D74 certificate (design ROBUST_BACKEND_2.5.md §7). Definitions, fixed before the first measured run:
//   - "certified" = S_g + eta I_R factorises at eta = rel_eta ||M||_inf (certificate.hpp); a certified point carries
//     the bound f - f* <= |gap| + 3 n eta;
//   - known-optimal graphs = noise-free graphs and low-noise synthetic graphs (sigma 0.02 m / 0.005 rad), plus the
//     SE-Sync benchmark graphs (env-gated: ARIADNE_PGO_SESYNC);
//   - planted-outlier graphs = the same synthetic graphs plus 3 random loop edges (uniform SE(3)) KEPT IN THE COST
//     (plain least squares, no GNC): the fraction certified is reported (PHASE2_PLAN §3: a certificate proves
//     optimality of the given cost, not correctness). The hard gate is the one-directional property: a stationary
//     point with a strictly cheaper competitor (another start) must come out not certified;
//   - the same planted-outlier graphs after GNC (outliers at weight 0) must certify.
#include "slam/pose_graph/chordal_init.hpp"
#include "slam/robust/certificate.hpp"
#include "slam/robust/gnc.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>

using namespace ariadne;
using namespace ariadne::slam;
using core::Mat3;
using core::Vec3;
using core::Vec6;
namespace fs = std::filesystem;

namespace {

Mat6 Diag(double st, double sr) {
    Vec6 d;
    d << st * st, st * st, st * st, sr * sr, sr * sr, sr * sr;
    return d.asDiagonal();
}

SE3 RandomPose(std::mt19937& g, double half_extent) {
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> u(-half_extent, half_extent);
    Eigen::Quaterniond q{n(g), n(g), n(g), n(g)};
    q.normalize();
    SE3 T;
    T.R = q.toRotationMatrix();
    T.t = Vec3{u(g), u(g), u(g)};
    return T;
}

Vec6 Draw(std::mt19937& g, const Mat6& cov) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec6 z;
    for (int i = 0; i < 6; ++i) z(i) = n(g);
    return Eigen::LLT<Mat6>(cov).matrixL() * z;
}

struct Synthetic {
    std::map<uint64_t, SE3> truth, dead_reckoning;
    PoseGraph g;
    std::vector<size_t> outliers;
};

/// A random walk of n poses, odometry chain + `loops` random loops, all with left noise `cov`; nodes at the truth.
Synthetic Make(std::mt19937& gen, int n, int loops, const Mat6& cov, int outliers = 0, bool exact = false) {
    Synthetic s;
    SE3 T;
    std::normal_distribution<double> nz(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            Vec6 step;
            step << 1.0, 0.3 * nz(gen), 0.1 * nz(gen), 0.1 * nz(gen), 0.1 * nz(gen), 0.5 * nz(gen);
            T = T * core::SE3Exp(step);
        }
        s.truth[uint64_t(i)] = T;
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        nd.T_W_C = T;
        s.g.AddNode(nd);
    }
    auto edge = [&](uint64_t a, uint64_t b, bool loop) {
        const SE3 Z = (exact ? SE3() : core::SE3Exp(Draw(gen, cov))) * (s.truth[a].inverse() * s.truth[b]);
        PoseEdge e = OdometryEdge(a, b, Z, cov);
        if (loop) e.type = EdgeType::kLocalLoop;
        s.g.AddEdge(e);
    };
    for (int i = 1; i < n; ++i) edge(uint64_t(i - 1), uint64_t(i), false);
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int k = 0; k < loops;) {
        const int a = pick(gen), b = pick(gen);
        if (std::abs(a - b) < 2) continue;
        edge(uint64_t(std::min(a, b)), uint64_t(std::max(a, b)), true);
        ++k;
    }
    for (int k = 0; k < outliers;) {
        const int a = pick(gen), b = pick(gen);
        if (std::abs(a - b) < 2) continue;
        PoseEdge e = OdometryEdge(uint64_t(a), uint64_t(b), RandomPose(gen, 5.0), cov);
        e.type = EdgeType::kLocalLoop;
        s.g.AddEdge(e);
        s.outliers.push_back(s.g.edges().size() - 1);
        ++k;
    }
    SE3 est = s.truth[0];
    s.dead_reckoning[0] = est;
    for (int i = 1; i < n; ++i) s.dead_reckoning[uint64_t(i)] = est = est * s.g.edges()[size_t(i - 1)].Z;
    return s;
}

std::map<uint64_t, SE3> Poses(const PoseGraph& g) {
    std::map<uint64_t, SE3> p;
    for (const auto& [id, n] : g.nodes()) p[id] = n.T_W_C;
    return p;
}

void SetPoses(PoseGraph& g, const std::map<uint64_t, SE3>& p) {
    for (const auto& [id, T] : p) g.UpdateNode(id, T, Mat6::Zero());
}

/// Eigenvalues of a small sparse symmetric matrix with OpenCV's dense solver (the oracle).
std::vector<double> DenseEigen(const SparseMat& S) {
    cv::Mat_<double> A(int(S.rows()), int(S.cols()), 0.0), w;
    for (int k = 0; k < S.outerSize(); ++k)
        for (SparseMat::InnerIterator it(S, k); it; ++it) A(int(it.row()), int(it.col())) = it.value();
    cv::eigen(A, w);   // descending
    std::vector<double> out(w.begin(), w.end());
    std::sort(out.begin(), out.end());
    return out;
}

/// The reduced rotation matrix of SE-Sync (translations eliminated) from S, densely: C - B^T A^+ B on the full
/// (ungrounded) blocks, A^+ the pseudo-inverse of the translation Laplacian.
Eigen::MatrixXd ReducedDense(const SparseMat& S, int n) {
    const Eigen::MatrixXd D(S);
    const Eigen::MatrixXd A = D.topLeftCorner(n, n), B = D.topRightCorner(n, 3 * n), C = D.bottomRightCorner(3 * n, 3 * n);
    const Eigen::MatrixXd Ap = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(A).pseudoInverse();
    return C - B.transpose() * Ap * B;
}

/// A ring of n nodes, identity measurements, rotations wound once about z: a stationary, non-global point of f.
PoseGraph WoundRing(int n) {
    PoseGraph g;
    for (int i = 0; i < n; ++i) {
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        nd.T_W_C.R = Eigen::AngleAxisd(2 * M_PI * i / n, Vec3::UnitZ()).toRotationMatrix();
        g.AddNode(nd);
    }
    for (int i = 0; i < n; ++i) g.AddEdge(OdometryEdge(uint64_t(i), uint64_t((i + 1) % n), SE3(), Diag(0.05, 0.02)));
    return g;
}

CertificateOptions NoRefine() {
    CertificateOptions o;
    o.refine = false;
    return o;
}

}  // namespace

TEST(Certificate, ObjectiveIsTheTraceFormAndGradientMatchesFiniteDifferences) {
    std::mt19937 gen(1);
    Synthetic s = Make(gen, 8, 4, Diag(0.05, 0.02));
    std::map<uint64_t, SE3> p = s.truth;
    for (auto& [id, T] : p) T = core::SE3Exp(Draw(gen, Diag(0.3, 0.2))) * T;
    const ChordalSurrogate cs(s.g, 0);
    const Eigen::MatrixXd X = cs.StackX(p);
    const double f = cs.Objective(p), tr = (X * cs.M() * X.transpose()).trace();
    EXPECT_NEAR(tr, f, 1e-10 * f);
    SparseMat H;
    Eigen::VectorXd grad;
    EXPECT_NEAR(cs.Linearize(p, H, grad), 0.5 * f, 1e-12 * f);
    double worst = 0;
    for (int i = 0; i < grad.size(); ++i) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(grad.size());
        x(i) = 1e-6;
        const double ep = 0.5 * cs.Objective(cs.Oplus(p, x));
        x(i) = -1e-6;
        const double em = 0.5 * cs.Objective(cs.Oplus(p, x));
        worst = std::max(worst, std::fabs((ep - em) / 2e-6 - grad(i)) / std::max(1.0, std::fabs(grad(i))));
    }
    EXPECT_LT(worst, 1e-6);
}

TEST(Certificate, NoiseFreeGraphHasTheGaugeNullSpaceAndCertifies) {
    std::mt19937 gen(2);
    const Synthetic s = Make(gen, 5, 3, Diag(0.05, 0.02), 0, true);
    const ChordalSurrogate cs(s.g, 0);
    const SparseMat S = cs.CertificateMatrix(cs.StackX(Poses(s.g)));
    double normM = 0;
    for (double v : DenseEigen(cs.M())) normM = std::max(normM, std::fabs(v));
    const std::vector<double> ev = DenseEigen(S);
    int zeros = 0;
    for (double v : ev) zeros += std::fabs(v) < 1e-9 * normM;
    EXPECT_EQ(zeros, 4) << "explicit form: 3 global-rotation + 1 translation-shift null directions";
    EXPECT_GT(ev[4], 1e-6 * normM);
    const Eigen::VectorXd er = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(ReducedDense(S, 5)).eigenvalues();
    int rzeros = 0;
    for (int i = 0; i < er.size(); ++i) rzeros += std::fabs(er(i)) < 1e-9 * normM;
    EXPECT_EQ(rzeros, 3) << "reduced form: SE-Sync's d zero eigenvalues";
    EXPECT_GT(er(3), 1e-6 * normM);
    // The other sign (arch §14.8's Lambda - Q~) is not PSD: the sign is fixed here, not by convention.
    const std::vector<double> wrong = DenseEigen(SparseMat(-S));
    EXPECT_LT(wrong.front(), -1e-3 * normM);
    const CertificateReport r = CertifyPoseGraph(s.g, 0);
    std::printf("noise-free 5 nodes: %s | gap %.2e, eta_min %.2e\n", r.verdict.c_str(), r.duality_gap,
                r.eta_min_passing);
    EXPECT_TRUE(r.certified);
    EXPECT_LT(r.objective, 1e-12 * normM);
}

// The explicit (grounded, rotation-shifted) Cholesky test decides exactly what the reduced test decides; shifting
// the translation blocks too would over-certify (advisor review). Checked on the first planted-outlier graph that is
// not certified (it has translation coupling, unlike the wound ring, where B = 0).
TEST(Certificate, GroundedTestEqualsTheReducedTest) {
    for (int seed = 0; seed < 10; ++seed) {
        std::mt19937 gen(200u + unsigned(seed));
        Synthetic s = Make(gen, 60, 30, Diag(0.02, 0.005), 3);
        std::map<uint64_t, SE3> pt;
        const CertificateReport probe = CertifyPoseGraph(s.g, 0, {}, &pt);
        if (probe.certified) continue;
        SetPoses(s.g, pt);
        const ChordalSurrogate cs(s.g, 0);
        const SparseMat S = cs.CertificateMatrix(cs.StackX(pt));
        const double lmin = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(ReducedDense(S, cs.n())).eigenvalues()(0);
        const double lfull = DenseEigen(S).front();
        ASSERT_LT(lmin, 0.0);
        for (const double f : {0.95, 1.05}) {
            CertificateOptions o = NoRefine();
            o.rel_eta = f * -lmin / probe.norm_M;
            EXPECT_EQ(CertifyPoseGraph(s.g, 0, o).certified, f > 1.0) << "eta = " << f << " x (-lambda_min reduced)";
        }
        std::printf("planted-outlier seed %d: reduced lambda_min %.4e, explicit-form lambda_min %.4e (a full shift "
                    "would pass at %.3g x the reduced eta); Ritz %.4e\n",
                    seed, lmin, lfull, lfull / lmin, probe.lambda_min_ritz);
        EXPECT_NEAR(probe.lambda_min_ritz, lmin, 1e-3 * std::fabs(lmin));
        return;
    }
    FAIL() << "no planted-outlier seed was not certified";
}

TEST(Certificate, StationaryNonGlobalPointIsNotCertified) {
    const PoseGraph g = WoundRing(12);
    const CertificateReport r = CertifyPoseGraph(g, 0, NoRefine());
    std::printf("wound ring 12: f %.4g, max|grad_R| %.1e: %s\n", r.objective, r.max_rot_gradient, r.verdict.c_str());
    EXPECT_LT(r.max_rot_gradient, 1e-9 * r.norm_M);
    EXPECT_GT(r.objective, 1.0);
    EXPECT_FALSE(r.certified);
    EXPECT_LT(r.lambda_min_ritz, 0.0);
}

TEST(Certificate, PerturbedPointIsNotCertified) {
    std::mt19937 gen(3);
    Synthetic s = Make(gen, 20, 10, Diag(0.02, 0.005));
    std::map<uint64_t, SE3> p = s.truth;
    for (auto& [id, T] : p)
        if (id) T.R = core::SO3Exp(Draw(gen, Diag(0.0, 0.3)).tail<3>()) * T.R;
    SetPoses(s.g, p);
    const CertificateReport r = CertifyPoseGraph(s.g, 0, NoRefine());
    std::printf("perturbed (0.3 rad, not refined): %s\n", r.verdict.c_str());
    EXPECT_FALSE(r.certified);
}

TEST(Certificate, LowNoiseGraphsCertify) {
    int ok = 0;
    double worst_bound = 0;
    for (int seed = 0; seed < 10; ++seed) {
        std::mt19937 gen(100u + unsigned(seed));
        Synthetic s = Make(gen, 60, 30, Diag(0.02, 0.005));
        const CertificateReport r = CertifyPoseGraph(s.g, 0);
        ok += r.certified;
        worst_bound = std::max(worst_bound, r.suboptimality_bound / r.objective);
        EXPECT_TRUE(r.certified) << "seed " << seed << ": " << r.verdict;
        EXPECT_LT(r.max_rot_gradient, 1e-6 * std::max(1.0, r.objective));
    }
    std::printf("low noise: %d / 10 certified, worst bound %.2e of f\n", ok, worst_bound);
}

TEST(Certificate, PlantedOutliersFractionReportedAndOneDirectional) {
    int certified[3] = {0, 0, 0}, pairs = 0, after_gnc = 0;
    const char* names[3] = {"truth", "dead reckoning", "chordal"};
    for (int seed = 0; seed < 10; ++seed) {
        std::mt19937 gen(200u + unsigned(seed));
        const Synthetic s = Make(gen, 60, 30, Diag(0.02, 0.005), 3);
        CertificateReport r[3];
        for (int k = 0; k < 3; ++k) {
            PoseGraph g = s.g;
            if (k == 1) SetPoses(g, s.dead_reckoning);
            if (k == 2) ApplyPoses(g, InitialPoses(g, 0, PgoInit::kChordal));
            r[k] = CertifyPoseGraph(g, 0);
            certified[k] += r[k].certified;
        }
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                if (r[a].objective > r[b].objective * (1 + 1e-6) + 1e-9) {
                    ++pairs;
                    EXPECT_FALSE(r[a].certified) << "seed " << seed << ": " << names[a] << " f " << r[a].objective
                                                 << " certified though " << names[b] << " reached " << r[b].objective;
                }
        PoseGraph g = s.g;
        SolveGnc(g, 0);
        bool rejected = true;
        for (size_t e : s.outliers) rejected = rejected && g.edges()[e].gnc_weight <= kInactiveGncWeight;
        const CertificateReport c = CertifyPoseGraph(g, 0);
        after_gnc += c.certified && rejected;
        EXPECT_TRUE(rejected && c.certified) << "seed " << seed << " after GNC: " << c.verdict;
    }
    std::printf("planted outliers (3 per graph, kept in the cost), certified out of 10: from truth %d, dead reckoning "
                "%d, chordal %d | %d strictly-worse stationary points, all not certified | after GNC %d / 10\n",
                certified[0], certified[1], certified[2], pairs, after_gnc);
}

// Flag off: the loop backend never calls the certificate (loop_backend.cpp), and it is read-only when on; the
// trace gains a "certificate" object only when it ran (intra_loop_closure.hpp). Here: read-only on the graph.
TEST(Certificate, DoesNotModifyTheGraph) {
    std::mt19937 gen(4);
    const Synthetic s = Make(gen, 30, 10, Diag(0.02, 0.005));
    PoseGraph g = s.g;
    const uint64_t v = g.version();
    CertifyPoseGraph(g, 0);
    EXPECT_EQ(g.version(), v);
    for (const auto& [id, n] : g.nodes()) {
        EXPECT_EQ(std::memcmp(n.T_W_C.R.data(), s.g.nodes().at(id).T_W_C.R.data(), sizeof(double) * 9), 0);
        EXPECT_EQ(std::memcmp(n.T_W_C.t.data(), s.g.nodes().at(id).T_W_C.t.data(), sizeof(double) * 3), 0);
    }
}

// The SE-Sync benchmark graphs (models.sh pgo-sesync; NOT in the pre-merge gate). Start: our chordal init (D70);
// refine on the chordal cost; the objective must reproduce SE-Sync's Table 1 (4 significant figures) and certify.
// ARIADNE_CERT_GRAPHS = comma-separated stems (default: all). cubicle and rim are included: their full 6x6
// information is indefinite (refused by the anisotropic path), but kappa/tau need only the diagonal blocks to be
// positive definite, which IsotropicFromInformation checks per edge.
TEST(Certificate, SeSyncBenchmarkGraphs) {
    const char* dir = std::getenv("ARIADNE_PGO_SESYNC");
    if (!dir || !fs::is_directory(dir)) GTEST_SKIP() << "set ARIADNE_PGO_SESYNC (infra/scripts/models.sh pgo-sesync)";
    const std::map<std::string, double> table1{{"sphere2500", 1.687e3}, {"torus3D", 2.423e4}, {"grid3D", 8.432e4},
                                               {"parking-garage", 1.263}, {"cubicle", 7.171e2}, {"rim", 5.461e3}};
    std::string which = std::getenv("ARIADNE_CERT_GRAPHS") ? std::getenv("ARIADNE_CERT_GRAPHS") : "";
    for (const auto& e : fs::directory_iterator(dir)) {
        const fs::path p = e.path();
        if (p.extension() != ".graph") continue;
        const std::string stem = p.stem().string();
        if (!which.empty() && ("," + which + ",").find("," + stem + ",") == std::string::npos) continue;
        std::ifstream f(p);
        std::string line;
        PoseGraph g;
        uint64_t anchor = 0;
        std::vector<PoseEdge> edges;
        auto pose = [](std::istringstream& in) {
            double x, y, z, qw, qx, qy, qz;
            in >> x >> y >> z >> qw >> qx >> qy >> qz;
            SE3 T;
            T.R = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
            T.t = Vec3(x, y, z);
            return T;
        };
        while (std::getline(f, line)) {
            std::istringstream in(line);
            std::string tag;
            in >> tag;
            if (tag == "ANCHOR") {
                in >> anchor;
            } else if (tag == "NODE") {
                PoseNode n;
                in >> n.keyframe;
                n.T_W_C = pose(in);
                g.AddNode(n);
            } else if (tag == "EDGE") {
                PoseEdge pe;
                pe.type = EdgeType::kOdometry;
                in >> pe.from >> pe.to;
                pe.Z = pose(in);
                for (int r = 0; r < 6; ++r)
                    for (int c = r; c < 6; ++c) {
                        in >> pe.information(r, c);
                        pe.information(c, r) = pe.information(r, c);
                    }
                edges.push_back(pe);
            }
        }
        for (PoseEdge pe : edges) {   // kappa/tau read only the diagonal blocks: dropping the coupling keeps them
            pe.information.topRightCorner<3, 3>().setZero();
            pe.information.bottomLeftCorner<3, 3>().setZero();
            g.AddEdge(pe);
        }
        const auto t0 = std::chrono::steady_clock::now();
        ApplyPoses(g, InitialPoses(g, anchor, PgoInit::kChordal));
        const CertificateReport r = CertifyPoseGraph(g, anchor);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto ref = table1.find(stem);
        std::printf("%-16s %6d nodes %6d edges: f %.6g (chordal init %.6g; Table 1 %s), %d its, max|grad_R| %.1e, "
                    "||M|| %.3g, eta_min %.2e, abs-eta %s | %s | %.1f s\n",
                    stem.c_str(), r.nodes, r.edges, r.objective, r.objective_input,
                    ref == table1.end() ? "-" : std::to_string(ref->second).c_str(), r.iterations, r.max_rot_gradient,
                    r.norm_M, r.eta_min_passing, r.certified_abs ? "certified" : "not certified", r.verdict.c_str(), s);
        if (ref != table1.end()) EXPECT_NEAR(r.objective, ref->second, 5e-4 * ref->second) << stem;
        EXPECT_TRUE(r.certified) << stem;
    }
}
