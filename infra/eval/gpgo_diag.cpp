// Diagnostic for GravityPgo.RecoversTheReferenceGravityErrorAndReducesTilt (x86 0.62 deg vs arm64 2.71 deg).
// Not a test: prints input digests, the per-call LM trace, and sensitivity experiments.
//   gpgo_diag trace          one solve with the test's chain (seed 3), every backend call logged
//   gpgo_diag tight          same chain, default vs tight tolerances (where is the true optimum?)
//   gpgo_diag ulp N          N solves of the same chain with each node's initial t perturbed by +-1 ulp
//   gpgo_diag seeds N        seeds 0..N-1: gerr at default and tight settings
//   gpgo_diag perturb N EPS  every initial pose and prior perturbed by EPS (relative), N draws
//   gpgo_diag pure           LevMarqBase's loop re-implemented with scalar loops (no cv::Mat arithmetic, no HAL)
//   gpgo_diag halcheck       the driver's CV_64F cv::Mat ops against scalar loops
//   gpgo_diag noisy          PgoLevMarq.NoisyGraphReachesAStationaryPoint's fixture and settings
// ORDER=0|1 fixes the order of the two prior-noise draws (unset: the compiler's order, as in the test).
#include "slam/pose_graph/pgo_levmarq.hpp"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

using namespace ariadne;
using namespace ariadne::slam;
using core::Vec3;
using core::Vec6;

namespace {

Mat6 Diag(double st, double sr) {
    Vec6 d;
    d << st * st, st * st, st * st, sr * sr, sr * sr, sr * sr;
    return d.asDiagonal();
}
const Vec3 kDown(0, 0, -1);
int kOrder = -1;   // env ORDER, see Make()
double kGenScale = 1.0;   // env GENSCALE: prior noise drawn at GENSCALE * sigma (information unchanged)
core::Mat3 Rot(const Vec3& aa) { return core::SO3Exp(aa); }
struct Chain {
    std::vector<SE3> truth;
    PoseGraph g;
};
Chain Make(std::mt19937& gen, int n, int every, double sigma_deg, double ref_err_deg) {   // copied from the test
    Chain c;
    std::normal_distribution<double> nz(0.0, 1.0);
    SE3 T;
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            Vec6 step;
            step << 0.4, 0.05 * nz(gen), 0.0, 0.02 * nz(gen), 0.02 * nz(gen), 0.15 * nz(gen);
            T = T * core::SE3Exp(step);
        }
        c.truth.push_back(T);
    }
    const Mat6 odo = Diag(0.01, 0.01);
    SE3 est = c.truth[0];
    for (int i = 0; i < n; ++i) {
        Vec6 noise;
        for (int k = 0; k < 6; ++k) noise(k) = nz(gen);
        const SE3 Z = core::SE3Exp(Eigen::LLT<Mat6>(odo).matrixL() * noise) *
                      (i ? c.truth[size_t(i - 1)].inverse() * c.truth[size_t(i)] : SE3());
        if (i > 0) est = est * Z;
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        nd.T_W_C = est;
        c.g.AddNode(nd);
        if (i > 0) c.g.AddEdge(OdometryEdge(uint64_t(i - 1), uint64_t(i), Z, odo));
    }
    const double s = sigma_deg * M_PI / 180.0;
    const Vec3 g_ref = Rot(Vec3(ref_err_deg * M_PI / 180.0, 0, 0)) * kDown;
    for (int i = 0; i < n; i += every) {
        GravityPrior p;
        p.keyframe = uint64_t(i);
        const Vec3 gC = c.truth[size_t(i)].R.transpose() * kDown;
        if (kOrder < 0) {   // as the test: the two draws' order is the compiler's (unspecified, [expr.call])
            p.g_C = (Rot(kGenScale * s * Vec3(nz(gen), nz(gen), 0)) * gC).normalized();
        } else {            // ORDER=0: first draw -> x (GCC aarch64), ORDER=1: first draw -> y (GCC x86-64)
            const double n1 = nz(gen), n2 = nz(gen);
            p.g_C = (Rot(kGenScale * s * (kOrder ? Vec3(n2, n1, 0) : Vec3(n1, n2, 0))) * gC).normalized();
        }
        p.g_ref_W = g_ref.normalized();
        p.information = Eigen::Matrix2d::Identity() / (s * s);
        c.g.AddGravityPrior(p);
    }
    return c;
}
double TiltErrorDeg(const Chain& c) {
    double e = 0;
    for (const auto& [id, n] : c.g.nodes()) {
        const Vec3 a = n.T_W_C.R.transpose() * kDown, b = c.truth[size_t(id)].R.transpose() * kDown;
        e += std::pow(std::acos(std::clamp(a.dot(b), -1.0, 1.0)), 2);
    }
    return std::sqrt(e / double(c.g.nodes().size())) * 180.0 / M_PI;
}
PgoSettings On() {
    PgoSettings s = PgoV2();
    s.recompute_covariance = false;
    s.gravity.enabled = true;
    return s;
}
PgoSettings Tight() {
    PgoSettings s = On();
    s.step_norm_tolerance = 1e-14;
    s.rel_energy_delta_tolerance = 1e-16;
    s.min_gradient_tolerance = 1e-14;
    s.max_iterations = 5000;
    return s;
}

uint64_t Fnv(uint64_t h, const void* p, size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
uint64_t Digest(const PoseGraph& g) {   // every input double, bit-exact
    uint64_t h = 1469598103934665603ull;
    for (const auto& [id, n] : g.nodes()) {
        h = Fnv(h, n.T_W_C.R.data(), 72);
        h = Fnv(h, n.T_W_C.t.data(), 24);
    }
    for (const auto& e : g.edges()) {
        h = Fnv(h, e.Z.R.data(), 72);
        h = Fnv(h, e.Z.t.data(), 24);
        h = Fnv(h, e.information.data(), 288);
    }
    for (const auto& p : g.gravity_priors()) {
        h = Fnv(h, p.g_C.data(), 24);
        h = Fnv(h, p.g_ref_W.data(), 24);
        h = Fnv(h, p.information.data(), 32);
    }
    return h;
}
uint64_t DigestTruth(const Chain& c) {
    uint64_t h = 1469598103934665603ull;
    for (const auto& T : c.truth) {
        h = Fnv(h, T.R.data(), 72);
        h = Fnv(h, T.t.data(), 24);
    }
    return h;
}
double Gerr(const Vec3& g) { return std::acos(std::clamp(g.dot(kDown), -1.0, 1.0)) * 180.0 / M_PI; }

// The production PgoBackend (pgo_levmarq.cpp, anonymous namespace) with a log line per call.
class TraceBackend : public cv::detail::LevMarqBackend {
public:
    TraceBackend(PgoProblem& p, bool log) : p_(p), log_(log), current_(p.poses()), probe_(p.poses()), gcur_(p.gravity()), gprobe_(p.gravity()) {}
    bool enableGeo() override { return false; }
    bool calcFunc(double& energy, bool calcEnergy, bool calcJacobian) override {
        if (calcJacobian) energy = p_.Linearize(probe_, gprobe_, H_, g_);
        else if (calcEnergy) energy = p_.Energy(probe_, gprobe_);
        if (log_)
            std::printf("  calc%s E=%.17g (%a) gprobe=%.6f deg%s\n", calcJacobian ? "J" : "E", energy, energy,
                        Gerr(gprobe_[0]), calcJacobian ? (" |g|=" + std::to_string(g_.norm())).c_str() : "");
        return std::isfinite(energy);
    }
    void currentOplusX(const cv::Mat_<double>& x, bool) override {
        Eigen::VectorXd v(x.rows);
        for (int i = 0; i < x.rows; ++i) v(i) = x(i);
        probe_ = p_.OplusX(current_, v);
        gprobe_ = p_.OplusGravity(gcur_, v);
        if (log_) std::printf("  step |x|=%.6e x_g=(%.6e, %.6e)\n", v.norm(), v(v.size() - 2), v(v.size() - 1));
    }
    void prepareVars() override { probe_ = current_; gprobe_ = gcur_; }
    const cv::Mat_<double> getJtb() override { return M(g_); }
    const cv::Mat_<double> getDiag() override { return M(H_.diagonal()); }
    void setDiag(const cv::Mat_<double>& d) override {
        // pgo_v2 (no clamp): d_i = H_ii (1 + lambda) for every i, so d_i / H_ii - 1 must be one number.
        double lo = 1e300, hi = -1e300;
        for (int i = 0; i < d.rows; ++i) {
            const double h = H_.coeff(i, i);
            if (h != 0) { lo = std::min(lo, d(i) / h - 1); hi = std::max(hi, d(i) / h - 1); }
            H_.coeffRef(i, i) = d(i);
        }
        if (log_) std::printf("  setDiag lambda in [%.17g, %.17g]\n", lo, hi);
    }
    void doJacobiScaling(const cv::Mat_<double>& di) override {
        Eigen::VectorXd d(di.rows);
        for (int i = 0; i < di.rows; ++i) d(i) = di(i);
        for (int k = 0; k < H_.outerSize(); ++k)
            for (SparseMat::InnerIterator it(H_, k); it; ++it) it.valueRef() *= d(it.row()) * d(it.col());
        g_ = g_.cwiseProduct(d);
    }
    bool decompose() override { return ldlt_.Factorize(H_); }
    bool solveDecomposed(const cv::Mat_<double>& right, cv::Mat_<double>& x) override {
        if (!ldlt_.ok()) return false;
        Eigen::VectorXd r(right.rows);
        for (int i = 0; i < right.rows; ++i) r(i) = right(i);
        const Eigen::VectorXd sol = ldlt_.Solve(-r);
        if (!sol.allFinite()) return false;
        if (log_) {   // the damped system's own residual: an accurate LDLT gives ~1e-15
            const Eigen::VectorXd res = H_.selfadjointView<Eigen::Lower>() * sol + r;
            const Eigen::VectorXd full = H_ * sol + r;
            std::printf("  solve rel residual %.3e (lower-sym %.3e) min pivot %.6e\n", full.norm() / r.norm(),
                        res.norm() / r.norm(), ldlt_.min_pivot());
        }
        x = M(sol);
        return true;
    }
    bool calcJtbv(cv::Mat_<double>&) override { return false; }
    void acceptProbe() override {
        current_ = probe_;
        gcur_ = gprobe_;
        if (log_) std::printf("  ACCEPT g=%.6f deg\n", Gerr(gcur_[0]));
    }
    const std::vector<Vec3>& gravity() const { return gcur_; }

private:
    static cv::Mat_<double> M(const Eigen::VectorXd& v) {
        cv::Mat_<double> m(int(v.size()), 1);
        for (Eigen::Index i = 0; i < v.size(); ++i) m(int(i)) = v(i);
        return m;
    }
    PgoProblem& p_;
    bool log_;
    std::map<uint64_t, SE3> current_, probe_;
    std::vector<Vec3> gcur_, gprobe_;
    SparseMat H_;
    Eigen::VectorXd g_;
    GuardedLdlt ldlt_;
};

// LevMarqBase::optimize (opencv geometry/src/levmarq.cpp @ 5.0.0) for the pgo_v2 settings, every vector op a scalar
// loop: no cv::Mat arithmetic, so no HAL. Same order of operations as the driver.
struct PureOut {
    int iters;
    bool found;
    double energy, gerr;
};
PureOut PureLm(PgoProblem& p, const PgoSettings& s, bool log) {
    TraceBackend b(p, false);
    b.prepareVars();
    double energy = 0;
    b.calcFunc(energy, true, true);
    double oldEnergy = energy, lmUp = s.initial_lm_up_factor, lambda = s.initial_lambda;
    int iter = 0;
    bool done = false, found = false;
    while (!done) {
        const cv::Mat_<double> jtb = b.getJtb();
        double gmax = 0;
        for (int i = 0; i < jtb.rows; ++i) gmax = std::max(gmax, std::fabs(jtb(i)));
        const cv::Mat_<double> diag = b.getDiag();
        bool enough = false;
        while (!enough && !done) {
            cv::Mat_<double> lmDiag(diag.rows, 1), jtjDiag(diag.rows, 1);
            for (int i = 0; i < diag.rows; ++i) { lmDiag(i) = diag(i) * lambda; jtjDiag(i) = lmDiag(i) + diag(i); }
            b.setDiag(jtjDiag);
            cv::Mat_<double> x(jtb.rows, 1);
            const bool solved = b.decompose() && b.solveDecomposed(jtb, x);
            double costChange = 0, jac = 0, q = 0, xn = 0;
            if (solved) {
                for (int i = 0; i < x.rows; ++i) { jac += x(i) * (jtb(i) - lmDiag(i) * x(i)); xn += x(i) * x(i); }
                jac *= -0.5;
                b.currentOplusX(x, false);
                b.calcFunc(energy, true, false);
                costChange = oldEnergy - energy;
                q = costChange / jac;
            }
            if (!solved || costChange < 0) {
                lambda *= lmUp;
                if (s.up_double) lmUp *= 2.0;
            } else {
                enough = true;
                lambda *= std::max(1.0 / s.initial_lm_down_factor, 1.0 - std::pow(2.0 * q - 1.0, 3));
                lmUp = s.initial_lm_up_factor;
                const bool sg = gmax < s.min_gradient_tolerance, ss = xn < s.step_norm_tolerance,
                           se = costChange / energy < s.rel_energy_delta_tolerance;
                b.acceptProbe();
                oldEnergy = energy;
                found = (s.check_min_gradient && sg) || (s.check_step_norm && ss) || (s.check_rel_energy_change && se);
                done = found;
            }
            if (log) std::printf("  pure it %d E %.17g q %.6f lambda %.6e\n", iter, energy, q, lambda);
            ++iter;
            done = done || iter >= s.max_iterations || lambda >= 1e32;
        }
        if (!done) b.calcFunc(energy, true, true);
    }
    return {iter, found, oldEnergy, Gerr(b.gravity()[0])};
}

// The driver's cv::Mat ops on CV_64F against scalar loops; prints the worst relative error per op (0 = bit-exact).
void HalCheck() {
    std::mt19937 g(11);
    std::uniform_real_distribution<double> e(-6, 6), u(-1, 1);
    const int n = 1206;
    cv::Mat_<double> a(n, 1), b(n, 1);
    for (int i = 0; i < n; ++i) { a(i) = std::pow(10.0, e(g)); b(i) = u(g) * std::pow(10.0, e(g)); }
    const double lam = 1.2345678901234567e-4;
    auto worst = [&](const char* name, const cv::Mat_<double>& got, auto ref) {
        double w = 0;
        for (int i = 0; i < n; ++i) { const double r = ref(i); w = std::max(w, r == 0 ? std::fabs(got(i)) : std::fabs(got(i) - r) / std::fabs(r)); }
        std::printf("halcheck %-22s worst rel err %.3e\n", name, w);
    };
    cv::Mat_<double> s1 = a * lam;
    worst("a*lambda (MatExpr)", s1, [&](int i) { return a(i) * lam; });
    cv::Mat_<double> s2 = s1 + a;
    worst("lmDiag+diag", s2, [&](int i) { return s1(i) + a(i); });
    cv::Mat_<double> s3 = b.mul(a);
    worst("x.mul(y)", s3, [&](int i) { return b(i) * a(i); });
    cv::Mat_<double> s4 = b - a.mul(b);
    worst("jtb - lmDiag.mul(x)", s4, [&](int i) { return b(i) - a(i) * b(i); });
    double sref = 0, l2 = 0, inf = 0;
    for (int i = 0; i < n; ++i) { sref += b(i); l2 += b(i) * b(i); inf = std::max(inf, std::fabs(b(i))); }
    std::printf("halcheck cv::sum rel err %.3e (got %.17g ref %.17g)\n", std::fabs(cv::sum(b)[0] - sref) / std::fabs(sref), cv::sum(b)[0], sref);
    std::printf("halcheck norm L2SQR rel err %.3e  norm INF rel err %.3e\n", std::fabs(cv::norm(b, cv::NORM_L2SQR) - l2) / l2,
                std::fabs(cv::norm(b, cv::NORM_INF) - inf) / inf);
    cv::Mat_<double> sq;
    cv::sqrt(a, sq);
    worst("cv::sqrt", sq, [&](int i) { return std::sqrt(a(i)); });
    std::printf("halcheck useOptimized %d\n", int(cv::useOptimized()));
}


// PgoLevMarq.NoisyGraphReachesAStationaryPoint's fixture (test_pgo_levmarq.cpp), copied verbatim.
core::SE3 RandomPose(std::mt19937& g, double rot, double trans) {
    std::normal_distribution<double> n(0.0, 1.0);
    core::Vec6 xi;
    xi << trans * n(g), trans * n(g), trans * n(g), rot * n(g), rot * n(g), rot * n(g);
    return core::SE3Exp(xi);
}
PoseGraph LoopGraph(std::mt19937& g, int n, int loops, double noise) {
    std::vector<core::SE3> truth{core::SE3::Identity()};
    for (int i = 1; i < n; ++i) truth.push_back(truth.back() * RandomPose(g, 0.3, 1.0));
    std::normal_distribution<double> nn(0.0, noise);
    auto noisy = [&](const core::SE3& Z) { core::Vec6 d; for (int k = 0; k < 6; ++k) d(k) = nn(g); return core::SE3Exp(d) * Z; };
    PoseGraph G;
    for (int i = 0; i < n; ++i) { PoseNode nd; nd.keyframe = uint64_t(i); nd.T_W_C = truth[size_t(i)]; G.AddNode(nd); }
    auto add = [&](int a, int b) {
        PoseEdge e; e.type = EdgeType::kLocalLoop; e.from = uint64_t(a); e.to = uint64_t(b);
        e.Z = noisy(truth[size_t(a)].inverse() * truth[size_t(b)]); e.information = Mat6::Identity() * 100.0; G.AddEdge(e);
    };
    for (int i = 0; i + 1 < n; ++i) add(i, i + 1);
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int k = 0; k < loops; ++k) { const int a = pick(g), b = pick(g); if (a != b) add(a, b); }
    for (const auto& [id, nd] : std::map<uint64_t, PoseNode>(G.nodes())) if (id) G.UpdateNode(id, RandomPose(g, 0.02, 0.1) * nd.T_W_C, nd.covariance);
    return G;
}

struct Out {
    double gerr, tilt, E0, E1;
    int iters;
    bool found;
};
Out Solve(Chain c, const PgoSettings& s) {
    const PgoReport r = SolvePoseGraph(c.g, 0, s);
    return {Gerr(r.gravity_W.begin()->second), TiltErrorDeg(c), r.initial_energy, r.final_energy, r.iterations, r.found};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "trace";
    if (const char* o = std::getenv("ORDER")) kOrder = std::atoi(o);
    if (const char* o = std::getenv("GENSCALE")) kGenScale = std::atof(o);
    const int N = argc > 2 ? std::atoi(argv[2]) : 50;
    std::mt19937 gen(3);
    const Chain c = Make(gen, 200, 10, 2.3, 1.5);
    std::printf("input digest %016" PRIx64 " truth digest %016" PRIx64 " tilt0 %.17g\n", Digest(c.g), DigestTruth(c), TiltErrorDeg(c));
    if (mode == "trace") {
        PgoProblem p(c.g, 0, On().gravity);
        std::printf("E0 %.17g (%a)\n", p.Energy(p.poses()), p.Energy(p.poses()));
        const cv::Ptr<TraceBackend> b(new TraceBackend(p, true));
        cv::detail::LevMarqBase lm(b, ToOpenCv(On()));
        const cv::LevMarq::Report r = lm.optimize();
        std::printf("found %d iters %d energy %.17g gerr %.6f deg\n", r.found, r.iters, r.energy, Gerr(b->gravity()[0]));
    } else if (mode == "tight") {
        for (const auto& [name, s] : {std::pair{"default", On()}, std::pair{"tight", Tight()}}) {
            const Out o = Solve(c, s);
            std::printf("%-8s found %d iters %4d E %.12g -> %.12g gerr %.4f tilt %.4f\n", name, o.found, o.iters, o.E0, o.E1,
                        o.gerr, o.tilt);
        }
    } else if (mode == "ulp") {
        std::mt19937 pg(42);
        for (int k = 0; k < N; ++k) {
            Chain d = c;
            for (const auto& [id, n] : c.g.nodes()) {
                if (id == 0) continue;
                SE3 T = n.T_W_C;
                for (int a = 0; a < 3; ++a) T.t(a) = std::nextafter(T.t(a), (pg() & 1) ? 1e300 : -1e300);
                d.g.UpdateNode(id, T, n.covariance);
            }
            const Out o = Solve(d, On()), t = Solve(d, Tight());
            std::printf("ulp %2d default gerr %.4f (iters %3d found %d E %.10g) tight gerr %.4f (E %.10g)\n", k, o.gerr, o.iters,
                        o.found, o.E1, t.gerr, t.E1);
        }
    } else if (mode == "noisy") {   // the PgoLevMarq test's settings, through LevMarqBase on the in-TU backend
        std::mt19937 g{3u};
        PoseGraph G = LoopGraph(g, 50, 40, 0.01);
        std::printf("loop graph digest %016" PRIx64 "\n", Digest(G));
        PgoSettings s;
        s.rel_energy_delta_tolerance = 1e-14; s.step_norm_tolerance = 1e-14; s.min_gradient_tolerance = 1e-10;
        PgoProblem p(G, 0);
        const cv::Ptr<TraceBackend> b(new TraceBackend(p, false));
        cv::detail::LevMarqBase lm(b, ToOpenCv(s));
        const cv::LevMarq::Report r = lm.optimize();
        std::printf("noisy: found %d iters %d energy %.17g (E0 %.4f)\n", r.found, r.iters, r.energy, p.Energy(p.poses()));
    } else if (mode == "nees") {   // D124: NEES of the gravity estimate against the solver's own marginal covariance
        double sum = 0; int pass_tilt = 0;
        for (int sd = 0; sd < N; ++sd) {
            std::mt19937 g2{uint32_t(sd)};
            Chain e = Make(g2, 200, 10, 2.3, 1.5);
            const double t0 = TiltErrorDeg(e);
            const PgoReport r = SolvePoseGraph(e.g, 0, On());
            const Vec3 gh = r.gravity_W.begin()->second;
            PgoProblem p(e.g, 0, On().gravity);
            p.gravity()[0] = gh;                       // linearise at the optimum, gravity included
            SparseMat H; Eigen::VectorXd gr;
            p.Linearize(p.poses(), p.gravity(), H, gr);
            GuardedLdlt l;
            if (!l.Factorize(H)) { std::printf("seed %d: H not PD\n", sd); continue; }
            Eigen::Matrix2d C;
            for (int c2 = 0; c2 < 2; ++c2) {
                Eigen::VectorXd u = Eigen::VectorXd::Zero(p.dim()); u(p.pose_dim() + c2) = 1.0;
                C.col(c2) = l.Solve(u).tail<2>();
            }
            // H is the Hessian of E = 1/2 sum r^T W r, so the covariance is H^-1 (no factor 2).
            const Eigen::Vector2d d = GravityTangentBasis(gh).transpose() * kDown;   // true minus estimate, tangent at gh
            const double nees = d.dot(C.ldlt().solve(d));
            sum += nees; pass_tilt += TiltErrorDeg(e) < t0;
            if (sd < 5) std::printf("seed %d gerr %.3f deg sigma %.3f/%.3f deg nees %.3f\n", sd, Gerr(gh),
                                    std::sqrt(C(0, 0)) * 180 / M_PI, std::sqrt(C(1, 1)) * 180 / M_PI, nees);
        }
        std::printf("mean NEES %.3f over %d seeds (chi2 99%% band for N=50: [1.347, 2.803]); tilt reduced %d/%d\n", sum / N, N, pass_tilt, N);
    } else if (mode == "halcheck") {
        HalCheck();
    } else if (mode == "pure") {
        PgoProblem p(c.g, 0, On().gravity);
        const PureOut o = PureLm(p, On(), true);
        std::printf("pure found %d iters %d energy %.17g gerr %.6f deg\n", o.found, o.iters, o.energy, o.gerr);
    } else if (mode == "perturb") {   // every node pose (R via a tiny left rotation, t) and every prior g_C, by eps
        const double eps = argc > 3 ? std::atof(argv[3]) : 1e-15;
        std::mt19937 pg(7);
        std::normal_distribution<double> nz(0.0, 1.0);
        for (int k = 0; k < N; ++k) {
            Chain d;
            d.truth = c.truth;
            for (const auto& [id, n] : c.g.nodes()) {
                PoseNode m = n;
                if (id != 0) {
                    m.T_W_C.R = Rot(eps * Vec3(nz(pg), nz(pg), nz(pg))) * m.T_W_C.R;
                    m.T_W_C.t += eps * Vec3(nz(pg), nz(pg), nz(pg));
                }
                d.g.AddNode(m);
            }
            for (const auto& e : c.g.edges()) d.g.AddEdge(e);
            for (GravityPrior p : c.g.gravity_priors()) {
                p.g_C = (Rot(eps * Vec3(nz(pg), nz(pg), 0)) * p.g_C).normalized();
                d.g.AddGravityPrior(p);
            }
            const Out o = Solve(d, On());
            std::printf("perturb %.0e #%2d gerr %.4f tilt %.4f iters %3d found %d\n", eps, k, o.gerr, o.tilt, o.iters, o.found);
        }
    } else if (mode == "seeds") {
        int pass_d = 0, pass_t = 0;
        for (int sd = 0; sd < N; ++sd) {
            std::mt19937 g2{uint32_t(sd)};
            const int every = std::getenv("EVERY") ? std::atoi(std::getenv("EVERY")) : 10;
            const double sig = std::getenv("SIGMA") ? std::atof(std::getenv("SIGMA")) : 2.3;
            const Chain e = Make(g2, 200, every, sig, 1.5);   // EVERY / SIGMA: fixture conditioning (default = the test's)
            const double t0 = TiltErrorDeg(e);
            const Out o = Solve(e, On()), t = Solve(e, Tight());
            pass_d += o.gerr < 1.0 && o.tilt < t0;
            pass_t += t.gerr < 1.0 && t.tilt < t0;
            std::printf("seed %3d default gerr %.4f iters %3d | tight gerr %.4f iters %4d | E %.8g vs %.8g\n", sd, o.gerr, o.iters,
                        t.gerr, t.iters, o.E1, t.E1);
        }
        std::printf("pass default %d/%d tight %d/%d\n", pass_d, N, pass_t, N);
    }
    return 0;
}
