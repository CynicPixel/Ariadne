#include "slam/bundle_adjustment/window_ba_levmarq.hpp"

#include <cmath>
#include <set>
#include <sstream>
#include <string>

namespace ariadne::slam {

namespace {

cv::Mat_<double> ToMat(const Eigen::VectorXd& v) {
    cv::Mat_<double> m(int(v.size()), 1);
    for (Eigen::Index i = 0; i < v.size(); ++i) m(int(i)) = v(i);
    return m;
}

Eigen::VectorXd ToVec(const cv::Mat_<double>& m) {
    Eigen::VectorXd v(m.rows);
    for (int i = 0; i < m.rows; ++i) v(i) = m(i);
    return v;
}

class BaBackend : public cv::detail::LevMarqBackend {
public:
    BaBackend(const BaWindow& w, WindowBaReport& rep) : L_(w), rep_(rep), current_(w), probe_(w) {}

    bool enableGeo() override { return false; }
    bool calcFunc(double& energy, bool calcEnergy, bool calcJacobian) override {
        if (calcJacobian) energy = BaLinearize(probe_, L_, sys_);
        else if (calcEnergy) energy = BaEnergy(probe_);
        return std::isfinite(energy);
    }
    void currentOplusX(const cv::Mat_<double>& x, bool geo) override {
        if (geo) CV_Error(cv::Error::StsBadArg, "BaBackend: geodesic acceleration is not implemented");
        probe_ = BaOplus(current_, L_, ToVec(x));
    }
    void prepareVars() override { probe_ = current_; }
    const cv::Mat_<double> getJtb() override { return ToMat(sys_.g); }
    const cv::Mat_<double> getDiag() override { return ToMat(BaDiagonal(sys_, L_)); }
    void setDiag(const cv::Mat_<double>& d) override { BaSetDiagonal(sys_, L_, ToVec(d)); }
    void doJacobiScaling(const cv::Mat_<double>&) override {
        CV_Error(cv::Error::StsNotImplemented, "BaBackend: Jacobi scaling is not implemented (set jacobi_scaling false)");
    }
    bool decompose() override {
        ++rep_.factorizations;
        const bool ok = schur_.Factorize(sys_, L_);
        rep_.failed_factorizations += !ok;
        return ok;
    }
    bool solveDecomposed(const cv::Mat_<double>& right, cv::Mat_<double>& x) override {
        const Eigen::VectorXd sol = schur_.Solve(sys_, L_, ToVec(right));
        if (!sol.allFinite()) return false;
        x = ToMat(sol);
        return true;
    }
    bool calcJtbv(cv::Mat_<double>&) override { return false; }
    void acceptProbe() override { current_ = probe_; }

    const BaWindow& current() const { return current_; }
    const BaLayout& layout() const { return L_; }

private:
    BaLayout L_;
    WindowBaReport& rep_;
    BaWindow current_, probe_;
    BaSystem sys_;
    BaSchur schur_;
};

int InvalidObservations(const BaWindow& w) {
    int n = 0;
    for (const auto& o : w.obs)
        n += !LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], w.X.at(o.landmark), o, w.z_floor).valid;
    return n;
}

/// Loud-failure text: which block of the undamped system is singular, and what the LM run looked like.
std::string NotPdDiagnostic(const BaWindow& w, const BaLayout& L, const BaSchur& schur, const WindowBaReport& rep) {
    std::ostringstream m;
    m << "SolveWindowBa: information not PD at the solution (found " << rep.found << ", " << rep.iterations
      << " iterations, " << rep.failed_factorizations << "/" << rep.factorizations << " factorizations failed, E "
      << rep.initial_energy << " -> " << rep.final_energy << ", " << InvalidObservations(w) << "/" << w.obs.size()
      << " observations invalid at the solution; ";
    if (schur.failed_landmark() < 0) {
        // Per window keyframe: observed landmarks, and how many of them another window keyframe also observes (only
        // those constrain the pose; a landmark seen by one keyframe moves with it).
        std::map<uint64_t, std::set<int>> kfs_of;
        for (const auto& o : w.obs) kfs_of[o.landmark].insert(o.kf);
        m << "reduced camera system; keyframe: observed/shared landmarks, S block min eigenvalue:";
        for (int k = 0; k < int(w.T_W_C.size()); ++k) {
            int seen = 0, shared = 0;
            for (const auto& [id, ks] : kfs_of)
                if (ks.count(k)) {
                    ++seen;
                    shared += ks.size() > 1;
                }
            m << " " << k << (w.fixed[size_t(k)] ? "F" : "") << ":" << seen << "/" << shared;
            if (L.cam(k) >= 0) {
                const Eigen::SelfAdjointEigenSolver<Mat6> es(schur.S().block<6, 6>(6 * L.cam(k), 6 * L.cam(k)));
                m << "," << es.eigenvalues()(0);
            }
        }
        // The null direction of the reduced camera system: which free keyframes move together, and how.
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(schur.S());
        m << "; S min eigenvalue " << es.eigenvalues()(0) << " (max " << es.eigenvalues()(es.eigenvalues().size() - 1)
          << "), its eigenvector per free keyframe [rho; phi]:";
        for (int k = 0; k < int(w.T_W_C.size()); ++k)
            if (L.cam(k) >= 0) m << " " << k << ":(" << es.eigenvectors().col(0).segment<6>(6 * L.cam(k)).transpose() << ")";
        m << ")";
        return m.str();
    }
    for (const auto& [id, X] : w.X) {
        if (L.lm(id) != schur.failed_landmark()) continue;
        int n = 0, valid = 0;
        for (const auto& o : w.obs) {
            if (o.landmark != id) continue;
            ++n;
            valid += LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], X, o, w.z_floor).valid;
        }
        m << "landmark " << id << " at (" << X.transpose() << "): " << valid << "/" << n << " observations valid)";
    }
    return m.str();
}

}  // namespace

bool TryWindowBaCovariances(const BaWindow& w, const std::vector<int>& keyframes, std::map<int, Mat6>& cov,
                            std::string* why, const WindowBaReport* solve) {
    // The undamped system at w (the solution, when called after SolveWindowBa).
    const BaLayout L(w);
    BaSystem sys;
    BaLinearize(w, L, sys);
    BaSchur schur;
    cov.clear();
    if (!schur.Factorize(sys, L)) {
        if (why) *why = NotPdDiagnostic(w, L, schur, solve ? *solve : WindowBaReport{});
        return false;
    }
    for (int k : keyframes) {
        const int c = L.cam(k);
        if (c < 0) throw std::invalid_argument("WindowBaCovariances: covariance requested for a fixed keyframe");
        cov[k] = schur.CameraCovariance(c);
    }
    return true;
}

std::map<int, Mat6> WindowBaCovariances(const BaWindow& w, const std::vector<int>& keyframes,
                                        const WindowBaReport* solve) {
    std::map<int, Mat6> cov;
    std::string why;
    if (!TryWindowBaCovariances(w, keyframes, cov, &why, solve)) throw std::runtime_error(why);
    return cov;
}

WindowBaReport SolveWindowBa(BaWindow& w, const PgoSettings& s, const std::vector<int>& covariance_for) {
    if (s.jacobi_scaling) throw std::invalid_argument("SolveWindowBa: jacobi_scaling is not supported");
    WindowBaReport rep;
    rep.initial_energy = BaEnergy(w);
    const cv::Ptr<BaBackend> backend(new BaBackend(w, rep));
    cv::detail::LevMarqBase lm(backend, ToOpenCv(s));
    const cv::LevMarq::Report r = lm.optimize();
    rep.found = r.found;
    rep.iterations = r.iters;
    w = backend->current();
    rep.final_energy = BaEnergy(w);
    if (!covariance_for.empty()) rep.covariance = WindowBaCovariances(w, covariance_for, &rep);
    return rep;
}

}  // namespace ariadne::slam
