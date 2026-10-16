#include "slam/bundle_adjustment/window_ba_levmarq.hpp"

#include <cmath>

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

}  // namespace

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
    if (!covariance_for.empty()) {
        // Undamped system at the solution.
        const BaLayout& L = backend->layout();
        BaSystem sys;
        BaLinearize(w, L, sys);
        BaSchur schur;
        if (!schur.Factorize(sys, L)) throw std::runtime_error("SolveWindowBa: information not PD at the solution");
        for (int k : covariance_for) {
            const int c = L.cam(k);
            if (c < 0) throw std::invalid_argument("SolveWindowBa: covariance requested for a fixed keyframe");
            rep.covariance[k] = schur.CameraCovariance(c);
        }
    }
    return rep;
}

}  // namespace ariadne::slam
