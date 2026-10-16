#include "slam/pose_graph/pgo_levmarq.hpp"

#include <cmath>
#include <limits>

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

class PgoBackend : public cv::detail::LevMarqBackend {
public:
    PgoBackend(PgoProblem& p, PgoReport& rep) : p_(p), rep_(rep), current_(p.poses()), probe_(p.poses()) {}

    bool enableGeo() override { return false; }

    bool calcFunc(double& energy, bool calcEnergy, bool calcJacobian) override {
        if (calcJacobian) {
            energy = p_.Linearize(probe_, H_, g_);
            ++rep_.linearizations;
        } else if (calcEnergy) {
            energy = p_.Energy(probe_);
        }
        return std::isfinite(energy);
    }

    void currentOplusX(const cv::Mat_<double>& x, bool geo) override {
        if (geo) CV_Error(cv::Error::StsBadArg, "PgoBackend: geodesic acceleration is not implemented");
        probe_ = p_.OplusX(current_, ToVec(x));
    }

    void prepareVars() override { probe_ = current_; }

    const cv::Mat_<double> getJtb() override { return ToMat(g_); }

    const cv::Mat_<double> getDiag() override { return ToMat(H_.diagonal()); }

    void setDiag(const cv::Mat_<double>& d) override {
        for (int i = 0; i < d.rows; ++i) H_.coeffRef(i, i) = d(i);   // diagonal entries always exist (see Linearize)
    }

    void doJacobiScaling(const cv::Mat_<double>& di) override {
        // H <- D H D, g <- D g with D = diag(di), as OpenCV's own backends do.
        const Eigen::VectorXd d = ToVec(di);
        for (int k = 0; k < H_.outerSize(); ++k)
            for (SparseMat::InnerIterator it(H_, k); it; ++it) it.valueRef() *= d(it.row()) * d(it.col());
        g_ = g_.cwiseProduct(d);
    }

    bool decompose() override {
        ++rep_.factorizations;
        const bool ok = ldlt_.Factorize(H_);
        rep_.failed_factorizations += !ok;
        return ok;
    }

    bool solveDecomposed(const cv::Mat_<double>& right, cv::Mat_<double>& x) override {
        if (!ldlt_.ok()) return false;
        const Eigen::VectorXd sol = ldlt_.Solve(-ToVec(right));
        if (!sol.allFinite()) return false;
        x = ToMat(sol);
        return true;
    }

    bool calcJtbv(cv::Mat_<double>&) override { return false; }

    void acceptProbe() override { current_ = probe_; }

    const std::map<uint64_t, SE3>& current() const { return current_; }

private:
    PgoProblem& p_;
    PgoReport& rep_;
    std::map<uint64_t, SE3> current_, probe_;
    SparseMat H_;
    Eigen::VectorXd g_;
    GuardedLdlt ldlt_;
};

}  // namespace

cv::LevMarq::Settings ToOpenCv(const PgoSettings& s) {
    return cv::LevMarq::Settings()
        .setJacobiScaling(s.jacobi_scaling)
        .setUpDouble(s.up_double)
        .setUseStepQuality(s.use_step_quality)
        .setClampDiagonal(s.clamp_diagonal)
        .setStepNormInf(s.step_norm_inf)
        .setCheckRelEnergyChange(s.check_rel_energy_change)
        .setCheckMinGradient(s.check_min_gradient)
        .setCheckStepNorm(s.check_step_norm)
        .setGeodesic(false)
        .setStepNormTolerance(s.step_norm_tolerance)
        .setRelEnergyDeltaTolerance(s.rel_energy_delta_tolerance)
        .setMinGradientTolerance(s.min_gradient_tolerance)
        .setSmallEnergyTolerance(s.small_energy_tolerance)
        .setMaxIterations(s.max_iterations)
        .setInitialLambda(s.initial_lambda)
        .setInitialLmUpFactor(s.initial_lm_up_factor)
        .setInitialLmDownFactor(s.initial_lm_down_factor);
}

PgoReport SolvePoseGraph(PoseGraph& g, uint64_t anchor, const PgoSettings& s) {
    PgoProblem p(g, anchor);
    PgoReport rep;
    rep.initial_energy = p.Energy(p.poses());
    // cv::makePtr takes its arguments by const reference; the backend needs the problem mutably.
    const cv::Ptr<PgoBackend> backend(new PgoBackend(p, rep));
    cv::detail::LevMarqBase lm(backend, ToOpenCv(s));
    const cv::LevMarq::Report r = lm.optimize();
    rep.found = r.found;
    rep.iterations = r.iters;
    p.poses() = backend->current();
    rep.final_energy = p.Energy(p.poses());
    std::map<uint64_t, Mat6> cov;
    if (s.recompute_covariance) {
        std::vector<uint64_t> ids;
        for (const auto& [id, T] : p.poses()) ids.push_back(id);
        cov = MarginalCovariances(p, ids);
    }
    const Mat6 unknown = Mat6::Constant(std::numeric_limits<double>::quiet_NaN());
    for (const auto& [id, T] : p.poses()) g.UpdateNode(id, T, s.recompute_covariance ? cov.at(id) : unknown);
    return rep;
}

}  // namespace ariadne::slam
