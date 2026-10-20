// GNC-TLS (arch §14.6, D73 part 1): see gnc.hpp for what and why.
#include "slam/robust/gnc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ariadne::slam {

double TlsWeight(double r2, double mu, double cbar2) {
    if (r2 <= mu / (mu + 1.0) * cbar2) return 1.0;
    if (r2 >= (mu + 1.0) / mu * cbar2) return 0.0;
    return std::sqrt(cbar2 / r2) * std::sqrt(mu * (mu + 1.0)) - mu;
}

GncReport SolveGnc(PoseGraph& g, uint64_t anchor, const std::vector<bool>* active, const GncParams& p) {
    GncReport rep;
    PoseGraph w = g;   // working copy: weights change every outer iteration, the input graph only at the end
    std::vector<bool> on;
    for (size_t i = 0; i < w.edges().size(); ++i) {
        if (!IsLoopType(w.edges()[i].type)) continue;
        rep.loops.push_back(i);
        on.push_back(!active || (i < active->size() && (*active)[i]));
        w.SetGncWeight(i, on.back() ? 1.0 : 0.0);
    }
    const size_t L = rep.loops.size();
    PgoSettings lm = p.lm;
    lm.recompute_covariance = false;
    auto r2_all = [&]() {
        std::vector<double> r2(L);
        for (size_t k = 0; k < L; ++k) {
            const PoseEdge& e = w.edges()[rep.loops[k]];
            const Vec6 r = w.Residual(e);
            r2[k] = r.dot(e.information * r);
        }
        return r2;
    };
    std::vector<double> weight(L);
    for (size_t k = 0; k < L; ++k) weight[k] = on[k] ? 1.0 : 0.0;

    PgoReport last = SolvePoseGraph(w, anchor, lm);   // every allowed loop at w = 1
    rep.lm_iterations.push_back(last.iterations);
    std::vector<double> r2 = r2_all();
    rep.r2_initial = r2;
    double r2max = -1.0;
    for (size_t k = 0; k < L; ++k)
        if (on[k]) r2max = std::max(r2max, r2[k]);

    if (r2max < 0.0) {
        rep.stop = "no_loops";
    } else if (2.0 * r2max <= p.cbar2) {
        rep.stop = "all_inliers";   // Remark 5: the surrogate is already TLS-optimal
    } else {
        double mu = p.cbar2 / (2.0 * r2max - p.cbar2);
        double prev_cost = std::numeric_limits<double>::infinity();
        for (int it = 0; it < p.max_outer; ++it) {
            for (size_t k = 0; k < L; ++k)
                if (on[k]) {
                    weight[k] = TlsWeight(r2[k], mu, p.cbar2);
                    w.SetGncWeight(rep.loops[k], weight[k]);
                }
            rep.mu.push_back(mu);
            rep.w_history.push_back(weight);
            last = SolvePoseGraph(w, anchor, lm);
            rep.lm_iterations.push_back(last.iterations);
            r2 = r2_all();
            double cost = 0.0;
            bool binary = true;
            for (size_t k = 0; k < L; ++k) {
                if (!on[k]) continue;
                cost += weight[k] * r2[k];
                binary = binary && (weight[k] < p.binary_tol || weight[k] > 1.0 - p.binary_tol);
            }
            if (binary) {
                rep.stop = "weights";
                break;
            }
            if (std::fabs(cost - prev_cost) / std::max(prev_cost, 1e-7) < p.rel_cost_tol) {
                rep.stop = "cost";
                break;
            }
            prev_cost = cost;
            mu *= p.mu_step;
        }
        if (rep.stop.empty()) rep.stop = "max_outer";
    }

    rep.w_final = weight;
    rep.r2_final = r2_all();
    rep.components = last.components;
    rep.final_energy = last.final_energy;
    for (size_t k = 0; k < L; ++k) {
        rep.non_binary += on[k] && weight[k] >= p.binary_tol && weight[k] <= 1.0 - p.binary_tol;
        (on[k] && weight[k] > p.inlier_weight ? rep.inliers : rep.outliers).push_back(rep.loops[k]);
    }
    if (p.marginals_at_end) {   // one pass on the final weights (it starts at the optimum: a few iterations)
        PgoSettings m = p.lm;
        m.recompute_covariance = true;
        SolvePoseGraph(w, anchor, m);
    }
    for (size_t k = 0; k < L; ++k) g.SetGncWeight(rep.loops[k], weight[k]);
    const Mat6 unknown = Mat6::Constant(std::numeric_limits<double>::quiet_NaN());
    for (const auto& [id, n] : w.nodes()) g.UpdateNode(id, n.T_W_C, p.marginals_at_end ? n.covariance : unknown);
    return rep;
}

}  // namespace ariadne::slam
