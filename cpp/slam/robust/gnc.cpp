// GNC-TLS (arch §14.6, D73 part 1): see gnc.hpp for what and why.
#include "slam/robust/gnc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ariadne::slam {

namespace {

/// chi^2 survival for an EVEN number of degrees of freedom k = 2m: e^-h sum_{i<m} h^i / i!, h = x / 2.
double ChiSquareEvenSurvival(double x, int k) {
    if (x <= 0) return 1.0;
    const double h = 0.5 * x;
    double term = 1.0, sum = 1.0;
    for (int i = 1; i < k / 2; ++i) sum += (term *= h / i);
    return std::exp(-h) * sum;
}

double ChiSquareEvenQuantile(double p, int k) {
    double lo = 0.0, hi = 1.0;
    while (1.0 - ChiSquareEvenSurvival(hi, k) < p) hi *= 2.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        (1.0 - ChiSquareEvenSurvival(mid, k) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

}  // namespace

double TlsWeight(double r2, double mu, double cbar2) {
    if (r2 <= mu / (mu + 1.0) * cbar2) return 1.0;
    if (r2 >= (mu + 1.0) / mu * cbar2) return 0.0;
    return std::sqrt(cbar2 / r2) * std::sqrt(mu * (mu + 1.0)) - mu;
}

GncReport SolveGnc(PoseGraph& g, uint64_t anchor, const std::vector<bool>* active, const GncParams& p,
                   GncState* state) {
    GncReport rep;
    std::vector<bool> on;
    for (size_t i = 0; i < g.edges().size(); ++i) {
        if (!IsLoopType(g.edges()[i].type)) continue;
        rep.loops.push_back(i);
        on.push_back(!active || (i < active->size() && (*active)[i]));
    }
    const size_t L = rep.loops.size();
    PgoSettings lm = p.lm;
    lm.recompute_covariance = false;
    auto r2_of = [&](const PoseGraph& pg) {
        std::vector<double> r2(L);
        for (size_t k = 0; k < L; ++k) {
            const PoseEdge& e = pg.edges()[rep.loops[k]];
            const Vec6 r = pg.Residual(e);
            r2[k] = r.dot(e.information * r);
        }
        return r2;
    };
    auto finish = [&](PoseGraph& w, const std::vector<double>& weight, int components, double energy) {
        rep.w_final = weight;
        rep.r2_final = r2_of(w);
        rep.components = components;
        rep.final_energy = energy;
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
        if (state) {
            state->decision.assign(g.edges().size(), int8_t(-1));
            for (size_t k = 0; k < L; ++k)
                if (on[k]) state->decision[rep.loops[k]] = int8_t(weight[k] > p.inlier_weight ? 1 : 0);
        }
        return rep;
    };

    // Weight reuse: the previous decisions as one solve; accepted only if the split is a TLS fixed point.
    rep.warm = state && !state->decision.empty();
    if (rep.warm) {
        PoseGraph f = g;
        std::vector<double> weight(L, 0.0);
        std::vector<bool> fresh(L, false);
        for (size_t k = 0; k < L; ++k) {
            if (!on[k]) continue;
            const size_t e = rep.loops[k];
            const bool known = e < state->decision.size() && state->decision[e] >= 0;
            weight[k] = known ? double(state->decision[e]) : 1.0;
            fresh[k] = !known;
            rep.new_loops += !known;
        }
        // Without the new loops (warm: the input is that problem's optimum, so this costs one or two iterations).
        double e_without = 0.0;
        if (rep.new_loops > 0) {
            for (size_t k = 0; k < L; ++k) f.SetGncWeight(rep.loops[k], fresh[k] ? 0.0 : weight[k]);
            const PgoReport r0 = SolvePoseGraph(f, anchor, lm);
            rep.lm_iterations.push_back(r0.iterations);
            e_without = r0.final_energy;
        }
        for (size_t k = 0; k < L; ++k) f.SetGncWeight(rep.loops[k], weight[k]);
        const PgoReport fr = SolvePoseGraph(f, anchor, lm);
        rep.lm_iterations.push_back(fr.iterations);
        if (rep.new_loops > 0) {
            rep.innovation = 2.0 * (fr.final_energy - e_without);   // the energy carries 1/2
            // Half the chi^2 bound: Remark 5's all-inlier region applied to the innovation, so the fast path takes
            // only what GNC's convex start would already accept; the ambiguous band goes to cold GNC (design §6).
            rep.innovation_bound = 0.5 * ChiSquareEvenQuantile(0.99, 6 * rep.new_loops);
            rep.fallback_innovation = rep.innovation > rep.innovation_bound;
        }
        const std::vector<double> r2 = r2_of(f);
        rep.r2_initial = r2;
        for (size_t k = 0; k < L; ++k) {
            if (!on[k]) continue;
            rep.fallback_held_in_over += weight[k] == 1.0 && r2[k] > p.cbar2;
            rep.fallback_held_out_under += weight[k] == 0.0 && r2[k] <= p.cbar2;
        }
        if (rep.fallback_held_in_over == 0 && rep.fallback_held_out_under == 0 && !rep.fallback_innovation) {
            rep.fast_path = true;
            rep.stop = "fast_path";
            return finish(f, weight, fr.components, fr.final_energy);
        }
        rep.lm_iterations.clear();   // the fast-path solve is discarded with its copy
    }

    PoseGraph w = g;   // working copy: weights change every outer iteration, the input graph only at the end
    for (size_t k = 0; k < L; ++k) w.SetGncWeight(rep.loops[k], on[k] ? 1.0 : 0.0);
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

    return finish(w, weight, last.components, last.final_energy);
}

}  // namespace ariadne::slam
