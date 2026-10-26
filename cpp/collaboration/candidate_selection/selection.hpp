#pragma once
// ---------------------------------------------------------------------------
// Candidate selection types (design spectral-selection.md, D170): the problem the broker hands a selection policy, the report
// it gets back, and the "greedy" baseline (highest similarity per marginal byte). No OpenCV or Eigen here: the
// protocol (collab_node.hpp) builds without OpenCV, and the spectral policy (spectral_selector.hpp) plugs in through
// NodeHooks::select.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

namespace ariadne::collaboration {

struct GraphEdge {
    int a = 0, b = 0;
    double kappa = 1.0;
};

struct CandidateEdge {
    uint64_t id = 0;        // stable id: the tie-break, so the selection is the same on every host
    int a = 0, b = 0;       // node indices
    double kappa = 1.0;     // retrieval similarity in (0, 1]
};

struct SelectionProblem {
    int n = 0;                          // nodes 0..n-1
    std::vector<GraphEdge> fixed;
    std::vector<CandidateEdge> candidates;
    std::vector<double> node_bytes;     // predicted bytes to fetch a node's features; 0 = held (broker's own, fetched)
    double budget_bytes = 0;
};

struct SelectionReport {
    std::vector<uint64_t> selected;     // in selection order
    double bytes = 0;
    int bootstrap = 0;                  // picks that joined components
    bool connected = false;
    double lambda2 = 0;                 // of the final graph (0 if disconnected)
    std::optional<double> bound;        // >= the optimum's lambda_2; absent when not computable (stated)
    int solves = 0, iterations = 0;     // Fiedler solves and their total iterations
    int swaps = 0;                      // improving swaps applied by the refinement
};

/// The "greedy" baseline of the gate plot: highest similarity per marginal byte, same budget, no spectrum.
inline SelectionReport SelectBySimilarity(const SelectionProblem& P) {
    SelectionReport rep;
    std::vector<char> held(size_t(P.n), 0);
    for (int i = 0; i < P.n; ++i) held[size_t(i)] = P.node_bytes[size_t(i)] <= 0.0;
    std::vector<size_t> order(P.candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        const auto& a = P.candidates[x];
        const auto& b = P.candidates[y];
        return a.kappa != b.kappa ? a.kappa > b.kappa : a.id < b.id;
    });
    for (size_t k : order) {
        const CandidateEdge& e = P.candidates[k];
        double c = (held[size_t(e.a)] ? 0.0 : P.node_bytes[size_t(e.a)]) +
                   (e.b == e.a || held[size_t(e.b)] ? 0.0 : P.node_bytes[size_t(e.b)]);
        if (rep.bytes + c > P.budget_bytes) continue;
        rep.bytes += c;
        held[size_t(e.a)] = held[size_t(e.b)] = 1;
        rep.selected.push_back(e.id);
    }
    return rep;
}

}  // namespace ariadne::collaboration
