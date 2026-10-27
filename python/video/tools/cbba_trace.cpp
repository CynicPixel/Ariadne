// Runs the project's CBBA (cpp/planning/mission/allocation.cpp, unchanged) on an instance read from stdin and prints
// every round's broadcast snapshots (y, z per agent) plus the converged allocation as JSON, for the beat 7 animation.
//
//   g++ -std=c++20 -O2 -I cpp python/video/tools/cbba_trace.cpp cpp/planning/mission/allocation.cpp -o cbba_trace
//
// stdin: U N L, then U rates, then U x N values, then U x N distances (agent -> task goal), then (L > 1 only)
// U x N x N goal -> goal legs.
#include "planning/mission/allocation.hpp"

#include <cstdio>
#include <iostream>
#include <vector>

using namespace ariadne::planning;

int main() {
    AuctionInstance in;
    int N = 0;
    std::cin >> in.n_agents >> N >> in.max_bundle;
    const int U = in.n_agents;
    for (int j = 0; j < N; ++j) in.task_ids.push_back(j);
    in.rate.resize(size_t(U));
    for (auto& r : in.rate) std::cin >> r;
    in.value.assign(size_t(U), std::vector<double>(size_t(N)));
    in.d0 = in.value;
    for (auto& row : in.value)
        for (auto& v : row) std::cin >> v;
    for (auto& row : in.d0)
        for (auto& v : row) std::cin >> v;
    in.leg.assign(size_t(U), std::vector<std::vector<double>>(size_t(N), std::vector<double>(size_t(N), 0.0)));
    if (in.max_bundle > 1)
        for (auto& a : in.leg)
            for (auto& row : a)
                for (auto& v : row) std::cin >> v;

    std::vector<std::vector<CbbaMessage>> rounds;   // the snapshots each agent broadcasts, per round (after its build)
    CbbaParams prm;
    prm.transport = [&](const CbbaMessage& m) {
        if (m.sender == 0) rounds.emplace_back();
        rounds.back().push_back(m);
        return m;
    };
    prm.bytes = TestCodecBytes;
    const auto full = [U](int) { return std::vector<std::vector<int>>(size_t(U), std::vector<int>(size_t(U), 1)); };
    const Allocation a = RunCbba(in, full, prm);

    std::printf("{\"rounds\":[");
    for (size_t t = 0; t < rounds.size(); ++t) {
        std::printf("%s[", t ? "," : "");
        for (size_t k = 0; k < rounds[t].size(); ++k) {
            const auto& m = rounds[t][k];
            std::printf("%s{\"y\":[", k ? "," : "");
            for (int j = 0; j < N; ++j) std::printf("%s%.9g", j ? "," : "", m.y[size_t(j)]);
            std::printf("],\"z\":[");
            for (int j = 0; j < N; ++j) std::printf("%s%d", j ? "," : "", m.z[size_t(j)]);
            std::printf("]}");
        }
        std::printf("]");
    }
    std::printf("],\"paths\":[");   // the converged visiting order per agent (each task at its best insertion)
    for (size_t i = 0; i < a.agents.size(); ++i) {
        std::printf("%s[", i ? "," : "");
        for (size_t n = 0; n < a.agents[i].path.size(); ++n) std::printf("%s%d", n ? "," : "", a.agents[i].path[n]);
        std::printf("]");
    }
    std::printf("],\"winner\":[");
    for (int j = 0; j < N; ++j) std::printf("%s%d", j ? "," : "", a.winner[size_t(j)]);
    std::printf("],\"total\":%.9g,\"optimum\":%.9g,\"sga_total\":%.9g,\"converged\":%d,\"n_rounds\":%d,"
                "\"messages\":%zu,\"bytes\":%zu}\n",
                a.total, BruteForceOptimum(in), SequentialGreedy(in).total, int(a.converged), a.rounds, a.messages,
                a.bytes);
}
