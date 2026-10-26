// Runs the project's D* Lite (cpp/planning/route/dstar_lite.cpp, unchanged) for the beat 9 animation: the first
// search on a traversable grid, then Update() with an obstruction, and a fresh search on the obstructed grid for
// comparison. Writes the g fields (cost-to-exit, cells) and a JSON summary.
//
//   g++ -std=c++20 -O2 -I cpp python/video/tools/dstar_trace.cpp cpp/planning/route/dstar_lite.cpp \
//       -I /opt/opencv5/include/opencv5 -L /opt/opencv5/lib -lopencv_core -o dstar_trace   (in ariadne/core:dev)
//   dstar_trace DIR H W start_r start_c r0 c0 r1 c1 ...   (DIR holds trav0.u8, trav1.u8; writes g0.f64, g1.f64)
#include "planning/route/dstar_lite.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ariadne::planning;

static cv::Mat Load(const std::string& path, int H, int W) {
    cv::Mat m(H, W, CV_8U);
    std::ifstream(path, std::ios::binary).read(reinterpret_cast<char*>(m.data), std::streamsize(H) * W);
    return m;
}

static void SaveG(const DStarLite& d, const std::string& path, int H, int W) {
    std::vector<double> g(size_t(H) * size_t(W));
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) g[size_t(r) * size_t(W) + size_t(c)] = d.G({r, c});
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(g.data()), std::streamsize(g.size() * 8));
}

static void PrintPath(const std::vector<Cell>& p) {
    std::printf("[");
    for (size_t i = 0; i < p.size(); ++i) std::printf("%s[%d,%d]", i ? "," : "", p[i].r, p[i].c);
    std::printf("]");
}

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    const int H = std::atoi(argv[2]), W = std::atoi(argv[3]);
    const Cell start{std::atoi(argv[4]), std::atoi(argv[5])};
    std::vector<Cell> goals;
    for (int i = 6; i + 1 < argc; i += 2) goals.push_back({std::atoi(argv[i]), std::atoi(argv[i + 1])});
    const cv::Mat t0 = Load(dir + "/trav0.u8", H, W), t1 = Load(dir + "/trav1.u8", H, W);
    const cv::Mat pen = cv::Mat::zeros(H, W, CV_32F);

    DStarLite d(t0, pen, start, goals);
    const size_t e0 = d.Expanded();
    const double c0 = d.Cost();
    const auto p0 = d.Path();
    SaveG(d, dir + "/g0.f64", H, W);
    const auto changed = d.Update(t1, pen);
    const size_t e1 = d.Expanded() - e0;
    SaveG(d, dir + "/g1.f64", H, W);
    DStarLite fresh(t1, pen, start, goals);

    std::printf("{\"cost0\":%.9g,\"cost1\":%.9g,\"expanded_first\":%zu,\"expanded_repair\":%zu,"
                "\"expanded_fresh\":%zu,\"changed_cells\":%zu,\"fresh_cost\":%.9g,\"path0\":",
                c0, d.Cost(), e0, e1, fresh.Expanded(), changed.size(), fresh.Cost());
    PrintPath(p0);
    std::printf(",\"path1\":");
    PrintPath(d.Path());
    std::printf(",\"exit0\":%d,\"exit1\":%d}\n", d.GoalIndex(p0.back()), d.GoalIndex(d.Path().back()));
}
