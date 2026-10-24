// ---------------------------------------------------------------------------
// ariadne_entry_nees: the D178 calibration of the known-entry gate's odometry covariance (pre-registered: scene 1
// only, never gate data).
//
//   ariadne_entry_nees --scene SCENEDIR --out YAML STORE_DIR [STORE_DIR ...]
//
// For every agent store (pass A, store_agent<k>.bin) in the given directories: its keyframes' odometry poses T_Wk_C and
// the odometry edges' information, composed along the chain exactly as the gate does (ChainCovariances in
// collaboration/broker/broker_solve.hpp), against ground truth expressed in the agent's own frame through the scene's
// entry placement (EntryPoses: T_Wk_C_gt = T_World_Wk^-1 T_World_C_gt). Per keyframe after the first:
//   delta = Log(T_Wk_C_gt T_Wk_C^-1)   (left perturbation, the gate's convention),   NEES = delta^T Sigma^-1 delta.
// kappa = max(1, mean(NEES) / 6): the factor that makes the composed covariance consistent on average (Bar-Shalom's
// NEES test; 6 = the dimension). Writes kappa, the NEES mean/median/p90/p99 and per-store counts.
// GROUND TRUTH IS CALIBRATION INPUT here and nowhere in the gate's decisions.
// ---------------------------------------------------------------------------
#include "collaboration/broker/broker_solve.hpp"
#include "planning/mission/frontier_mission.hpp"
#include "runtime/agent/collab_store.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ariadne;
namespace ac = ariadne::collaboration;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string scene, out;
    std::vector<std::string> dirs;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--scene" && i + 1 < argc) scene = argv[++i];
        else if (k == "--out" && i + 1 < argc) out = argv[++i];
        else dirs.push_back(k);
    }
    if (scene.empty() || out.empty() || dirs.empty()) {
        std::fprintf(stderr, "usage: see the header of ariadne_entry_nees.cpp\n");
        return 2;
    }
    try {
        std::vector<double> nees;
        std::ofstream y(out);
        y << "tool: ariadne_entry_nees\nscene: " << scene << "\nstores:\n";
        for (const auto& d : dirs)
            for (const auto& e : fs::directory_iterator(d)) {
                const std::string name = e.path().filename().string();
                if (name.rfind("store_agent", 0) != 0 || e.path().extension() != ".bin") continue;
                const runtime::CollabStore st = runtime::ReadCollabStore(e.path().string());
                const auto entry = planning::mission::EntryPoses(scene, int(st.agent) + 1);
                const core::SE3 T_World_W = entry.at(st.agent);
                ac::SolveInput in;
                core::PoseGraphValue& g = in.graphs[st.agent];
                for (const auto& r : st.records) {
                    core::PoseNodeValue n;
                    n.keyframe = r.id;
                    n.pose = ac::ToPoseValue(r.T_W_C);
                    g.nodes.push_back(n);
                    if (r.odo_from) {
                        core::LoopEdgeValue o;
                        o.agent_a = o.agent_b = st.agent;
                        o.keyframe_a = *r.odo_from, o.keyframe_b = r.id;
                        o.measurement = ac::ToPoseValue(r.odo_Z);
                        core::Mat6 I;
                        for (int q = 0; q < 36; ++q) I(q / 6, q % 6) = r.odo_information[size_t(q)];
                        o.information = ac::PackInformation(I);
                        g.edges.push_back(o);
                    }
                }
                const auto chain = ac::ChainCovariances(in);
                size_t used = 0;
                for (const auto& r : st.records) {
                    const auto c = chain.find({st.agent, r.id});
                    if (!r.has_gt || c == chain.end() || c->second.second.isZero(0)) continue;
                    const core::SE3 gt = T_World_W.inverse() * (r.gt_T_W_B * r.T_B_C);
                    const core::Vec6 dlt = core::SE3Log(gt * r.T_W_C.inverse());
                    const double v = dlt.dot(c->second.second.ldlt().solve(dlt));
                    if (std::isfinite(v)) nees.push_back(v), ++used;
                }
                y << "  - {store: " << e.path().string() << ", agent: " << st.agent << ", keyframes: " << used << "}\n";
            }
        if (nees.empty()) throw std::runtime_error("no keyframes with ground truth and a composed covariance");
        std::vector<double> s = nees;
        std::sort(s.begin(), s.end());
        auto q = [&](double p) { return s[std::min(s.size() - 1, size_t(p * double(s.size())))]; };
        double mean = 0;
        for (double v : s) mean += v;
        mean /= double(s.size());
        const double kappa = std::max(1.0, mean / 6.0);
        y << "keyframes: " << s.size() << "\nnees: {mean: " << mean << ", median: " << q(0.5) << ", p90: " << q(0.9)
          << ", p99: " << q(0.99) << "}\nkappa: " << kappa << "\n";
        std::printf("entry_nees: %zu keyframes, NEES mean %.3f median %.3f p99 %.3f -> kappa %.4f\n", s.size(), mean,
                    q(0.5), q(0.99), kappa);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_entry_nees: %s\n", e.what());
        return 1;
    }
}
