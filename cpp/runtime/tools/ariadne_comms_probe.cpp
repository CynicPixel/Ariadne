// ---------------------------------------------------------------------------
// ariadne_comms_probe: the 3.3 comms node's link state along a sim mission's GT trajectories (D102).
//
//   ariadne_comms_probe --rec DIR --scene DIR --comms YAML --out DIR [--rate-hz R]
//
// For every directed agent pair, at R Hz over the mission: distance, walls crossed, mean received power, up,
// loss_p (for the scenario's datagram size) and loss_p for a 40-byte heartbeat (the seam limit, COMMS_3.3 L8).
// Writes links.csv and summary.yaml: per pair the fraction of time up and with loss < 1 %, the wall histogram,
// and the L8 table (goodput vs datagram size at the pair's median received power).
// Evidence that link state follows range, walls and the scripted faults; it moves no protocol traffic (that needs
// 3.2's sim integration).
// ---------------------------------------------------------------------------
#include "collaboration/comms/comms_node.hpp"
#include "core/serialization/serializer.hpp"
#include "runtime/datasets/scene_walls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace ac = ariadne::collaboration;
namespace core = ariadne::core;

int main(int argc, char** argv) {
    std::string rec, scene, comms, out;
    double rate_hz = 1.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--rec") rec = v;
        else if (k == "--scene") scene = v;
        else if (k == "--comms") comms = v;
        else if (k == "--out") out = v;
        else if (k == "--rate-hz") rate_hz = std::stod(v);
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (rec.empty() || scene.empty() || comms.empty() || out.empty() || !(rate_hz > 0.0)) {
        std::fprintf(stderr, "usage: %s --rec DIR --scene DIR --comms YAML --out DIR [--rate-hz R]\n", argv[0]);
        return 2;
    }
    try {
        const ac::CommsScenario sc = ac::LoadCommsScenario(comms);
        const auto tracks = ariadne::runtime::LoadAgentTracks(rec, scene);
        const auto walls = ariadne::runtime::LoadSceneWalls(scene);
        int64_t t0 = INT64_MAX, t1 = INT64_MIN;
        for (const auto& [a, tr] : tracks) t0 = std::min(t0, tr.t_ns.front()), t1 = std::max(t1, tr.t_ns.back());
        const auto base = ac::MakeLinkModel(sc, tracks, walls);
        ac::CommsNode node(*base, 1, core::Timestamp{t0});
        node.Apply(sc);
        // The same physics at 40 bytes: what a heartbeat would see (the seam computes one size).
        ac::RfParams small = sc.rf;
        small.datagram_bytes = 40;
        if (sc.mode == "range_limited") small.wall_loss_db = 0.0;
        const ac::RfLinkModel hb(small, tracks, sc.mode == "attenuated" ? walls : std::vector<ac::WallTriangle>{});
        const auto* rf = dynamic_cast<const ac::RfLinkModel*>(base.get());

        std::filesystem::create_directories(out);
        std::ofstream csv(std::filesystem::path(out) / "links.csv");
        csv << "t_s,from,to,distance_m,walls,rx_mean_dbm,up,loss_p,loss_p_40B\n";
        struct Acc { int n = 0, up = 0, good = 0; std::map<int, int> walls; std::vector<double> rx; };
        std::map<std::pair<ac::AgentId, ac::AgentId>, Acc> acc;
        const int64_t step = int64_t(1e9 / rate_hz);
        for (int64_t t = t0; t <= t1; t += step) {
            for (const auto& [a, ta] : tracks) {
                for (const auto& [b, tb] : tracks) {
                    if (a == b) continue;
                    const core::Timestamp ts{t};
                    const ac::LinkState s = node.links().Sample(a, b, ts);   // base + faults: what the Network enforces
                    ac::RfLinkDetail d;
                    if (rf) d = rf->Detail(a, b, ts);
                    const double l40 = (rf && s.up) ? hb.Detail(a, b, ts).state.loss_p : (s.up ? 0.0 : 1.0);
                    csv << (t - t0) * 1e-9 << ',' << a << ',' << b << ',' << d.distance_m << ',' << d.walls << ','
                        << d.rx_mean_dbm << ',' << int(s.up) << ',' << (s.up ? s.loss_p : 1.0) << ',' << l40 << '\n';
                    Acc& c = acc[{a, b}];
                    ++c.n;
                    c.up += s.up;
                    c.good += s.up && s.loss_p < 0.01;
                    ++c.walls[d.walls];
                    if (rf && std::isfinite(d.rx_mean_dbm)) c.rx.push_back(d.rx_mean_dbm);
                }
            }
        }
        std::ofstream sm(std::filesystem::path(out) / "summary.yaml");
        sm << "scenario: " << sc.name << "\nmode: " << sc.mode << "\nrate_hz: " << rate_hz
           << "\nduration_s: " << (t1 - t0) * 1e-9 << "\nwall_triangles: " << walls.size() << "\npairs:\n";
        for (auto& [k, c] : acc) {
            sm << "  - {from: " << k.first << ", to: " << k.second << ", samples: " << c.n
               << ", up_fraction: " << double(c.up) / c.n << ", loss_below_1pct_fraction: " << double(c.good) / c.n
               << ", walls_histogram: {";
            bool first = true;
            for (const auto& [w, n] : c.walls) sm << (first ? "" : ", ") << w << ": " << n, first = false;
            sm << "}";
            if (rf && !c.rx.empty()) {
                std::nth_element(c.rx.begin(), c.rx.begin() + c.rx.size() / 2, c.rx.end());
                const double med = c.rx[c.rx.size() / 2];
                // L8: goodput fraction (payload / wire) x (1 - E[PER]) at the median received power, per datagram size.
                sm << ", rx_median_dbm: " << med << ", l8_goodput: {";
                first = true;
                for (int payload : {64, 256, 512, 1024, int(core::kDatagramPayloadBytes)}) {
                    const double wire = payload + double(core::kDatagramOverheadBytes);
                    const double per = ac::rf::ExpectedPer(med, sc.rf.sigma_db, sc.rf.noise_floor_dbm, wire);
                    sm << (first ? "" : ", ") << payload << ": " << (payload / wire) * (1.0 - per);
                    first = false;
                }
                sm << "}";
            }
            sm << "}\n";
        }
        std::printf("%s: %zu agents, %zu wall triangles, %.0f s at %.1f Hz -> %s\n", sc.name.c_str(), tracks.size(),
                    walls.size(), (t1 - t0) * 1e-9, rate_hz, out.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_comms_probe: %s\n", e.what());
        return 1;
    }
}
