// ---------------------------------------------------------------------------
// ariadne_euroc_vo: run the 1.5 placeholder stereo VO over one EuRoC sequence
// and score it against ground truth (Invariant 5: datasets before simulator).
//
//   ariadne_euroc_vo --sequence DIR --perception-config YAML --slam-config YAML
//                    --out DIR [--frames N] [--seed S] [--outages start:length,...]
//
// The pipeline and scoring live in stereo_vo_runner.hpp, shared with
// ariadne_sim_vo; this file only binds the EuRoC source.
// ---------------------------------------------------------------------------
#include "runtime/tools/stereo_vo_runner.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    ariadne::tools::StereoVoArgs a;
    std::string sequence;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); return 2; }
        const std::string v = argv[++i];
        if (k == "--sequence") sequence = v;
        else if (k == "--perception-config") a.perception_config = v;
        else if (k == "--slam-config") a.slam_config = v;
        else if (k == "--out") a.out = v;
        else if (k == "--frames") a.frames = std::stoi(v);
        else if (k == "--seed") a.seed = std::stoull(v);
        else if (k == "--outages") {   // "start:length,start:length" in seconds from the first frame
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) {
                const size_t c = item.find(':');
                if (c == std::string::npos) { std::fprintf(stderr, "--outages: expected start:length\n"); return 2; }
                a.outages.emplace_back(std::stod(item.substr(0, c)), std::stod(item.substr(c + 1)));
            }
        }
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (sequence.empty() || a.perception_config.empty() || a.slam_config.empty() || a.out.empty()) {
        std::fprintf(stderr, "usage: %s --sequence DIR --perception-config YAML --slam-config YAML --out DIR "
                             "[--frames N] [--seed S]\n", argv[0]);
        return 2;
    }
    ariadne::runtime::EurocSource source(sequence);
    return ariadne::tools::RunStereoVo(source, a, argc, argv);
}
