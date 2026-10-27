// ariadne_frame_dump: write chosen left-camera frames of a sim recording as colour PNGs, exactly the RGB pixels the
// pipeline sees (StereoFrame::left_rgb: rgb8 decoded, the D41 lens applied, no degradation). These are the reference
// inputs of the sim-rig exports (models.sh eigenplaces-sim / aliked-lg-sim, perception.md §14).
//
//   ariadne_frame_dump --rec RECDIR --scene SCENEDIR [--agent 0] --frames 400,405,1500 --out DIR
//
// The PNG is written in OpenCV's BGR file convention (cv::imwrite), so any viewer shows the true colours and
// cv::imread / cv2.imread give BGR back; converting to RGB is the reader's job, explicitly.
#include "runtime/datasets/stereo_source.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    std::string rec, scene, out, frames;
    int agent = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--rec") rec = v;
        else if (k == "--scene") scene = v;
        else if (k == "--agent") agent = std::stoi(v);
        else if (k == "--frames") frames = v;
        else if (k == "--out") out = v;
        else {
            std::fprintf(stderr, "unknown flag %s\n", k.c_str());
            return 2;
        }
    }
    if (rec.empty() || scene.empty() || out.empty() || frames.empty()) {
        std::fprintf(stderr, "usage: %s --rec RECDIR --scene SCENEDIR [--agent N] --frames i,j,... --out DIR\n", argv[0]);
        return 2;
    }
    try {
        std::set<uint64_t> want;
        std::stringstream ss(frames);
        for (std::string t; std::getline(ss, t, ',');) want.insert(std::stoull(t));
        std::filesystem::create_directories(out);
        ariadne::runtime::SimSource source(rec, scene, agent);
        size_t written = 0;
        source.ForEachFrame(*want.rbegin() + 1, [&](const ariadne::runtime::StereoFrame& f) {
            if (!want.count(f.id)) return;
            if (f.left_rgb.empty()) throw std::runtime_error("frame has no left_rgb");
            cv::Mat bgr;
            cv::cvtColor(f.left_rgb, bgr, cv::COLOR_RGB2BGR);
            const std::string p = out + "/agent" + std::to_string(agent) + "_f" + std::to_string(f.id) + ".png";
            if (!cv::imwrite(p, bgr)) throw std::runtime_error("cannot write " + p);
            std::printf("%s %dx%d t_ns=%lld\n", p.c_str(), bgr.cols, bgr.rows, static_cast<long long>(f.t_ns));
            ++written;
        });
        if (written != want.size()) throw std::runtime_error("only " + std::to_string(written) + " frames found");
        std::printf("scene %s, bag %s\n", source.SceneName().c_str(), source.BagChecksum().c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_frame_dump: %s\n", e.what());
        return 1;
    }
    return 0;
}
