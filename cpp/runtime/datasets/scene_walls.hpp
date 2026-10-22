#pragma once
// ---------------------------------------------------------------------------
// 3.3 (D102): the sim scene's GT wall faces and the agents' GT body tracks, for the comms node's link model
// (collaboration/comms/rf_link_model.hpp). File I/O stays here, in runtime; the collaboration library gets plain
// triangles and tracks.
//
// Walls: every scene.json surface of kind "wall", read from meshes/<name>.obj (worldgen writes one quad = two
// triangles per surface). Worldgen builds walls as the faces between free and solid grid cells (geometry.py
// wall_runs), so a doorway is free space with no face across it, and a wall block has a face on each side; the
// surface normal points into free space. rf::WallsCrossed counts face ENTRIES, i.e. blocks.
// ---------------------------------------------------------------------------
#include "collaboration/comms/rf_link_model.hpp"
#include "runtime/datasets/stereo_source.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::runtime {

inline std::vector<collaboration::WallTriangle> LoadSceneWalls(const std::string& scene) {
    const YAML::Node sc = YAML::LoadFile(scene + "/scene.json");
    std::vector<collaboration::WallTriangle> out;
    for (const auto& s : sc["surfaces"]) {
        if (s["kind"].as<std::string>() != "wall") continue;
        const std::string name = s["name"].as<std::string>();
        const auto nv = s["normal"].as<std::vector<double>>();
        const core::Vec3 n(nv.at(0), nv.at(1), nv.at(2));
        std::ifstream in(scene + "/meshes/" + name + ".obj");
        if (!in) throw std::runtime_error("scene walls: cannot open meshes/" + name + ".obj");
        std::vector<core::Vec3> v;
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string tag;
            ls >> tag;
            if (tag == "v") {
                double x, y, z;
                ls >> x >> y >> z;
                v.emplace_back(x, y, z);
            } else if (tag == "f") {
                int idx[3];
                for (int& i : idx) {
                    std::string tok;
                    ls >> tok;
                    i = std::stoi(tok.substr(0, tok.find('/'))) - 1;   // "v/vt/vn", 1-based
                }
                out.push_back({v.at(idx[0]), v.at(idx[1]), v.at(idx[2]), n, s["sid"].as<uint64_t>()});
            }
        }
    }
    return out;
}

/// Every mission agent's GT body positions (SimSource's canonical GT).
inline std::map<collaboration::AgentId, collaboration::AgentTrack> LoadAgentTracks(const std::string& rec,
                                                                                   const std::string& scene,
                                                                                   const std::string& mission = "default") {
    const YAML::Node m = YAML::LoadFile(scene + "/mission_" + mission + ".json");
    std::map<collaboration::AgentId, collaboration::AgentTrack> out;
    for (const auto& a : m["agents"]) {
        const int k = a["agent"].as<int>();
        SimSource src(rec, scene, k, mission);
        collaboration::AgentTrack tr;
        for (const TimedPose& p : src.GroundTruth()) {
            tr.t_ns.push_back(p.t_ns);
            tr.p.push_back(p.T.t);
        }
        out[collaboration::AgentId(k)] = std::move(tr);
    }
    return out;
}

}  // namespace ariadne::runtime
