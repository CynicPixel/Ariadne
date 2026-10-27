#pragma once
// ---------------------------------------------------------------------------
// Operator directives (design agentic.md §0b; arch §23, §29): the SubT "directed autonomy" vocabulary
// (arXiv:2212.05626: geometric prioritisation regions + direct task assignment on an allocator; CERBERUS
// arXiv:2207.04914: explore within/beyond comms with a time budget, waypoint as last resort).
//
// A directive is a RECORDED INPUT: the runtime takes new ones only at a replan barrier, stamps the replan time it takes
// effect (effective_ns), appends it to <run>/directives.jsonl, and replay re-applies that file, so live = replay.
// Authors: "operator" (console) or "llm-proposed/operator-confirmed" (the LLM proposes, a person confirms; §25.1).
// Planners receive the active set in PlanContext and apply it to their task values / constraints.
// ---------------------------------------------------------------------------
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ariadne::planning::mission {

struct Directive {
    std::string id, author, kind;   // kind: priority_region | keep_out | assign | explore | goto | hold | release | cancel
    std::string target;             // cancel: the id it withdraws
    int agent = -1;                 // -1 = every agent (priority_region, keep_out)
    std::vector<std::pair<double, double>> polygon;   // world xy (m), for the region kinds
    double weight = 1.0;            // priority_region: task value multiplier (> 1 favours the region)
    double x = 0, y = 0, yaw = 0;   // goto
    double time_budget_s = 0;       // explore: active for this long after effective_ns (0 = until released)
    int64_t effective_ns = -1;      // the replan time it took effect (stamped by the runtime)
    double wall_unix_s = 0;         // when the author submitted it (provenance only; never used to decide)
    std::string text;               // the operator's words, when it came from language

    bool Active(int64_t T) const {
        return effective_ns >= 0 && T >= effective_ns &&
               (kind != "explore" || time_budget_s <= 0 || T < effective_ns + int64_t(time_budget_s * 1e9));
    }
    bool Covers(double px, double py) const {   // even-odd point-in-polygon
        bool in = false;
        for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            const auto [xi, yi] = polygon[i];
            const auto [xj, yj] = polygon[j];
            if ((yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) in = !in;
        }
        return in;
    }
    /// Distance (m) from (px, py) to the region: 0 inside, else to the nearest polygon edge.
    double Distance(double px, double py) const {
        if (Covers(px, py)) return 0;
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            const auto [ax, ay] = polygon[j];
            const auto [bx, by] = polygon[i];
            const double dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
            const double u = L2 > 0 ? std::clamp(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0) : 0.0;
            best = std::min(best, std::hypot(ax + u * dx - px, ay + u * dy - py));
        }
        return best;
    }
};

inline Directive ParseDirective(const YAML::Node& n) {
    Directive d;
    d.id = n["id"].as<std::string>();
    d.author = n["author"].as<std::string>();
    d.kind = n["kind"].as<std::string>();
    static const std::vector<std::string> kinds{"priority_region", "keep_out", "assign", "explore", "goto", "hold", "release", "cancel"};
    if (std::find(kinds.begin(), kinds.end(), d.kind) == kinds.end()) throw std::invalid_argument("directive: kind " + d.kind);
    if (n["agent"]) d.agent = n["agent"].as<int>();
    if (n["polygon"]) for (const auto& p : n["polygon"]) d.polygon.emplace_back(p[0].as<double>(), p[1].as<double>());
    if (n["weight"]) d.weight = n["weight"].as<double>();
    if (n["x"]) d.x = n["x"].as<double>(), d.y = n["y"].as<double>();
    if (n["yaw"]) d.yaw = n["yaw"].as<double>();
    if (n["time_budget_s"]) d.time_budget_s = n["time_budget_s"].as<double>();
    if (n["effective_ns"]) d.effective_ns = n["effective_ns"].as<int64_t>();
    if (n["wall_unix_s"]) d.wall_unix_s = n["wall_unix_s"].as<double>();
    if (n["text"]) d.text = n["text"].as<std::string>();
    if (n["target"]) d.target = n["target"].as<std::string>();
    if (d.kind == "cancel" && d.target.empty()) throw std::invalid_argument("directive " + d.id + ": cancel needs a target");
    const bool region = d.kind == "priority_region" || d.kind == "keep_out" || d.kind == "assign" || d.kind == "explore";
    if (region && d.polygon.size() < 3) throw std::invalid_argument("directive " + d.id + ": polygon needs >= 3 points");
    if ((d.kind == "assign" || d.kind == "explore" || d.kind == "goto" || d.kind == "hold" || d.kind == "release") &&
        d.agent < 0) throw std::invalid_argument("directive " + d.id + ": needs an agent");
    if (d.kind == "priority_region" && !(d.weight > 0)) throw std::invalid_argument("directive " + d.id + ": weight > 0");
    return d;
}

/// One JSON line (the recorded form; ParseDirective reads it back, so replay applies the identical directive).
inline std::string DirectiveJson(const Directive& d) {
    std::ostringstream o;
    o << std::setprecision(17);
    auto q = [](const std::string& v) {
        std::string r = "\"";
        for (char c : v) { if (c == '"' || c == '\\') r += '\\'; r += c; }
        return r + "\"";
    };
    o << "{\"id\":" << q(d.id) << ",\"author\":" << q(d.author) << ",\"kind\":" << q(d.kind) << ",\"agent\":" << d.agent
      << ",\"polygon\":[";
    for (size_t i = 0; i < d.polygon.size(); ++i) o << (i ? "," : "") << "[" << d.polygon[i].first << "," << d.polygon[i].second << "]";
    o << "],\"weight\":" << d.weight << ",\"x\":" << d.x << ",\"y\":" << d.y << ",\"yaw\":" << d.yaw
      << ",\"time_budget_s\":" << d.time_budget_s << ",\"effective_ns\":" << d.effective_ns << ",\"wall_unix_s\":"
      << d.wall_unix_s << ",\"text\":" << q(d.text) << ",\"target\":" << q(d.target) << "}";
    return o.str();
}

/// The directives in force at T, in order: cancels withdraw their target; for each agent, release ends the agent-level
/// ones (hold, goto, assign, explore) issued before it, and a later agent-level directive replaces an earlier one.
inline std::vector<Directive> ActiveDirectives(const std::vector<Directive>& all, int64_t T) {
    std::vector<Directive> out;
    for (const auto& d : all) {
        if (d.effective_ns < 0 || d.effective_ns > T) continue;
        if (d.kind == "cancel") {
            out.erase(std::remove_if(out.begin(), out.end(), [&](const Directive& x) { return x.id == d.target; }), out.end());
            continue;
        }
        const bool agent_level = d.kind == "hold" || d.kind == "goto" || d.kind == "assign" || d.kind == "explore" || d.kind == "release";
        if (agent_level) {
            out.erase(std::remove_if(out.begin(), out.end(), [&](const Directive& x) {
                return x.agent == d.agent && (x.kind == "hold" || x.kind == "goto" || x.kind == "assign" || x.kind == "explore");
            }), out.end());
            if (d.kind == "release") continue;
        }
        if (d.Active(T)) out.push_back(d);
    }
    return out;
}

}  // namespace ariadne::planning::mission
