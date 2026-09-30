#pragma once
// ---------------------------------------------------------------------------
// The KinematicMover's <plugin> configuration and its per-tick sample, shared by
// the gz plugin (KinematicMover.cc) and the offline canonical-stream writer
// (tools/canonical_streams.cc, DECISIONS D39).
//
// Why one header: the canonical IMU / GT-pose bag must hold, for every stamp the
// plugin publishes, the SAME doubles the plugin publishes. Sharing the parser
// (same SDF text -> same doubles) and the sampler (same ns -> same arithmetic,
// compiled with -ffp-contract=off in both targets) makes that true by
// construction; sim_record.py then checks it byte-for-byte against every live
// sample that reached the bag.
// ---------------------------------------------------------------------------
#include "core/trajectory/scripted.hpp"

#include <sdf/Element.hh>

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::sim {

using ariadne::core::scripted::Script;
using ariadne::core::scripted::Segment;
using ariadne::core::scripted::State;
using ariadne::core::scripted::Vec5;

struct MoverConfig {
    std::string frame_id;        // header.frame_id of both streams, e.g. agent_0/base_link
    double gravity = 0.0;        // m/s^2, magnitude along -z_W
    int64_t period_ns = 0;       // IMU / GT-pose publication period
    std::string imu_topic, pose_topic;
    Script script;
};

/// One published instant: the script state and the two messages' payload values.
struct MoverSample {
    int64_t ns = 0;
    State state;
    ariadne::core::scripted::ImuSample imu;
    std::array<double, 4> q{};   // (w, x, y, z) of R = Rz(yaw) Ry(pitch)
};

/// The ONE definition of what the mover publishes at simulated time ns.
inline MoverSample SampleAt(const MoverConfig& c, int64_t ns) {
    MoverSample m;
    m.ns = ns;
    m.state = c.script.Eval(double(ns) * 1e-9);
    m.imu = ariadne::core::scripted::Imu(m.state, c.gravity);
    m.q = ariadne::core::scripted::Quaternion(m.state.pose[3], m.state.pose[4]);
    return m;
}

namespace detail {

inline Vec5 ParseVec5(const std::string& text, const std::string& what) {
    std::istringstream in(text);
    Vec5 v{};
    for (auto& x : v) {
        if (!(in >> x)) throw std::runtime_error("KinematicMover: bad 5-vector in " + what + ": '" + text + "'");
    }
    std::string rest;
    if (in >> rest) throw std::runtime_error("KinematicMover: extra values in " + what);
    return v;
}

template <typename T>
T RequiredAttr(const sdf::ElementPtr& e, const std::string& name) {
    if (!e->HasAttribute(name)) throw std::runtime_error("KinematicMover: <segment> lacks '" + name + "'");
    T v{};
    std::istringstream in(e->GetAttribute(name)->GetAsString());
    if (!(in >> v)) throw std::runtime_error("KinematicMover: bad attribute '" + name + "'");
    return v;
}

template <typename T>
T RequiredElem(const std::shared_ptr<const sdf::Element>& sdf, const std::string& name) {
    if (!sdf->HasElement(name)) throw std::runtime_error("KinematicMover: missing <" + name + ">");
    return sdf->Get<T>(name);
}

}  // namespace detail

/// Parses and validates the <plugin> element (everything except the ESDF, which
/// only the plugin's envelope check needs). Throws on any malformed field.
inline MoverConfig ParseMoverConfig(const std::shared_ptr<const sdf::Element>& sdf) {
    using detail::RequiredAttr;
    using detail::RequiredElem;
    MoverConfig c;
    c.frame_id = RequiredElem<std::string>(sdf, "frame_id");
    c.gravity = RequiredElem<double>(sdf, "gravity");
    c.period_ns = RequiredElem<int>(sdf, "imu_period_ns");   // sdf params have no int64 alternative
    if (c.period_ns <= 0) throw std::runtime_error("KinematicMover: imu_period_ns must be > 0");
    c.imu_topic = RequiredElem<std::string>(sdf, "imu_topic");
    c.pose_topic = RequiredElem<std::string>(sdf, "pose_topic");

    if (!sdf->HasElement("script")) throw std::runtime_error("KinematicMover: missing <script>");
    auto script = std::const_pointer_cast<sdf::Element>(sdf)->GetElement("script");
    std::vector<Segment> segs;
    for (auto e = script->GetFirstElement(); e; e = e->GetNextElement()) {
        if (e->GetName() != "segment") continue;
        Segment s;
        s.kind = Segment::ParseKind(RequiredAttr<std::string>(e, "kind"));
        s.t0 = RequiredAttr<double>(e, "t0");
        s.duration = RequiredAttr<double>(e, "duration");
        s.profile = {RequiredAttr<double>(e, "D"), RequiredAttr<double>(e, "Ta"), RequiredAttr<double>(e, "vc")};
        for (const char* k : {"start", "end"}) {   // whole attribute text: a 5-vector has spaces
            if (!e->HasAttribute(k)) throw std::runtime_error(std::string("KinematicMover: <segment> lacks '") + k + "'");
        }
        s.start = detail::ParseVec5(e->GetAttribute("start")->GetAsString(), "start");
        s.end = detail::ParseVec5(e->GetAttribute("end")->GetAsString(), "end");
        if (e->HasAttribute("tag")) s.tag = e->GetAttribute("tag")->GetAsString();
        segs.push_back(s);
    }
    if (segs.empty()) throw std::runtime_error("KinematicMover: empty <script>");
    c.script = Script(std::move(segs));
    c.script.Validate();
    return c;
}

}  // namespace ariadne::sim
