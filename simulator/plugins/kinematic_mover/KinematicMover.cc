// ---------------------------------------------------------------------------
// ariadne::sim::KinematicMover — the kinematic agent mover of arch §5.8 /
// design SIM_1.1-1.3 §5. A gz-sim system attached to an agent <model>:
//
//   Configure  parse the script (<segment> elements written by the mission
//              composer), validate continuity, check the collision envelope
//              against the scene's ESDF (every 10 ms of script time), and
//              self-check that gz's Pose3d rotation equals the script's
//              R = Rz(yaw) * Ry(pitch). Any failure throws: the run fails
//              loudly instead of recording a wrong world.
//   PreUpdate  write the model's Pose at info.simTime (no Physics system is
//              loaded; SMOKE §4g: cameras, depth and lamp follow the link and
//              render at exactly this pose), and at every multiple of the IMU
//              period publish the clean analytic IMU and the exact GT pose,
//              stamped with the same simulated time.
//
// The trajectory math lives in cpp/core/trajectory/scripted.hpp, tested against
// the Python generator's golden samples; the <plugin> parser and the per-tick
// sample live in MoverScript.hh, shared with the offline canonical-stream writer
// (tools/canonical_streams.cc, D39). This file only adapts them to gz.
// IMU noise is NOT added here: it is a seeded consumption-time operator (D37).
//
// LIVE MODE (design LIVE_RUNTIME_L1 D151; only with a <live> element): the mover
// also takes commands on a gz Pose_V topic (bridged from geometry_msgs/PoseArray):
// 0 poses = hold(), 1 = goto(pose), N = follow(path); the header stamp is the
// command id. A command replaces every segment not yet started; the segment in
// progress finishes (all end at rest). New segments come from
// core/trajectory/command_planner.hpp (the worldgen profile family) and are
// checked against the ESDF like the script: a violation refuses the whole command
// (the agent holds). Every command is answered on a StringMsg state topic (JSON),
// and the executed script is appended to a JSONL log after every change, exact
// doubles, so the canonical streams of a live run can be regenerated (D39).
// ---------------------------------------------------------------------------
#include "kinematic_mover/MoverScript.hh"

#include "core/trajectory/command_planner.hpp"

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/msgs/imu.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/SourceFilePath.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::sim {

namespace {

using detail::RequiredElem;

/// The scene's navigable ESDF (worldgen build.py: raw float32 LE, row 0 = south).
struct Esdf {
    std::vector<float> v;
    long rows = 0, cols = 0;
    long ox = 0, oy = 0;       // origin in grid cells
    double cell = 0.0;

    double At(double x, double y) const {
        const long c = long(std::floor(x / cell)) - ox;   // the Python formula exactly
        const long r = long(std::floor(y / cell)) - oy;
        if (r < 0 || c < 0 || r >= rows || c >= cols) return -1e9;   // outside = solid
        return v[size_t(r) * size_t(cols) + size_t(c)];
    }
};

}  // namespace

class KinematicMover : public gz::sim::System,
                       public gz::sim::ISystemConfigure,
                       public gz::sim::ISystemPreUpdate {
public:
    void Configure(const gz::sim::Entity& entity, const std::shared_ptr<const sdf::Element>& sdf,
                   gz::sim::EntityComponentManager& ecm, gz::sim::EventManager&) override {
        model_ = entity;
        cfg_ = ParseMoverConfig(sdf);   // throws on any malformed field or discontinuous script
        imu_pub_ = node_.Advertise<gz::msgs::IMU>(cfg_.imu_topic);
        pose_pub_ = node_.Advertise<gz::msgs::Pose>(cfg_.pose_topic);

        SelfCheckRotationConvention();
        CheckEnvelope(sdf, ecm);
        if (sdf->HasElement("live")) ConfigureLive(sdf);
        gzmsg << "[KinematicMover] " << cfg_.frame_id << ": " << cfg_.script.segments().size() << " segments, "
              << cfg_.script.duration() << " s, envelope ok\n";
    }

    void PreUpdate(const gz::sim::UpdateInfo& info, gz::sim::EntityComponentManager& ecm) override {
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
        // Commands are taken even while paused (D165 lockstep: a command sent at the barrier T is applied AT T and
        // answered before the runtime resumes the world); motion below advances only when running.
        if (live_) ApplyCommands(ns);
        if (info.paused) return;
        const MoverSample m = SampleAt(cfg_, ns);
        const Vec5& s = m.state.pose;
        const gz::math::Pose3d pose(s[0], s[1], s[2], 0.0, s[4], s[3]);
        auto* comp = ecm.Component<gz::sim::components::Pose>(model_);
        if (comp == nullptr) {
            ecm.CreateComponent(model_, gz::sim::components::Pose(pose));
        } else {
            *comp = gz::sim::components::Pose(pose);
            ecm.SetChanged(model_, gz::sim::components::Pose::typeId, gz::sim::ComponentState::PeriodicChange);
        }
        if (ns % cfg_.period_ns == 0 && ns != last_pub_ns_) {
            last_pub_ns_ = ns;
            Publish(m);
        }
    }

private:
    void Stamp(gz::msgs::Header* h, int64_t ns) const {
        h->mutable_stamp()->set_sec(ns / 1000000000);
        h->mutable_stamp()->set_nsec(int32_t(ns % 1000000000));
        auto* kv = h->add_data();
        kv->set_key("frame_id");
        kv->add_value(cfg_.frame_id);
    }

    void Publish(const MoverSample& m) {
        gz::msgs::IMU imu;
        Stamp(imu.mutable_header(), m.ns);
        imu.set_entity_name(cfg_.frame_id);
        // Orientation is deliberately NOT the ground truth: an IMU does not measure
        // it, and the estimator must not see GT. Identity, documented as unused.
        imu.mutable_orientation()->set_w(1.0);
        imu.mutable_angular_velocity()->set_x(m.imu.gyro[0]);
        imu.mutable_angular_velocity()->set_y(m.imu.gyro[1]);
        imu.mutable_angular_velocity()->set_z(m.imu.gyro[2]);
        imu.mutable_linear_acceleration()->set_x(m.imu.accel[0]);
        imu.mutable_linear_acceleration()->set_y(m.imu.accel[1]);
        imu.mutable_linear_acceleration()->set_z(m.imu.accel[2]);
        imu_pub_.Publish(imu);

        gz::msgs::Pose p;
        Stamp(p.mutable_header(), m.ns);
        p.set_name(cfg_.frame_id);
        p.mutable_position()->set_x(m.state.pose[0]);
        p.mutable_position()->set_y(m.state.pose[1]);
        p.mutable_position()->set_z(m.state.pose[2]);
        p.mutable_orientation()->set_w(m.q[0]);
        p.mutable_orientation()->set_x(m.q[1]);
        p.mutable_orientation()->set_y(m.q[2]);
        p.mutable_orientation()->set_z(m.q[3]);
        pose_pub_.Publish(p);
    }

    /// The SDF file this agent came from, for resolving relative paths. sdformat does
    /// not propagate FilePath to <plugin> elements, so fall back to the model's (then
    /// the world's) SourceFilePath component, which gz-sim sets from the loaded file.
    std::string SourceFile(const std::shared_ptr<const sdf::Element>& sdf,
                           const gz::sim::EntityComponentManager& ecm) const {
        if (!sdf->FilePath().empty()) return sdf->FilePath();
        if (const auto* c = ecm.Component<gz::sim::components::SourceFilePath>(model_); c && !c->Data().empty()) {
            return c->Data();
        }
        const auto world = gz::sim::worldEntity(ecm);
        if (const auto* c = ecm.Component<gz::sim::components::SourceFilePath>(world); c && !c->Data().empty()) {
            return c->Data();
        }
        return {};
    }

    void CheckEnvelope(const std::shared_ptr<const sdf::Element>& sdf, const gz::sim::EntityComponentManager& ecm) {
        auto el = std::const_pointer_cast<sdf::Element>(sdf)->GetElement("esdf");
        const std::string rel = el->Get<std::string>("file");
        std::filesystem::path path(rel);
        if (path.is_relative()) {
            // Relative to the SDF file that declared the agent (the mission world
            // lives in the scene directory, next to esdf_f32.bin).
            const std::string src = SourceFile(sdf, ecm);
            if (src.empty()) {
                throw std::runtime_error("KinematicMover: relative ESDF path '" + rel +
                                         "' but no source file path (plugin element, model, world) is known");
            }
            path = std::filesystem::path(src).parent_path() / path;
        }
        Esdf& e = esdf_;
        e.rows = el->Get<int>("rows");
        e.cols = el->Get<int>("cols");
        e.ox = el->Get<int>("origin_x_cells");
        e.oy = el->Get<int>("origin_y_cells");
        e.cell = el->Get<double>("cell_m");
        const double rho = rho_ = RequiredElem<double>(sdf, "rho_agent");
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("KinematicMover: cannot open ESDF " + path.string());
        e.v.resize(size_t(e.rows) * size_t(e.cols));
        in.read(reinterpret_cast<char*>(e.v.data()), std::streamsize(e.v.size() * sizeof(float)));
        if (!in || in.peek() != EOF) throw std::runtime_error("KinematicMover: ESDF size != rows*cols floats");
        double worst = 1e9, worst_t = 0.0;
        for (double t = 0.0; t <= cfg_.script.duration() + 1e-9; t += 0.01) {
            const State s = cfg_.script.Eval(t);
            const double d = e.At(s.pose[0], s.pose[1]);
            if (d < worst) worst = d, worst_t = t;
        }
        if (worst < rho - 1e-9) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "KinematicMover: script leaves the collision envelope at t=%.2f s "
                          "(ESDF %.3f m < rho_agent %.3f m)", worst_t, worst, rho);
            throw std::runtime_error(buf);
        }
    }

    /// gz Pose3d(x, y, z, roll, pitch, yaw) must equal R = Rz(yaw) Ry(pitch) with roll 0.
    static void SelfCheckRotationConvention() {
        for (double yaw : {-2.0, 0.3, 1.9}) {
            for (double pitch : {-0.3, 0.2}) {
                const gz::math::Quaterniond g(0.0, pitch, yaw);
                const auto q = ariadne::core::scripted::Quaternion(yaw, pitch);
                const double dot = g.W() * q[0] + g.X() * q[1] + g.Y() * q[2] + g.Z() * q[3];
                if (std::abs(std::abs(dot) - 1.0) > 1e-12) {
                    throw std::runtime_error("KinematicMover: gz Euler convention != Rz(yaw) Ry(pitch)");
                }
            }
        }
    }


    // ---- live mode -------------------------------------------------------------------------------------------
    struct Command {
        int64_t stamp_ns = 0;
        std::string frame_id;
        std::vector<ariadne::core::scripted::Waypoint> path;
    };

    void ConfigureLive(const std::shared_ptr<const sdf::Element>& sdf) {
        auto el = std::const_pointer_cast<sdf::Element>(sdf)->GetElement("live");
        auto req = [&](const char* k) {
            if (!el->HasElement(k)) throw std::runtime_error(std::string("KinematicMover: <live> lacks <") + k + ">");
            return el->Get<std::string>(k);
        };
        const std::string cmd_topic = req("cmd_topic"), state_topic = req("state_topic");
        log_path_ = req("log_file");
        limits_.v_cap = std::stod(req("v_cap"));
        limits_.a_max = std::stod(req("a_max"));
        limits_.omega_max = std::stod(req("omega_max"));
        limits_.alpha_max = std::stod(req("alpha_max"));
        limits_.v_lk_per_m = std::stod(req("v_lk_per_m"));
        ceiling_m_ = std::stod(req("ceiling_m"));
        segs_ = cfg_.script.segments();
        state_pub_ = node_.Advertise<gz::msgs::StringMsg>(state_topic);
        if (!node_.Subscribe(cmd_topic, &KinematicMover::OnCommand, this)) {
            throw std::runtime_error("KinematicMover: cannot subscribe to " + cmd_topic);
        }
        log_.open(log_path_, std::ios::out | std::ios::trunc);
        if (!log_) throw std::runtime_error("KinematicMover: cannot open live log " + log_path_);
        log_.precision(17);
        LogScript(-1, 0);
        live_ = true;
        gzmsg << "[KinematicMover] " << cfg_.frame_id << ": LIVE, commands on " << cmd_topic << "\n";
    }

    /// gz-transport thread: queue only; PreUpdate applies at a tick boundary.
    void OnCommand(const gz::msgs::Pose_V& m) {
        Command c;
        c.stamp_ns = int64_t(m.header().stamp().sec()) * 1000000000 + m.header().stamp().nsec();
        for (const auto& d : m.header().data()) {
            if (d.key() == "frame_id" && d.value_size() > 0) c.frame_id = d.value(0);
        }
        for (int i = 0; i < m.pose_size(); ++i) {
            const auto& p = m.pose(i);
            ariadne::core::scripted::Waypoint w;
            w.x = p.position().x();
            w.y = p.position().y();
            w.z = p.position().z();
            const auto& q = p.orientation();
            // A zero quaternion = "no final yaw" (a follow that may end in any heading).
            w.has_yaw = !(q.w() == 0 && q.x() == 0 && q.y() == 0 && q.z() == 0);
            if (w.has_yaw) w.yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()), 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
            c.path.push_back(w);
        }
        std::lock_guard<std::mutex> lk(mu_);
        inbox_.push_back(std::move(c));
    }

    double ClearanceAlong(double x0, double y0, double z0, double x1, double y1, double z1) const {
        // routes.py _clearance_along: ESDF every grid cell along the move, the envelope asserted, and the vertical
        // clearance to floor and ceiling.
        const int n = std::max(int(std::ceil(std::hypot(x1 - x0, y1 - y0) / esdf_.cell)), 1);
        double cmin = 1e9;
        for (int i = 0; i <= n; ++i) {
            const double x = x0 + (x1 - x0) * i / n, y = y0 + (y1 - y0) * i / n;
            const double e = esdf_.At(x, y);
            if (e < rho_ - 1e-9) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "move leaves the collision envelope at (%.2f, %.2f): ESDF %.3f < %.3f",
                              x, y, e, rho_);
                throw std::runtime_error(buf);
            }
            cmin = std::min(cmin, e);
        }
        return std::min({cmin, z0, z1, ceiling_m_ - std::max(z0, z1)});
    }

    void ApplyCommands(int64_t ns) {
        std::deque<Command> todo;
        {
            std::lock_guard<std::mutex> lk(mu_);
            todo.swap(inbox_);
        }
        for (const Command& c : todo) Apply(c, ns);
    }

    void Apply(const Command& c, int64_t ns) {
        using namespace ariadne::core::scripted;
        const double t = double(ns) * 1e-9;
        const char* kind = c.path.empty() ? "hold" : c.path.size() == 1 ? "goto" : "follow";
        // Keep every segment that has started; the one in progress (if any) finishes.
        std::vector<Segment> keep;
        for (const Segment& s : segs_) {
            if (s.t0 <= t) keep.push_back(s);
        }
        double t_start = keep.back().t0 + keep.back().duration;
        const State at_end = keep.back().Eval(t_start);
        if (t_start < t) {   // the rig is at rest past the script's end: hold until now, so time stays contiguous
            Segment h;
            h.kind = Segment::Kind::kHold;
            h.t0 = t_start;
            h.duration = t - t_start;
            h.start = h.end = at_end.pose;
            h.tag = "live:idle";
            keep.push_back(h);
            t_start = t;
        }
        std::string reason;
        std::vector<Segment> add;
        if (c.frame_id != "world") {
            reason = "frame_id '" + c.frame_id + "' != world";
        } else {
            try {
                add = PlanCommand(at_end.pose, t_start, c.path, limits_,
                                  [this](double a, double b, double cz, double d, double e, double f) {
                                      return ClearanceAlong(a, b, cz, d, e, f);
                                  },
                                  std::string("cmd:") + std::to_string(c.stamp_ns));
                for (const Segment& s : add) {   // the start-up rule: every 10 ms of the new motion
                    for (double u = s.t0; u <= s.t0 + s.duration + 1e-9; u += 0.01) {
                        const State st = s.Eval(u);
                        if (esdf_.At(st.pose[0], st.pose[1]) < rho_ - 1e-9) {
                            throw std::runtime_error("sampled pose leaves the collision envelope");
                        }
                    }
                }
            } catch (const std::exception& e) {
                reason = e.what();
                add.clear();
            }
        }
        const bool accepted = reason.empty();
        // A refused command still drops the queued (not started) motion: refusal = hold, the fail-safe.
        std::vector<Segment> next = keep;
        next.insert(next.end(), add.begin(), add.end());
        Script sc(next);
        sc.Validate();   // throws = a planner bug, not a refusal: the run fails loudly
        segs_ = std::move(next);
        cfg_.script = std::move(sc);
        LogScript(c.stamp_ns, ns);
        const double t_end = segs_.back().t0 + segs_.back().duration;
        std::ostringstream js;
        js.precision(17);
        js << "{\"frame_id\":\"" << cfg_.frame_id << "\",\"cmd_stamp_ns\":" << c.stamp_ns << ",\"applied_ns\":" << ns
           << ",\"kind\":\"" << kind << "\",\"waypoints\":" << c.path.size() << ",\"accepted\":"
           << (accepted ? "true" : "false") << ",\"reason\":\"" << Escape(reason) << "\",\"segments\":" << add.size()
           << ",\"t_start_s\":" << t_start << ",\"t_end_s\":" << t_end << "}";
        gz::msgs::StringMsg out;
        Stamp(out.mutable_header(), ns);
        out.set_data(js.str());
        state_pub_.Publish(out);
        if (!accepted) gzwarn << "[KinematicMover] " << cfg_.frame_id << " REFUSED " << kind << " " << c.stamp_ns << ": " << reason << "\n";
    }

    static std::string Escape(const std::string& s) {
        std::string o;
        for (char ch : s) {
            if (ch == '"' || ch == '\\') o += '\\';
            o += ch;
        }
        return o;
    }

    void LogScript(int64_t cmd_stamp_ns, int64_t applied_ns) {
        log_ << "{\"cmd_stamp_ns\":" << cmd_stamp_ns << ",\"applied_ns\":" << applied_ns << ",\"segments\":[";
        for (size_t i = 0; i < segs_.size(); ++i) {
            const auto& s = segs_[i];
            const char* k = s.kind == ariadne::core::scripted::Segment::Kind::kHold ? "hold"
                            : s.kind == ariadne::core::scripted::Segment::Kind::kTurn ? "turn" : "move";
            log_ << (i ? "," : "") << "{\"kind\":\"" << k << "\",\"t0\":" << s.t0 << ",\"duration\":" << s.duration
                 << ",\"D\":" << s.profile.D << ",\"Ta\":" << s.profile.Ta << ",\"vc\":" << s.profile.vc << ",\"start\":[";
            for (int j = 0; j < 5; ++j) log_ << (j ? "," : "") << s.start[j];
            log_ << "],\"end\":[";
            for (int j = 0; j < 5; ++j) log_ << (j ? "," : "") << s.end[j];
            log_ << "],\"tag\":\"" << Escape(s.tag) << "\"}";
        }
        log_ << "]}\n";
        log_.flush();
    }

    bool live_ = false;
    std::mutex mu_;
    std::deque<Command> inbox_;
    std::vector<ariadne::core::scripted::Segment> segs_;
    ariadne::core::scripted::MotionLimits limits_;
    double ceiling_m_ = 0, rho_ = 0;
    Esdf esdf_;
    std::string log_path_;
    std::ofstream log_;
    gz::transport::Node::Publisher state_pub_;

    gz::sim::Entity model_{gz::sim::kNullEntity};
    MoverConfig cfg_;
    int64_t last_pub_ns_ = -1;
    gz::transport::Node node_;
    gz::transport::Node::Publisher imu_pub_, pose_pub_;
};

}  // namespace ariadne::sim

GZ_ADD_PLUGIN(ariadne::sim::KinematicMover, gz::sim::System, gz::sim::ISystemConfigure,
              gz::sim::ISystemPreUpdate)
