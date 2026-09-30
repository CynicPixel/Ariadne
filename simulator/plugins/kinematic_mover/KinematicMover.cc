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
// ---------------------------------------------------------------------------
#include "kinematic_mover/MoverScript.hh"

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/msgs/imu.pb.h>
#include <gz/msgs/pose.pb.h>
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
#include <fstream>
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
        gzmsg << "[KinematicMover] " << cfg_.frame_id << ": " << cfg_.script.segments().size() << " segments, "
              << cfg_.script.duration() << " s, envelope ok\n";
    }

    void PreUpdate(const gz::sim::UpdateInfo& info, gz::sim::EntityComponentManager& ecm) override {
        if (info.paused) return;
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
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
        Esdf e;
        e.rows = el->Get<int>("rows");
        e.cols = el->Get<int>("cols");
        e.ox = el->Get<int>("origin_x_cells");
        e.oy = el->Get<int>("origin_y_cells");
        e.cell = el->Get<double>("cell_m");
        const double rho = RequiredElem<double>(sdf, "rho_agent");
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

    gz::sim::Entity model_{gz::sim::kNullEntity};
    MoverConfig cfg_;
    int64_t last_pub_ns_ = -1;
    gz::transport::Node node_;
    gz::transport::Node::Publisher imu_pub_, pose_pub_;
};

}  // namespace ariadne::sim

GZ_ADD_PLUGIN(ariadne::sim::KinematicMover, gz::sim::System, gz::sim::ISystemConfigure,
              gz::sim::ISystemPreUpdate)
