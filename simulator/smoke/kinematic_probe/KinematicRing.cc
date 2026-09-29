// Throwaway smoke-test system: writes components::Pose of one model every
// PreUpdate from an analytic trajectory, with NO Physics system loaded.
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Model.hh>
#include <gz/math/Pose3.hh>
#include <chrono>
#include <cmath>
#include <string>

using namespace gz;
using namespace sim;

class KinematicRing : public System, public ISystemConfigure, public ISystemPreUpdate {
 public:
  void Configure(const Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 EntityComponentManager &, EventManager &) override {
    name_ = sdf->Get<std::string>("model_name"); y0_ = sdf->Get<double>("y0"); speed_ = sdf->Get<double>("speed");
  }
  void PreUpdate(const UpdateInfo &info, EntityComponentManager &ecm) override {
    if (model_ == kNullEntity) {
      model_ = ecm.EntityByComponents(components::Name(name_), components::Model());
      if (model_ == kNullEntity) return;
    }
    const double t = std::chrono::duration<double>(info.simTime).count();
    // Phase A: toward the wall along +x at 5 m/s; phase B: lateral along +y at 5 m/s.
    // Circle of radius 1.5 m around room centre, yaw tangent.
    const double th = speed_ * t / 1.5;
    math::Pose3d p(1.5 * std::cos(th), y0_ + 1.5 * std::sin(th), 1.0, 0, 0, th + 1.5707963);
    auto *c = ecm.Component<components::Pose>(model_);
    if (!c) { ecm.CreateComponent(model_, components::Pose(p)); return; }
    *c = components::Pose(p);
    ecm.SetChanged(model_, components::Pose::typeId, ComponentState::PeriodicChange);
  }
 private:
  std::string name_; double y0_ = 0, speed_ = 0.5;
  Entity model_ = kNullEntity;
};
GZ_ADD_PLUGIN(KinematicRing, System, ISystemConfigure, ISystemPreUpdate)
