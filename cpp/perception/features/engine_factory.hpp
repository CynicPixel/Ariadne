#pragma once
// The §7.1 engine named by features.engine (perception.md §4.1): a config change swaps it everywhere the runtime
// takes a LocalFeatureEngine (tracker replenish, keyframe description, re-association). ORB stays the default and the
// ablation baseline.
#include "perception/features/feature_engine.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/features/xfeat_engine.hpp"
#include "perception/perception_config.hpp"

#include <memory>

namespace ariadne::perception {

inline std::unique_ptr<LocalFeatureEngine> MakeFeatureEngine(const FeatureParams& p) {
    if (p.engine == "xfeat") return std::make_unique<XFeatEngine>(p);
    return std::make_unique<OrbEngine>(p);
}

}  // namespace ariadne::perception
