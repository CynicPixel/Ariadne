// §30.2 metric registry and the rotational part of ATE (Phase 1, step 1).
#include "runtime/metrics/metric_registry.hpp"
#include "runtime/metrics/trajectory_metrics.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <set>

using namespace ariadne::core;
using namespace ariadne::runtime;

// --- Registry --------------------------------------------------------------

// §30.2 lists 4 + 3 + 6 + 7 + 7 = 27 metrics. If this count changes, §30.2
// changed or an id was dropped; either way the exit gate's meaning changed.
TEST(MetricRegistry, HasExactlyTheSection302MetricsPerGroup) {
    std::map<MetricGroup, int> per_group;
    std::set<std::string_view> ids;
    for (const MetricSpec& m : kCoreMetrics) {
        ++per_group[m.group];
        EXPECT_TRUE(ids.insert(m.id).second) << "duplicate id " << m.id;
    }
    EXPECT_EQ(ids.size(), 27u);
    EXPECT_EQ(per_group[MetricGroup::kLocalization], 4);
    EXPECT_EQ(per_group[MetricGroup::kRegistration], 3);
    EXPECT_EQ(per_group[MetricGroup::kMapping], 6);
    EXPECT_EQ(per_group[MetricGroup::kCommunication], 7);
    EXPECT_EQ(per_group[MetricGroup::kMission], 7);
}

TEST(MetricRegistry, AnEmptyReportIsMissingEverything) {
    EXPECT_EQ(MissingMetrics(MetricReport{}).size(), 27u);
}

TEST(MetricRegistry, ACompleteReportPassesTheGate) {
    MetricReport report;
    for (const MetricSpec& m : kCoreMetrics) report[std::string(m.id)] = 1.0;
    EXPECT_TRUE(MissingMetrics(report).empty());
    EXPECT_TRUE(UnknownMetrics(report).empty());
}

// "Emits a number, however bad": a bad number passes, a non-number does not.
// Without this, a metric stuck at NaN would satisfy a presence check forever.
TEST(MetricRegistry, NonFiniteValuesCountAsMissing) {
    MetricReport report;
    for (const MetricSpec& m : kCoreMetrics) report[std::string(m.id)] = 1e9;  // terrible, but a number
    report["chamfer_m"] = std::numeric_limits<double>::quiet_NaN();
    report["route_validity"] = std::numeric_limits<double>::infinity();
    const auto missing = MissingMetrics(report);
    ASSERT_EQ(missing.size(), 2u);
    EXPECT_EQ(missing[0], "chamfer_m");
    EXPECT_EQ(missing[1], "route_validity");
}

// A typo'd key must be caught, or its value is orphaned AND the real metric
// silently reads as missing.
TEST(MetricRegistry, UnknownIdsAreReported) {
    MetricReport report{{"ate_rmse_m", 0.1}, {"ate_rsme_m", 0.1}};
    const auto unknown = UnknownMetrics(report);
    ASSERT_EQ(unknown.size(), 1u);
    EXPECT_EQ(unknown[0], "ate_rsme_m");
    EXPECT_NE(FindMetric("ate_rmse_m"), nullptr);
}

// --- Rotational ATE --------------------------------------------------------

namespace {
std::vector<TrajectorySample> Gt() {
    TrajectoryParams p;
    return GenerateGroundTruth(p);
}
}  // namespace

// A rigid re-framing of the whole trajectory is exactly what alignment exists
// to remove: both ATE components must be ~0.
TEST(TrajectoryMetrics, RigidReframingGivesZeroAbsoluteError) {
    const auto gt = Gt();
    SE3 frame;
    frame.R = SO3Exp(Vec3(0.3, -0.2, 0.9));
    frame.t = Vec3(5.0, -2.0, 1.0);
    auto est = gt;
    for (auto& s : est) s.pose = frame * s.pose;

    EXPECT_LT(ComputeATE(est, gt).rmse, 1e-9);
    EXPECT_LT(ComputeATERotation(est, gt).rmse, 1e-9);
}

// THE PROPERTY rotational ATE exists for: positions exactly right, every
// orientation wrong by a fixed angle. Translation-only ATE reports 0 and would
// pass this estimator; the rotation part must report exactly that angle.
TEST(TrajectoryMetrics, RotationErrorIsInvisibleToTranslationAtE) {
    const auto gt = Gt();
    const double angle = 0.05;   // rad
    auto est = gt;
    for (auto& s : est) s.pose.R = s.pose.R * SO3Exp(Vec3(0.0, 0.0, angle));

    EXPECT_LT(ComputeATE(est, gt).rmse, 1e-9);
    const ErrorStats rot = ComputeATERotation(est, gt);
    EXPECT_NEAR(rot.rmse, angle, 1e-9);
    EXPECT_NEAR(rot.max, angle, 1e-9);
}

// Both components must use the same alignment. With a rigid re-framing PLUS a
// per-pose orientation error, the re-framing is removed and only the injected
// angle remains.
TEST(TrajectoryMetrics, RotationErrorSurvivesReframing) {
    const auto gt = Gt();
    SE3 frame;
    frame.R = SO3Exp(Vec3(-0.4, 0.1, 0.2));
    frame.t = Vec3(-1.0, 3.0, 0.5);
    const double angle = 0.02;
    auto est = gt;
    for (auto& s : est) {
        s.pose.R = s.pose.R * SO3Exp(Vec3(angle, 0.0, 0.0));
        s.pose = frame * s.pose;
    }
    EXPECT_LT(ComputeATE(est, gt).rmse, 1e-9);
    EXPECT_NEAR(ComputeATERotation(est, gt).rmse, angle, 1e-9);
}

TEST(TrajectoryMetrics, MismatchedLengthsYieldNoNumber) {
    const auto gt = Gt();
    auto est = gt;
    est.pop_back();
    EXPECT_EQ(ComputeATERotation(est, gt).rmse, 0.0);
    EXPECT_FALSE(AlignTrajectories(est, gt, false).valid);
}
