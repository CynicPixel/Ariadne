// THE PHASE 0 EXIT GATE.
//
// From arch §2:
//   "a synthetic trajectory is generated, serialized, replayed, and scored
//    bit-identically twice"
// and from arch §33 Iteration 0:
//   "A synthetic trajectory can be generated, serialized, replayed and scored
//    exactly. The same bag plus the same seed produces bit-identical metrics."
//
// WHY "TWICE" IS THE WHOLE POINT. Every comparison claim in every later phase
// assumes that re-running a configuration reproduces it. If scoring drifts
// between runs, each ablation delta carries run-to-run noise of unknown size,
// and no amount of later care recovers the difference between a real effect and
// that noise. This test is the guarantee, and it runs the FULL loop rather than
// its components, because the components already pass individually.
#include "core/serialization/serializer.hpp"
#include "core/trajectory/synthetic.hpp"
#include "runtime/metrics/trajectory_metrics.hpp"
#include "runtime/replay/record_file.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace ariadne::core;
using namespace ariadne::runtime;

namespace {

/// A temporary record file that removes itself, so a failing run cannot leave
/// state that makes the next run pass.
class ScopedRecordPath {
public:
    explicit ScopedRecordPath(const std::string& name)
        : path_("./phase0_gate_" + name + ".arec") {
        std::remove(path_.c_str());
    }
    ~ScopedRecordPath() { std::remove(path_.c_str()); }
    const std::string& Get() const { return path_; }

private:
    std::string path_;
};

PoseValue ToWireValue(const SE3& pose) {
    Eigen::Quaterniond q(pose.R);
    q.normalize();
    // Fix the sign convention. q and -q are the same rotation, and Eigen may
    // return either; without this the serialized bytes could differ between
    // runs for an identical pose, which would fail the gate for a reason that
    // has nothing to do with the estimator.
    if (q.w() < 0.0) q.coeffs() *= -1.0;
    PoseValue v;
    v.qw = static_cast<float>(q.w());
    v.qx = static_cast<float>(q.x());
    v.qy = static_cast<float>(q.y());
    v.qz = static_cast<float>(q.z());
    v.tx = static_cast<float>(pose.t.x());
    v.ty = static_cast<float>(pose.t.y());
    v.tz = static_cast<float>(pose.t.z());
    return v;
}

SE3 FromWireValue(const PoseValue& v) {
    SE3 pose;
    Eigen::Quaterniond q(v.qw, v.qx, v.qy, v.qz);
    q.normalize();
    pose.R = q.toRotationMatrix();
    pose.t = Vec3(v.tx, v.ty, v.tz);
    return pose;
}

/// Everything one run of the pipeline produces.
struct RunResult {
    ErrorStats ate;
    RelativeError rpe;
    uint64_t total_wire_bytes = 0;
    uint64_t message_count = 0;
    size_t replayed_poses = 0;
    /// Digest over the replayed poses, from raw float bits, so a one-ulp change
    /// cannot slip past a tolerance-based comparison.
    uint64_t pose_digest = 0;
};

uint64_t AbsorbFloat(uint64_t digest, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 4; ++i) {
        digest ^= (bits >> (i * 8)) & 0xFFull;
        digest *= 1099511628211ull;
    }
    return digest;
}

/// The full loop: generate, serialize to a record file, replay it back, score
/// the replayed estimate against ground truth.
RunResult RunPipeline(const TrajectoryParams& params, uint64_t seed,
                      const std::string& path) {
    RunResult result;

    // --- Generate ---------------------------------------------------------
    const std::vector<TrajectorySample> gt = GenerateGroundTruth(params);
    const std::vector<TrajectorySample> est =
        GenerateNoisyEstimate(gt, params, seed);

    // --- Serialize --------------------------------------------------------
    // Batched into pose-graph messages, which is how a real agent ships its
    // graph (§15.2), so the recording exercises the production wire path.
    const WireCodec codec;
    ByteLedger ledger;
    {
        RecordWriter writer(path);
        EXPECT_TRUE(writer.Ok());

        constexpr size_t kBatch = 50;
        uint64_t sequence = 0;
        for (size_t start = 0; start < est.size(); start += kBatch) {
            PoseGraphValue graph;
            graph.anchor = FrameTag{0, 1};
            const size_t stop = std::min(start + kBatch, est.size());
            for (size_t i = start; i < stop; ++i) {
                PoseNodeValue node;
                node.agent = 0;
                node.keyframe = est[i].keyframe;
                node.stamp_ns = est[i].t.ns;
                node.pose = ToWireValue(est[i].pose);
                node.frame = FrameTag{0, 1};
                graph.nodes.push_back(node);
            }
            SerializedMessage msg = codec.EncodePoseGraph(
                graph, 0, 1, Timestamp::Nanoseconds(est[start].t.ns), sequence++);
            EXPECT_TRUE(writer.Write(msg, &ledger));
        }
        writer.Close();
    }
    result.total_wire_bytes = ledger.Total();
    result.message_count = ledger.Records().size();

    // --- Replay -----------------------------------------------------------
    std::vector<TrajectorySample> replayed;
    {
        RecordReader reader(path);
        EXPECT_TRUE(reader.Ok());
        std::vector<uint8_t> frame;
        bool truncated = false;
        while (reader.Next(&frame, &truncated)) {
            DecodedMessage decoded;
            const auto status = codec.Decode(frame, &decoded);
            EXPECT_EQ(status, WireCodec::Status::kOk)
                << WireCodec::StatusName(status);
            // EXPECT rather than ASSERT: this helper returns a value, and
            // ASSERT_* expands to a bare `return;`.
            EXPECT_TRUE(decoded.has_pose_graph);
            if (!decoded.has_pose_graph) continue;
            for (const auto& node : decoded.pose_graph.nodes) {
                result.pose_digest = AbsorbFloat(result.pose_digest, node.pose.qw);
                result.pose_digest = AbsorbFloat(result.pose_digest, node.pose.qx);
                result.pose_digest = AbsorbFloat(result.pose_digest, node.pose.qy);
                result.pose_digest = AbsorbFloat(result.pose_digest, node.pose.qz);
                result.pose_digest = AbsorbFloat(result.pose_digest, node.pose.tx);
                result.pose_digest = AbsorbFloat(result.pose_digest, node.pose.ty);
                result.pose_digest = AbsorbFloat(result.pose_digest, node.pose.tz);

                TrajectorySample s;
                s.keyframe = node.keyframe;
                s.t = Timestamp::Nanoseconds(node.stamp_ns);
                s.pose = FromWireValue(node.pose);
                replayed.push_back(s);
            }
        }
        EXPECT_FALSE(truncated) << "record file was truncated";
    }
    result.replayed_poses = replayed.size();

    // --- Score ------------------------------------------------------------
    // Scoring runs against the REPLAYED trajectory, not the in-memory one, so
    // the metrics measure what a downstream consumer would actually receive.
    result.ate = ComputeATE(replayed, gt);
    result.rpe = ComputeRPE(replayed, gt, 1);
    return result;
}

}  // namespace

// --- THE GATE -------------------------------------------------------------

TEST(Phase0Gate, GenerateSerializeReplayScoreIsBitIdenticalTwice) {
    TrajectoryParams params;
    params.keyframes = 200;

    ScopedRecordPath path_a("run_a");
    ScopedRecordPath path_b("run_b");

    const RunResult a = RunPipeline(params, 20260906ull, path_a.Get());
    const RunResult b = RunPipeline(params, 20260906ull, path_b.Get());

    ASSERT_EQ(a.replayed_poses, static_cast<size_t>(params.keyframes));
    ASSERT_GT(a.message_count, 0u);

    // Bit-identical, not merely close. EXPECT_EQ on doubles is deliberate here:
    // a tolerance would hide exactly the drift this gate exists to detect.
    EXPECT_EQ(a.pose_digest, b.pose_digest);
    EXPECT_EQ(a.total_wire_bytes, b.total_wire_bytes);
    EXPECT_EQ(a.message_count, b.message_count);
    EXPECT_EQ(a.ate.rmse, b.ate.rmse);
    EXPECT_EQ(a.ate.mean, b.ate.mean);
    EXPECT_EQ(a.ate.median, b.ate.median);
    EXPECT_EQ(a.ate.max, b.ate.max);
    EXPECT_EQ(a.rpe.translation.rmse, b.rpe.translation.rmse);
    EXPECT_EQ(a.rpe.rotation.rmse, b.rpe.rotation.rmse);

    std::cout << "[ GATE     ] poses=" << a.replayed_poses
              << " messages=" << a.message_count
              << " wire_bytes=" << a.total_wire_bytes << "\n"
              << "[ GATE     ] ATE rmse=" << a.ate.rmse
              << " m, max=" << a.ate.max << " m\n"
              << "[ GATE     ] RPE trans rmse=" << a.rpe.translation.rmse
              << " m, rot rmse=" << a.rpe.rotation.rmse << " rad\n";
}

// A gate that cannot fail proves nothing. Changing the seed must change the
// noise realization and therefore the metrics.
TEST(Phase0Gate, ADifferentSeedProducesDifferentMetrics) {
    TrajectoryParams params;
    params.keyframes = 120;
    ScopedRecordPath pa("seed_a"), pb("seed_b");

    const RunResult a = RunPipeline(params, 1ull, pa.Get());
    const RunResult b = RunPipeline(params, 2ull, pb.Get());

    EXPECT_NE(a.pose_digest, b.pose_digest);
    EXPECT_NE(a.ate.rmse, b.ate.rmse);
}

// The recording must survive the round trip exactly. float32 in, float32 out.
TEST(Phase0Gate, ReplayReconstructsTheRecordedTrajectoryExactly) {
    TrajectoryParams params;
    params.keyframes = 64;
    params.translation_noise_m = 0.0;
    params.rotation_noise_rad = 0.0;

    ScopedRecordPath path("exact");
    const RunResult r = RunPipeline(params, 7ull, path.Get());

    // With zero odometry noise the estimate equals ground truth, so the only
    // error left is float32 quantization at the serialization boundary.
    EXPECT_EQ(r.replayed_poses, 64u);
    EXPECT_LT(r.ate.rmse, 1e-5) << "float32 round-trip lost more than expected";
    EXPECT_LT(r.rpe.translation.rmse, 1e-5);
}

// Noise must actually show up, otherwise the metrics are not measuring.
TEST(Phase0Gate, OdometryNoiseAccumulatesIntoDrift) {
    TrajectoryParams params;
    params.keyframes = 200;
    params.translation_noise_m = 0.02;
    params.rotation_noise_rad = 0.004;

    ScopedRecordPath path("drift");
    const RunResult r = RunPipeline(params, 3ull, path.Get());

    EXPECT_GT(r.ate.rmse, 1e-3);
    // ATE grows along the chain while RPE stays local, which is precisely why
    // §30.2 reports both: a locally good estimator can still drift globally.
    EXPECT_GT(r.ate.max, r.rpe.translation.rmse);
}

// --- Alignment -----------------------------------------------------------

TEST(Phase0Gate, UmeyamaRecoversAKnownRigidTransform) {
    TrajectoryParams params;
    params.keyframes = 80;
    const auto gt = GenerateGroundTruth(params);

    // Displace the whole trajectory by a known transform.
    Vec6 xi;
    xi << 3.0, -1.5, 0.75, 0.2, -0.1, 0.35;
    const SE3 applied = SE3Exp(xi);

    std::vector<Vec3> src, dst;
    for (const auto& s : gt) {
        dst.push_back(s.pose.t);
        src.push_back(applied.inverse() * s.pose.t);
    }
    const AlignmentResult a = UmeyamaAlign(src, dst);
    ASSERT_TRUE(a.valid);
    EXPECT_LT((a.transform.R - applied.R).norm(), 1e-9);
    EXPECT_LT((a.transform.t - applied.t).norm(), 1e-9);
    EXPECT_EQ(a.scale, 1.0);   // SE(3): scale is not estimated
}

// The determinant fix. An unconstrained SVD can return a REFLECTION, which fits
// mirrored points better than any rotation but is not a physical motion. A
// mirrored trajectory must therefore NOT align well.
TEST(Phase0Gate, AlignmentRejectsReflections) {
    TrajectoryParams params;
    params.keyframes = 80;
    const auto gt = GenerateGroundTruth(params);

    std::vector<Vec3> src, dst;
    for (const auto& s : gt) {
        dst.push_back(s.pose.t);
        Vec3 mirrored = s.pose.t;
        mirrored.x() = -mirrored.x();       // improper transform
        src.push_back(mirrored);
    }
    const AlignmentResult a = UmeyamaAlign(src, dst);
    ASSERT_TRUE(a.valid);
    // The result must be a proper rotation.
    EXPECT_NEAR(a.transform.R.determinant(), 1.0, 1e-9);

    double worst = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        const Vec3 mapped = a.transform.R * src[i] + a.transform.t;
        worst = std::max(worst, (mapped - dst[i]).norm());
    }
    // A reflection would fit this near-perfectly; a proper rotation cannot.
    EXPECT_GT(worst, 1.0) << "a reflection was accepted as an alignment";
}

TEST(Phase0Gate, ScaleAlignmentRecoversAKnownScale) {
    TrajectoryParams params;
    params.keyframes = 80;
    const auto gt = GenerateGroundTruth(params);

    std::vector<Vec3> src, dst;
    for (const auto& s : gt) {
        dst.push_back(s.pose.t);
        src.push_back(s.pose.t / 2.5);      // scaled, as in a monocular estimate
    }
    const AlignmentResult a = UmeyamaAlign(src, dst, /*with_scale=*/true);
    ASSERT_TRUE(a.valid);
    EXPECT_NEAR(a.scale, 2.5, 1e-9);
}

// --- Record file robustness ----------------------------------------------

TEST(Phase0Gate, TruncatedRecordFileIsDetectedNotSilentlyAccepted) {
    TrajectoryParams params;
    params.keyframes = 100;
    ScopedRecordPath path("trunc");

    RunPipeline(params, 5ull, path.Get());

    // Chop the tail off the file.
    std::vector<uint8_t> all;
    {
        std::ifstream in(path.Get(), std::ios::binary);
        all.assign(std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>());
    }
    ASSERT_GT(all.size(), 64u);
    {
        std::ofstream out(path.Get(), std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(all.data()), all.size() - 17);
    }

    RecordReader reader(path.Get());
    std::vector<uint8_t> frame;
    bool truncated = false;
    while (reader.Next(&frame, &truncated)) {}
    // A partial trailing record is a corrupt bag, not an empty one, and must be
    // reported rather than passed off as a clean end of file.
    EXPECT_TRUE(truncated);
}
