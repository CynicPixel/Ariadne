// MCAP record/replay (arch §5.9), and the Phase 0 gate re-run over it.
//
// The record-file gate already passes; this proves the same property holds
// through the container §5.9 actually specifies, so switching to MCAP does not
// quietly cost determinism.
#include "core/serialization/serializer.hpp"
#include "core/trajectory/synthetic.hpp"
#include "runtime/metrics/trajectory_metrics.hpp"
#include "runtime/replay/mcap_recorder.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace ariadne::core;
using namespace ariadne::runtime;

namespace {

class ScopedBag {
public:
    explicit ScopedBag(const std::string& name)
        : path_("./phase0_mcap_" + name + ".mcap") {
        std::remove(path_.c_str());
    }
    ~ScopedBag() { std::remove(path_.c_str()); }
    const std::string& Get() const { return path_; }

private:
    std::string path_;
};

PoseValue ToWireValue(const SE3& pose) {
    Eigen::Quaterniond q(pose.R);
    q.normalize();
    if (q.w() < 0.0) q.coeffs() *= -1.0;   // q and -q are the same rotation
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

struct BagRun {
    ErrorStats ate;
    RelativeError rpe;
    uint64_t wire_bytes = 0;
    size_t messages_written = 0;
    size_t messages_replayed = 0;
    size_t poses = 0;
    uint64_t pose_digest = 0;
    std::vector<uint64_t> log_times;
};

uint64_t Absorb(uint64_t digest, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 4; ++i) {
        digest ^= (bits >> (i * 8)) & 0xFFull;
        digest *= 1099511628211ull;
    }
    return digest;
}

/// Generate, record to MCAP, replay from MCAP, score.
BagRun RunOverBag(const TrajectoryParams& params, uint64_t seed,
                  const std::string& path) {
    BagRun result;
    const std::vector<TrajectorySample> gt = GenerateGroundTruth(params);
    const std::vector<TrajectorySample> est =
        GenerateNoisyEstimate(gt, params, seed);

    const WireCodec codec;
    ByteLedger ledger;
    {
        McapRecordWriter writer;
        // A real bag carries the schema text; a short stand-in keeps the test
        // independent of the source tree layout.
        const mcap::Status opened = writer.Open(path, "table Envelope {}");
        EXPECT_TRUE(opened.ok()) << opened.message;

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
            const mcap::Status wrote = writer.Write(msg, &ledger);
            EXPECT_TRUE(wrote.ok()) << wrote.message;
            ++result.messages_written;
        }
        writer.Close();
    }
    result.wire_bytes = ledger.Total();

    std::vector<TrajectorySample> replayed;
    {
        McapRecordReader reader;
        const mcap::Status opened = reader.Open(path);
        EXPECT_TRUE(opened.ok()) << opened.message;

        std::vector<std::string> problems;
        result.messages_replayed = reader.ForEach(
            [&](const std::vector<uint8_t>& frame, uint64_t log_time) {
                result.log_times.push_back(log_time);
                DecodedMessage decoded;
                const auto status = codec.Decode(frame, &decoded);
                EXPECT_EQ(status, WireCodec::Status::kOk)
                    << WireCodec::StatusName(status);
                if (!decoded.has_pose_graph) return;
                for (const auto& node : decoded.pose_graph.nodes) {
                    result.pose_digest = Absorb(result.pose_digest, node.pose.qw);
                    result.pose_digest = Absorb(result.pose_digest, node.pose.qx);
                    result.pose_digest = Absorb(result.pose_digest, node.pose.qy);
                    result.pose_digest = Absorb(result.pose_digest, node.pose.qz);
                    result.pose_digest = Absorb(result.pose_digest, node.pose.tx);
                    result.pose_digest = Absorb(result.pose_digest, node.pose.ty);
                    result.pose_digest = Absorb(result.pose_digest, node.pose.tz);
                    TrajectorySample s;
                    s.keyframe = node.keyframe;
                    s.t = Timestamp::Nanoseconds(node.stamp_ns);
                    s.pose = FromWireValue(node.pose);
                    replayed.push_back(s);
                }
            },
            [&problems](const mcap::Status& s) { problems.push_back(s.message); });
        EXPECT_TRUE(problems.empty())
            << "bag reported " << problems.size() << " problem(s): "
            << (problems.empty() ? "" : problems.front());
        reader.Close();
    }
    result.poses = replayed.size();
    result.ate = ComputeATE(replayed, gt);
    result.rpe = ComputeRPE(replayed, gt, 1);
    return result;
}

}  // namespace

// --- The gate, over MCAP --------------------------------------------------

TEST(McapReplay, GateHoldsOverTheMcapContainer) {
    TrajectoryParams params;
    params.keyframes = 200;
    ScopedBag bag_a("run_a"), bag_b("run_b");

    const BagRun a = RunOverBag(params, 20260906ull, bag_a.Get());
    const BagRun b = RunOverBag(params, 20260906ull, bag_b.Get());

    ASSERT_EQ(a.poses, static_cast<size_t>(params.keyframes));
    ASSERT_EQ(a.messages_replayed, a.messages_written);

    // Exact equality, as in the record-file gate: a tolerance would hide the
    // drift this exists to detect.
    EXPECT_EQ(a.pose_digest, b.pose_digest);
    EXPECT_EQ(a.wire_bytes, b.wire_bytes);
    EXPECT_EQ(a.ate.rmse, b.ate.rmse);
    EXPECT_EQ(a.rpe.translation.rmse, b.rpe.translation.rmse);
    EXPECT_EQ(a.rpe.rotation.rmse, b.rpe.rotation.rmse);

    std::cout << "[ MCAP     ] poses=" << a.poses
              << " messages=" << a.messages_replayed
              << " wire_bytes=" << a.wire_bytes << "\n"
              << "[ MCAP     ] ATE rmse=" << a.ate.rmse
              << " m, RPE trans rmse=" << a.rpe.translation.rmse << " m\n";
}

// The container must not perturb the payload: what comes out is what went in,
// byte for byte, or replay would not exercise the production decode path.
TEST(McapReplay, FramesSurviveTheContainerByteForByte) {
    ScopedBag bag("bytes");
    const WireCodec codec;

    std::vector<std::vector<uint8_t>> written;
    {
        McapRecordWriter writer;
        ASSERT_TRUE(writer.Open(bag.Get(), "table Envelope {}").ok());
        for (int i = 0; i < 12; ++i) {
            LoopEdgeValue edge;
            edge.agent_a = 0;
            edge.keyframe_a = static_cast<uint64_t>(i);
            edge.agent_b = 1;
            edge.keyframe_b = static_cast<uint64_t>(i + 100);
            edge.frame_a = FrameTag{0, 1};
            edge.frame_b = FrameTag{1, 1};
            edge.inlier_count = static_cast<uint32_t>(i * 7 + 3);
            edge.retrieval_score = 0.5f + 0.01f * i;
            for (int k = 0; k < 21; ++k) edge.information[k] = 0.25f * k;
            SerializedMessage msg = codec.EncodeLoopEdge(
                edge, 0, 1, Timestamp::Nanoseconds(1000 * (i + 1)),
                static_cast<uint64_t>(i));
            written.push_back(msg.bytes);
            ASSERT_TRUE(writer.Write(msg, nullptr).ok());
        }
        writer.Close();
    }

    std::vector<std::vector<uint8_t>> read_back;
    {
        McapRecordReader reader;
        ASSERT_TRUE(reader.Open(bag.Get()).ok());
        reader.ForEach([&](const std::vector<uint8_t>& frame, uint64_t) {
            read_back.push_back(frame);
        });
    }
    ASSERT_EQ(read_back.size(), written.size());
    for (size_t i = 0; i < written.size(); ++i) {
        EXPECT_EQ(read_back[i], written[i]) << "frame " << i;
    }
}

// Replay order must be log-time order. The estimator consumes a stream, and a
// reordered stream is a different experiment.
TEST(McapReplay, MessagesReplayInLogTimeOrder) {
    TrajectoryParams params;
    params.keyframes = 150;
    ScopedBag bag("order");
    const BagRun r = RunOverBag(params, 5ull, bag.Get());

    ASSERT_GT(r.log_times.size(), 1u);
    for (size_t i = 1; i < r.log_times.size(); ++i) {
        EXPECT_LE(r.log_times[i - 1], r.log_times[i])
            << "log times out of order at " << i;
    }
}

// Log time must come from the message, never a wall clock. A wall clock would
// make every bag unique and destroy the reproducibility §5.9 exists to provide.
TEST(McapReplay, LogTimesComeFromMessageStampsNotWallClock) {
    ScopedBag bag("stamps");
    const WireCodec codec;
    const std::vector<int64_t> stamps = {5000, 15000, 25000};
    {
        McapRecordWriter writer;
        ASSERT_TRUE(writer.Open(bag.Get(), "table Envelope {}").ok());
        for (size_t i = 0; i < stamps.size(); ++i) {
            PoseGraphValue graph;
            graph.anchor = FrameTag{0, 1};
            PoseNodeValue node;
            node.frame = FrameTag{0, 1};
            node.stamp_ns = stamps[i];
            graph.nodes.push_back(node);
            SerializedMessage msg = codec.EncodePoseGraph(
                graph, 0, 1, Timestamp::Nanoseconds(stamps[i]), i);
            ASSERT_TRUE(writer.Write(msg, nullptr).ok());
        }
        writer.Close();
    }

    std::vector<uint64_t> times;
    McapRecordReader reader;
    ASSERT_TRUE(reader.Open(bag.Get()).ok());
    reader.ForEach([&](const std::vector<uint8_t>&, uint64_t t) {
        times.push_back(t);
    });
    ASSERT_EQ(times.size(), stamps.size());
    for (size_t i = 0; i < stamps.size(); ++i) {
        EXPECT_EQ(times[i], static_cast<uint64_t>(stamps[i]));
    }
}

// A bag written twice from identical inputs should be byte-identical. This is
// stronger than the gate requires (which only demands identical METRICS), and
// it is asserted rather than assumed because it is the property that lets a bag
// checksum appear in a run manifest (§5.9, §28.2).
TEST(McapReplay, IdenticalInputsProduceAByteIdenticalBag) {
    TrajectoryParams params;
    params.keyframes = 100;
    ScopedBag bag_a("det_a"), bag_b("det_b");
    RunOverBag(params, 77ull, bag_a.Get());
    RunOverBag(params, 77ull, bag_b.Get());

    auto slurp = [](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(in),
                                 std::istreambuf_iterator<char>());
    };
    const std::vector<char> a = slurp(bag_a.Get());
    const std::vector<char> b = slurp(bag_b.Get());
    ASSERT_FALSE(a.empty());
    EXPECT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b) << "bag bytes differ between identical runs; a bag checksum "
                       "in a run manifest would then be meaningless";
}

TEST(McapReplay, OpeningAMissingOrCorruptBagFails) {
    McapRecordReader reader;
    EXPECT_FALSE(reader.Open("./definitely_not_a_bag.mcap").ok());

    // A file that exists but is not MCAP must be rejected, not silently
    // treated as empty.
    const std::string path = "./not_really_a_bag.mcap";
    {
        std::ofstream out(path, std::ios::binary);
        out << "this is not an MCAP file";
    }
    McapRecordReader bad;
    EXPECT_FALSE(bad.Open(path).ok());
    std::remove(path.c_str());
}
