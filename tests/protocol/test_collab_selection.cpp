// 3.4 selection in the protocol (design spectral-selection.md §5-§6, D170), seeded discrete-event runs on the sim clock:
//   - the token bucket caps the fetch bytes actually sent (first transmission, as the ledger counts them);
//   - the byte prediction meets the I6 gate (<= 15% per class) on a scene DISJOINT from the one its compression ratio
//     was calibrated on (collaboration.md §7);
//   - a run under selection still ends consistent.
// The policy here is SelectBySimilarity (OpenCV-free); the spectral policy in the swarm is in test_opencv.
#include "collab_scenario.hpp"

#include <gtest/gtest.h>

#include <cstdio>

namespace act = ariadne::collaboration::testing;
namespace ac = ariadne::collaboration;
using ariadne::core::MessageClass;

namespace {

struct ClassBytes {
    double payload = 0, transmitted = 0, wire = 0;
};
ClassBytes FirstTransmission(const ariadne::core::ByteLedger& l, MessageClass cls) {
    ClassBytes b;
    for (const auto& r : l.Records())
        if (r.type == cls && !r.retransmit) b.payload += r.payload_bytes, b.transmitted += r.transmitted_bytes,
                                                b.wire += r.wire_bytes;
    return b;
}

/// ratio <= 0: uncalibrated (priced at 1.0, an upper bound).
ac::NodeParams Budgeted(double rate, double burst, double ratio, const char* run = "scene A, seed 1") {
    ac::NodeParams p;
    p.budget_bytes_per_s = rate;
    p.budget_burst_bytes = burst;
    if (ratio > 0) p.response_cost.response_ratio[32] = ratio, p.response_cost.calibration_run = run;
    return p;
}

void UseGreedy(ac::NodeHooks& h) {
    h.similarity = [](const ac::KeyframeRef& q, const ac::KeyframeRef& c) {   // deterministic, in (0.3, 1]
        return 0.3 + 0.7 * double((q.keyframe * 7919 + c.keyframe * 104729 + q.agent * 31 + c.agent) % 1000) / 999.0;
    };
    h.select = [](const ac::SelectionProblem& P) { return ac::SelectBySimilarity(P); };
}

/// The calibration of the gate test, reused: scene A (200 keypoints, 4 agents, seed 1), unbudgeted.
double CalibrateRatio() {
    act::Scenario a;
    a.advertise_features = true;
    act::Swarm calib(a, {true, 0.02, 0.0, 0.0}, 1, {}, Budgeted(1e9, 1e9, 0), UseGreedy);
    calib.RunUntil(0, 90);
    const ClassBytes ca = FirstTransmission(calib.ledger, MessageClass::MATCH_RESPONSE);
    return ca.transmitted / ca.payload;
}

}  // namespace

TEST(CollabSelection, TokenBucketCapsTheFetchBytesSent) {
    // Calibrated pricing (scene A), run on scene B: with an uncalibrated ratio the bucket over-prices ~5x and the cap
    // is trivially met (advisor, 2026-10-23).
    const double ratio = CalibrateRatio();
    act::Scenario sc;
    sc.advertise_features = true;
    sc.keypoints = 300;
    sc.agents = 3;
    const double rate = 4000, burst = 20000, T = 90;
    act::Swarm s(sc, {true, 0.02, 0.0, 0.0}, 2, {}, Budgeted(rate, burst, ratio), UseGreedy);
    s.RunUntil(0, T);
    const double sent = FirstTransmission(s.ledger, MessageClass::MATCH_RESPONSE).wire +
                        FirstTransmission(s.ledger, MessageClass::MATCH_REQUEST).wire;
    const auto& st = s.nodes[0]->stats();
    std::printf("bucket: %.0f B sent <= %.0f B allowed; %llu of %llu candidates selected, %llu rounds\n", sent,
                rate * T + burst, (unsigned long long)st.candidates_selected, (unsigned long long)st.candidates,
                (unsigned long long)st.selection_rounds);
    EXPECT_LE(sent, (rate * T + burst) * 1.15) << "within the gate's 15% of the calibrated prediction";
    EXPECT_GT(sent, 0.5 * (rate * T + burst)) << "a calibrated bucket should be spent, not idle";
    EXPECT_EQ(st.uncalibrated_prices, 0u);
    EXPECT_EQ(st.unadvertised_prices, 0u);
    EXPECT_GT(st.selection_rounds, 0u);
    EXPECT_GT(st.candidates_selected, 0u);
    EXPECT_LT(st.candidates_selected, st.candidates) << "the budget should bind in this scene";
    EXPECT_EQ(s.Outcome().outcome, ac::Outcome::kConsistent);
}

TEST(CollabSelection, PredictionMeetsTheByteGateOnADisjointScene) {
    // Calibrate the response compression ratio on scene A (200 keypoints, 4 agents) ...
    const double ratio = CalibrateRatio();
    ASSERT_GT(ratio, 0.0);
    // ... and gate on scene B (300 keypoints, 3 agents, another seed), never seen by the calibration.
    act::Scenario b;
    b.advertise_features = true;
    b.keypoints = 300;
    b.agents = 3;
    act::Swarm s(b, {true, 0.02, 0.0, 0.0}, 2, {}, Budgeted(1e9, 1e9, ratio), UseGreedy);
    s.RunUntil(0, 90);   // keyframes stop at 60 s; 30 s drains the last rounds' responses (else predicted > sent)
    const auto& st = s.nodes[0]->stats();
    const double resp = FirstTransmission(s.ledger, MessageClass::MATCH_RESPONSE).wire;
    const double req = FirstTransmission(s.ledger, MessageClass::MATCH_REQUEST).wire;
    const double e_resp = std::fabs(resp - st.predicted_response_bytes) / resp;
    const double e_req = std::fabs(req - st.predicted_request_bytes) / req;
    std::printf("byte gate: ratio %.4f (scene A); scene B response %.0f measured / %.0f predicted (%.2f%%), request "
                "%.0f / %.0f (%.2f%%)\n", ratio, resp, st.predicted_response_bytes, 100 * e_resp, req,
                st.predicted_request_bytes, 100 * e_req);
    ASSERT_GT(resp, 0);
    EXPECT_LE(e_resp, 0.15);
    EXPECT_LE(e_req, 0.15);
}

// D170: the appended GlobalDescriptor fields must leave v1 messages byte-identical (FlatBuffers omits default-valued
// fields). Golden from the pre-change serializer at 5bc95a9 (same values, encoded by that tree): 1142 B, FNV below.
TEST(CollabSelection, V1DescriptorBytesUnchangedAndNewFieldsRoundTrip) {
    using namespace ariadne::core;
    const WireCodec codec(Framing::kWholeMessage);
    GlobalDescriptorValue g;
    g.keyframe_id = 4242;
    g.stamp_ns = 1234567890123;
    g.quality = 0.75f;
    for (int i = 0; i < 256; ++i) g.descriptor.push_back(float(i % 17) * 0.125f - 1.0f);
    auto fnv = [](const std::vector<uint8_t>& b) {
        uint64_t h = 1469598103934665603ull;
        for (uint8_t x : b) h ^= x, h *= 1099511628211ull;
        return h;
    };
    const auto v1 = codec.EncodeGlobalDescriptor(g, 3, 0, Timestamp{987654321}, 77);
    EXPECT_EQ(v1.bytes.size(), 1142u);
    EXPECT_EQ(fnv(v1.bytes), 0x653ebb54ae4fd40bull) << "v1 descriptor bytes changed";
    g.n_features = 800;
    g.feature_bytes = 32;
    const auto v2 = codec.EncodeGlobalDescriptor(g, 3, 0, Timestamp{987654321}, 77);
    EXPECT_GT(v2.bytes.size(), v1.bytes.size());
    DecodedMessage d;
    ASSERT_EQ(codec.Decode(v2.bytes, &d), WireCodec::Status::kOk);
    EXPECT_EQ(d.global_descriptor.n_features, 800u);
    EXPECT_EQ(d.global_descriptor.feature_bytes, 32u);

    // D171: fp16 on the wire (arch §15.7): half the descriptor bytes, values = the fp16 rounding of the input.
    g.fp16 = true;
    const auto h = codec.EncodeGlobalDescriptor(g, 3, 0, Timestamp{987654321}, 77);
    EXPECT_NEAR(double(v2.bytes.size()) - double(h.bytes.size()), 512.0, 16.0) << "256 x 2 B saved, give or take padding";
    ASSERT_EQ(codec.Decode(h.bytes, &d), WireCodec::Status::kOk);
    ASSERT_EQ(d.global_descriptor.descriptor.size(), 256u);
    EXPECT_TRUE(d.global_descriptor.fp16);
    for (int i = 0; i < 256; ++i)
        EXPECT_EQ(d.global_descriptor.descriptor[size_t(i)], float(Eigen::half(g.descriptor[size_t(i)]))) << i;
}

TEST(CollabSelection, GraphStartsAtTheFirstReleasedKeyframe) {
    // A VO startup prefix is never released (collaboration.md#simulated-collaboration §1), so ids start above 0. The selection graph spans the
    // ids seen (min..max per agent), not 0..max: no phantom chain for keyframes no member ever had.
    act::Scenario sc;
    sc.advertise_features = true;
    sc.agents = 2;
    sc.first_keyframe = 20;
    size_t max_nodes = 0, rounds = 0;
    act::Swarm s(sc, {true, 0.02, 0.0, 0.0}, 3, {}, Budgeted(4000, 20000, 0), [&](ac::NodeHooks& h) {
        UseGreedy(h);
        h.select = [&](const ac::SelectionProblem& P) {
            max_nodes = std::max(max_nodes, size_t(P.n)), ++rounds;
            return ac::SelectBySimilarity(P);
        };
    });
    s.RunUntil(0, 90);
    const size_t released = size_t(sc.keyframes_until_s / sc.keyframe_dt_s) + 1 - sc.first_keyframe;
    ASSERT_GT(rounds, 0u);
    EXPECT_LE(max_nodes, sc.agents * released) << "nodes only for ids that were released";
    EXPECT_GT(max_nodes, 0u);
    EXPECT_EQ(s.Outcome().outcome, ac::Outcome::kConsistent);
}
