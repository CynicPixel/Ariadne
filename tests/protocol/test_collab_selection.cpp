// 3.4 selection in the protocol (design SPECTRAL_3.4 §5-§6, D170), seeded discrete-event runs on the sim clock:
//   - the token bucket caps the fetch bytes actually sent (first transmission, as the ledger counts them);
//   - the byte prediction meets the I6 gate (<= 15% per class) on a scene DISJOINT from the one its compression ratio
//     was calibrated on (COLLAB_3.2 §7);
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

ac::NodeParams Budgeted(double rate, double burst, double ratio = 1.0) {
    ac::NodeParams p;
    p.budget_bytes_per_s = rate;
    p.budget_burst_bytes = burst;
    p.response_cost.response_ratio = ratio;
    return p;
}

void UseGreedy(ac::NodeHooks& h) { h.select = [](const ac::SelectionProblem& P) { return ac::SelectBySimilarity(P); }; }

}  // namespace

TEST(CollabSelection, TokenBucketCapsTheFetchBytesSent) {
    act::Scenario sc;
    sc.advertise_features = true;
    const double rate = 4000, burst = 20000, T = 90;
    act::Swarm s(sc, {true, 0.02, 0.0, 0.0}, 1, {}, Budgeted(rate, burst), UseGreedy);
    s.RunUntil(0, T);
    const double sent = FirstTransmission(s.ledger, MessageClass::MATCH_RESPONSE).wire +
                        FirstTransmission(s.ledger, MessageClass::MATCH_REQUEST).wire;
    const auto& st = s.nodes[0]->stats();
    std::printf("bucket: %.0f B sent <= %.0f B allowed; %llu of %llu candidates selected, %llu rounds\n", sent,
                rate * T + burst, (unsigned long long)st.candidates_selected, (unsigned long long)st.candidates,
                (unsigned long long)st.selection_rounds);
    EXPECT_LE(sent, rate * T + burst);
    EXPECT_GT(st.selection_rounds, 0u);
    EXPECT_GT(st.candidates_selected, 0u);
    EXPECT_LT(st.candidates_selected, st.candidates) << "the budget should bind in this scene";
    EXPECT_EQ(s.Outcome().outcome, ac::Outcome::kConsistent);
}

TEST(CollabSelection, PredictionMeetsTheByteGateOnADisjointScene) {
    // Calibrate the response compression ratio on scene A (200 keypoints, 4 agents) ...
    act::Scenario a;
    a.advertise_features = true;
    act::Swarm calib(a, {true, 0.02, 0.0, 0.0}, 1, {}, Budgeted(1e9, 1e9), UseGreedy);
    calib.RunUntil(0, 60);
    const ClassBytes ca = FirstTransmission(calib.ledger, MessageClass::MATCH_RESPONSE);
    ASSERT_GT(ca.payload, 0);
    const double ratio = ca.transmitted / ca.payload;
    // ... and gate on scene B (300 keypoints, 3 agents, another seed), never seen by the calibration.
    act::Scenario b;
    b.advertise_features = true;
    b.keypoints = 300;
    b.agents = 3;
    act::Swarm s(b, {true, 0.02, 0.0, 0.0}, 2, {}, Budgeted(1e9, 1e9, ratio), UseGreedy);
    s.RunUntil(0, 60);
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
