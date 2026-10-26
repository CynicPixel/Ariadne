// D174 (live-collaboration.md §2): the features hook may answer "not ready". The node keeps the request and answers it on a later
// Tick (a peer's MATCH_REQUEST, or the broker's own keyframe), so a slow, lazy extraction in the live worker never
// blocks a tick; a request never ready by the TTL is dropped and counted, and the run still never ends corrupt.
#include "collab_scenario.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>

namespace ac = ariadne::collaboration;
namespace act = ariadne::collaboration::testing;
namespace core = ariadne::core;

namespace {

/// Wraps each node's (always ready) scenario hook: a keyframe's features are "not ready" for its first `delay` asks.
/// The node asks once on arrival and once per Tick, so delay = 2 answers two ticks after the request arrived.
std::function<void(ac::NodeHooks&)> Delayed(int delay, std::shared_ptr<std::map<std::pair<int, uint64_t>, int>> asks) {
    auto next = std::make_shared<int>(0);
    return [=](ac::NodeHooks& h) {
        const int self = (*next)++;
        auto ready = h.features;
        h.features = [=](uint64_t kf) -> std::optional<core::FeatureResponseValue> {
            if ((*asks)[{self, kf}]++ < delay) return std::nullopt;
            return ready(kf);
        };
    };
}

}  // namespace

TEST(DeferredFeatures, AnsweredOnALaterTickEndsConsistent) {
    auto asks = std::make_shared<std::map<std::pair<int, uint64_t>, int>>();
    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 1, {}, {}, Delayed(2, asks));
    s.RunUntil(0, 90);
    const auto r = s.Outcome();
    ASSERT_EQ(r.outcome, ac::Outcome::kConsistent);
    uint64_t deferred = 0, given_up = 0;
    for (const auto& n : s.nodes) deferred += n->stats().answers_deferred, given_up += n->stats().answers_given_up;
    EXPECT_GT(deferred, 0u);   // peers' requests really waited
    EXPECT_EQ(given_up, 0u);
    EXPECT_GT(s.nodes[0]->stats().edges_verified, 4u);
    for (size_t a = 0; a < 4; ++a) {
        const auto T = s.nodes[a]->T_G_W();
        ASSERT_TRUE(T.has_value()) << "agent " << a;
        const core::SE3 gt = s.sc.T_World_W(0).inverse() * s.sc.T_World_W(a);
        EXPECT_LT((T->t - gt.t).norm(), 2e-3) << "agent " << a;
    }
}

TEST(DeferredFeatures, AlwaysReadyHooksNeverDefer) {
    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 1);
    s.RunUntil(0, 30);
    for (const auto& n : s.nodes) EXPECT_EQ(n->stats().answers_deferred, 0u);
}

TEST(DeferredFeatures, NeverReadyIsGivenUpAndNeverCorrupt) {
    auto asks = std::make_shared<std::map<std::pair<int, uint64_t>, int>>();
    ac::NodeParams np;
    np.response_ttl_s = 5.0;
    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 1, {}, np, Delayed(1 << 30, asks));
    s.RunUntil(0, 60);
    uint64_t given_up = 0, verified = 0;
    for (const auto& n : s.nodes) given_up += n->stats().answers_given_up, verified += n->stats().edges_verified;
    EXPECT_GT(given_up, 0u);
    EXPECT_EQ(verified, 0u);   // no features ever crossed, so no inter-agent edge
    EXPECT_NE(s.Outcome().outcome, ac::Outcome::kCorrupt);
}
