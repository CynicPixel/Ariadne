// 1.6 protocol pieces (design COLLAB_1.6): the three new payloads round-trip
// through the one serializer boundary, malformed responses are rejected, and the
// inbox is idempotent and order-restoring (§15.3; Phase 1 exit protocol test).
#include "collaboration/protocol/inbox.hpp"
#include "collaboration/retrieval/mean_bit_descriptor.hpp"
#include "core/serialization/serializer.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace ariadne;
using core::MessageClass;

namespace {
core::DecodedMessage RoundTrip(const core::SerializedMessage& m) {
    core::DecodedMessage d;
    EXPECT_EQ(core::WireCodec().Decode(m.bytes, &d), core::WireCodec::Status::kOk);
    return d;
}
core::DecodedMessage Msg(uint32_t sender, MessageClass cls, uint64_t seq) {
    core::DecodedMessage d;
    d.sender = sender;
    d.cls = cls;
    d.sequence = seq;
    return d;
}
}  // namespace

TEST(CollabProtocol, PayloadsRoundTrip) {
    const core::WireCodec codec;
    core::GlobalDescriptorValue g{42, 123456789, std::vector<float>(256, 0.0625f), 900.0f};
    const auto dg = RoundTrip(codec.EncodeGlobalDescriptor(g, 1, 0, core::Timestamp{5}, 7));
    ASSERT_TRUE(dg.has_global_descriptor);
    EXPECT_EQ(dg.cls, MessageClass::GLOBAL_DESCRIPTOR);
    EXPECT_EQ(dg.sequence, 7u);
    EXPECT_EQ(dg.global_descriptor.keyframe_id, 42u);
    EXPECT_EQ(dg.global_descriptor.descriptor, g.descriptor);

    const auto dr = RoundTrip(codec.EncodeMatchRequest({3, 10, 20, 0}, 0, 1, core::Timestamp{6}, 0));
    ASSERT_TRUE(dr.has_match_request);
    EXPECT_EQ(dr.match_request.candidate_keyframe, 20u);

    core::FeatureResponseValue f{3, 20, {1.5f, 2.5f, 3.5f, 4.5f}, std::vector<uint8_t>(64, 0xA5),
                                 {0.1f, 0.2f, 1.0f, NAN, NAN, NAN}, {680, 680, 639.5f, 359.5f, 0.1f}};
    const auto m = codec.EncodeFeatureResponse(f, 1, 0, core::Timestamp{7}, 0);
    EXPECT_EQ(m.record.type, MessageClass::MATCH_RESPONSE);
    const auto df = RoundTrip(m);
    ASSERT_TRUE(df.has_feature_response);
    EXPECT_EQ(df.feature_response.keypoints, f.keypoints);
    EXPECT_EQ(df.feature_response.descriptors, f.descriptors);
    EXPECT_TRUE(std::isnan(df.feature_response.points3d[3]));   // "no depth" survives the wire
    EXPECT_EQ(df.feature_response.points3d[2], 1.0f);
}

TEST(CollabProtocol, MalformedFeatureResponseIsRejected) {
    core::FeatureResponseValue f{1, 2, {1.0f, 2.0f}, std::vector<uint8_t>(31, 0), {0, 0, 1}, {1, 1, 0, 0, 0.1f}};
    core::DecodedMessage d;
    EXPECT_EQ(core::WireCodec().Decode(core::WireCodec().EncodeFeatureResponse(f, 1, 0, core::Timestamp{1}, 0).bytes, &d),
              core::WireCodec::Status::kMalformed);   // 31 descriptor bytes for one keypoint
}

TEST(CollabProtocol, InboxIsIdempotentAndRestoresOrder) {
    collaboration::Inbox in;
    EXPECT_EQ(in.Offer(Msg(1, MessageClass::POSE_EDGE, 1)).size(), 0u);   // early: buffered
    EXPECT_EQ(in.Offer(Msg(1, MessageClass::POSE_EDGE, 2)).size(), 0u);
    const auto d = in.Offer(Msg(1, MessageClass::POSE_EDGE, 0));          // the gap fills
    ASSERT_EQ(d.size(), 3u);
    for (uint64_t i = 0; i < 3; ++i) EXPECT_EQ(d[i].sequence, i);
    EXPECT_TRUE(in.Offer(Msg(1, MessageClass::POSE_EDGE, 1)).empty());     // duplicate of a delivered one
    EXPECT_TRUE(in.Offer(Msg(1, MessageClass::POSE_EDGE, 4)).empty());
    EXPECT_TRUE(in.Offer(Msg(1, MessageClass::POSE_EDGE, 4)).empty());     // duplicate of a buffered one
    EXPECT_EQ(in.duplicates(), 2u);
    // Streams are independent per (sender, class).
    EXPECT_EQ(in.Offer(Msg(2, MessageClass::POSE_EDGE, 0)).size(), 1u);
    EXPECT_EQ(in.Offer(Msg(1, MessageClass::GLOBAL_DESCRIPTOR, 0)).size(), 1u);
    EXPECT_EQ(in.waiting(), 1u);   // POSE_EDGE #4 from sender 1 still waits for #3
}

TEST(CollabProtocol, MeanBitDescriptorIsUnitAndSelfSimilar) {
    std::vector<uint8_t> d(32 * 10);
    for (size_t i = 0; i < d.size(); ++i) d[i] = uint8_t(i * 37);
    const auto v = collaboration::MeanBitDescriptor(d);
    ASSERT_EQ(v.size(), 256u);
    EXPECT_NEAR(collaboration::Cosine(v, v), 1.0, 1e-6);
    EXPECT_THROW(collaboration::MeanBitDescriptor(std::vector<uint8_t>(31)), std::invalid_argument);
}
