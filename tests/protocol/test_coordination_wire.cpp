// D175 coordination messages (MULTIAGENT_PIPELINE): SUBMAP_PATCH, TASK_BID, TASK_CONSENSUS round-trip through the one
// serializer boundary, cells pack at 2 bits, frames are mandatory, and malformed payloads are refused.
#include "core/serialization/serializer.hpp"

#include <gtest/gtest.h>

namespace ac = ariadne::core;

namespace {

ac::SubmapPatchValue Patch() {
    ac::SubmapPatchValue p;
    p.sender = 2, p.epoch = 7, p.version = 41, p.t_ns = 12'500'000'000;
    p.frame = {2, 7};
    for (int k = 0; k < 3; ++k) {
        ac::SlabBlockValue b;
        b.bx = k - 1, b.by = 5 + k;
        b.cells.resize(ac::kSlabBlockCells);
        for (int i = 0; i < ac::kSlabBlockCells; ++i) b.cells[size_t(i)] = uint8_t((i * 7 + k) % 4);
        p.blocks.push_back(b);
    }
    return p;
}

}  // namespace

TEST(CoordinationWire, SubmapPatchRoundTripsAndPacksTwoBitsPerCell) {
    const ac::WireCodec codec;
    const ac::SubmapPatchValue p = Patch();
    const auto m = codec.EncodeSubmapPatch(p, 2, 0, ac::Timestamp{p.t_ns}, 9);
    EXPECT_EQ(m.record.type, ac::MessageClass::SUBMAP_PATCH);
    // 3 blocks x 256 packed bytes, plus FlatBuffers overhead of at most 64 B per block table and 128 B for the envelope:
    // about 2 bits per cell, never a byte per cell
    EXPECT_GE(m.record.payload_bytes, 3u * ac::kSlabBlockCells / 4);
    EXPECT_LE(m.record.payload_bytes, 3u * (ac::kSlabBlockCells / 4 + 64u) + 128u);
    ac::DecodedMessage d;
    ASSERT_EQ(codec.Decode(m.bytes, &d), ac::WireCodec::Status::kOk);
    ASSERT_TRUE(d.has_submap_patch);
    const auto& q = d.submap_patch;
    EXPECT_EQ(q.sender, 2);
    EXPECT_EQ(q.epoch, 7u);
    EXPECT_EQ(q.version, 41u);
    EXPECT_EQ(q.frame.agent, 2u);
    ASSERT_EQ(q.blocks.size(), 3u);
    for (size_t k = 0; k < 3; ++k) {
        EXPECT_EQ(q.blocks[k].bx, p.blocks[k].bx);
        EXPECT_EQ(q.blocks[k].by, p.blocks[k].by);
        EXPECT_EQ(q.blocks[k].cells, p.blocks[k].cells);
    }
}

TEST(CoordinationWire, UniformBlocksCompress) {
    const ac::WireCodec codec;
    ac::SubmapPatchValue p;
    p.frame = {0, 0};
    for (int k = 0; k < 20; ++k) {
        ac::SlabBlockValue b;
        b.bx = k, b.cells.assign(ac::kSlabBlockCells, uint8_t(k % 2 ? 3 : 0));   // all free-passable or all unknown
        p.blocks.push_back(b);
    }
    const auto m = codec.EncodeSubmapPatch(p, 0, 1, ac::Timestamp{0}, 1);
    EXPECT_TRUE(m.record.compressed);
    EXPECT_LT(m.record.transmitted_bytes, m.record.payload_bytes / 4);
}

TEST(CoordinationWire, PatchRefusesBadCells) {
    const ac::WireCodec codec;
    ac::SubmapPatchValue p = Patch();
    p.blocks[0].cells[3] = 4;
    EXPECT_THROW(codec.EncodeSubmapPatch(p, 2, 0, ac::Timestamp{0}, 1), std::invalid_argument);
    p = Patch();
    p.blocks[1].cells.pop_back();
    EXPECT_THROW(codec.EncodeSubmapPatch(p, 2, 0, ac::Timestamp{0}, 1), std::invalid_argument);
}

TEST(CoordinationWire, TaskBidRoundTrips) {
    const ac::WireCodec codec;
    ac::TaskBidValue b;
    b.sender = 1, b.epoch = 3, b.seq = 17, b.t_ns = 25'000'000'000, b.lambda_path = 0.35f;
    b.frame = {0, 3};
    for (int i = 0; i < 8; ++i) {
        ac::BidOptionValue o;
        o.task_key = 1000 + i, o.goal_x = 1.25 * i, o.goal_y = -0.5 * i;
        for (int j = 0; j < 5; ++j) o.ig_raw[size_t(j)] = float(i + j) * 0.1f;
        o.path_m = 3.0f + float(i), o.cost_bandwidth = 0.2f, o.cost_risk = 0.05f, o.risk = 0.01f;
        o.source = uint8_t('F');
        b.options.push_back(o);
    }
    const auto m = codec.EncodeTaskBid(b, 1, 0, ac::Timestamp{b.t_ns}, 4);
    EXPECT_FALSE(m.record.compressed);
    ac::DecodedMessage d;
    ASSERT_EQ(codec.Decode(m.bytes, &d), ac::WireCodec::Status::kOk);
    ASSERT_TRUE(d.has_task_bid);
    const auto& q = d.task_bid;
    EXPECT_EQ(q.sender, 1);
    EXPECT_EQ(q.seq, 17u);
    EXPECT_FLOAT_EQ(q.lambda_path, 0.35f);
    ASSERT_EQ(q.options.size(), 8u);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(q.options[i].task_key, b.options[i].task_key);
        EXPECT_DOUBLE_EQ(q.options[i].goal_x, b.options[i].goal_x);
        EXPECT_DOUBLE_EQ(q.options[i].goal_y, b.options[i].goal_y);
        EXPECT_EQ(q.options[i].ig_raw, b.options[i].ig_raw);
        EXPECT_FLOAT_EQ(q.options[i].path_m, b.options[i].path_m);
        EXPECT_EQ(q.options[i].source, b.options[i].source);
    }
}

TEST(CoordinationWire, TaskConsensusRoundTrips) {
    const ac::WireCodec codec;
    ac::TaskConsensusValue c;
    c.broker = 0, c.epoch = 3, c.seq = 5, c.t_ns = 30'000'000'000, c.bids_t_ns = 25'000'000'000;
    c.members = {0, 1, 2};
    c.assignments = {{0, 11, 1.5, 2.5}, {2, 1004, -3.0, 0.25}};
    c.frame = {0, 3};
    const auto m = codec.EncodeTaskConsensus(c, 0, 1, ac::Timestamp{c.t_ns}, 6);
    ac::DecodedMessage d;
    ASSERT_EQ(codec.Decode(m.bytes, &d), ac::WireCodec::Status::kOk);
    ASSERT_TRUE(d.has_task_consensus);
    const auto& q = d.task_consensus;
    EXPECT_EQ(q.broker, 0);
    EXPECT_EQ(q.bids_t_ns, c.bids_t_ns);
    EXPECT_EQ(q.members, c.members);
    ASSERT_EQ(q.assignments.size(), 2u);
    EXPECT_EQ(q.assignments[1].agent, 2);
    EXPECT_EQ(q.assignments[1].task_key, 1004);
    EXPECT_DOUBLE_EQ(q.assignments[1].goal_y, 0.25);
}
