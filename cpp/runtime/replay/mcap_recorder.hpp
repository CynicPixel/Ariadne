#pragma once
// ---------------------------------------------------------------------------
// MCAP recording and replay (arch §5.9).
//
// Gazebo renders once per (scene, trajectory, seed) and writes a bag; every
// perception, SLAM, collaboration and ablation run afterwards consumes that
// bag, with degradation applied at read time by the seeded operator. Identical
// pixels across every baseline follow by construction rather than discipline.
//
// WHAT IS STORED: the FRAMED message, exactly as the network would carry it
// (frame header plus possibly-compressed FlatBuffer). Replay therefore
// exercises the identical decode path, frame-tag validation and byte
// accounting as a live run, and a wire-format bug cannot hide behind a
// different on-disk representation.
//
// This uses the STANDALONE Foxglove SDK, not rosbag2 (D3). §3's dependency rule
// requires replay to work from files alone with no ROS present. MCAP is a
// container rather than a ROS API, so the bags remain rosbag2-readable.
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"

#include <mcap/reader.hpp>
#include <mcap/writer.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ariadne::runtime {

using core::ByteLedger;
using core::MessageClass;
using core::SerializedMessage;

/// Topic every Ariadne wire message is recorded on. One topic keeps ordering
/// total across message classes, which matters because replay must reproduce
/// the exact interleaving a live run saw.
inline constexpr const char* kAriadneWireTopic = "/ariadne/wire";

/// MCAP profile string. "ariadne" rather than "ros2" because these are not ROS
/// messages: a reader that assumed ROS conventions would mis-handle them.
inline constexpr const char* kAriadneProfile = "ariadne";

class McapRecordWriter {
public:
    /// Opens `path` for writing. `schema_text` should be the contents of
    /// wire.fbs, so a bag carries the schema that produced it and stays
    /// readable after the schema evolves.
    mcap::Status Open(const std::string& path, const std::string& schema_text) {
        mcap::McapWriterOptions options(kAriadneProfile);
        // Chunk-level zstd. This is a FILE-SIZE choice and has no effect on
        // wire byte accounting, which is computed at the serializer: a bag is
        // storage, not a link. Classes 4-6 arrive already compressed, so the
        // additional gain is mostly on the uncompressed classes.
        options.compression = mcap::Compression::Zstd;

        const mcap::Status status = writer_.open(path, options);
        if (!status.ok()) return status;

        mcap::Schema schema("ariadne.wire.Envelope", "flatbuffer", schema_text);
        writer_.addSchema(schema);

        mcap::Channel channel(kAriadneWireTopic, "flatbuffer", schema.id);
        writer_.addChannel(channel);
        channel_id_ = channel.id;
        open_ = true;
        return status;
    }

    /// Append one message and charge its bytes to the ledger. Writing and
    /// accounting happen together so a recorded message cannot go uncounted.
    mcap::Status Write(SerializedMessage& msg, ByteLedger* ledger) {
        mcap::Message record;
        record.channelId = channel_id_;
        record.sequence = sequence_++;
        // logTime and publishTime both come from the message's own timestamp.
        // Using a wall clock here would make bags non-reproducible, which would
        // defeat the entire point of §5.9.
        record.logTime = static_cast<mcap::Timestamp>(msg.record.t.ns);
        record.publishTime = record.logTime;
        record.dataSize = msg.bytes.size();
        record.data = reinterpret_cast<const std::byte*>(msg.bytes.data());

        const mcap::Status status = writer_.write(record);
        if (status.ok() && ledger != nullptr) ledger->Record(msg.record);
        return status;
    }

    void Close() {
        if (open_) {
            writer_.close();
            open_ = false;
        }
    }

    ~McapRecordWriter() { Close(); }

private:
    mcap::McapWriter writer_;
    mcap::ChannelId channel_id_ = 0;
    uint32_t sequence_ = 0;
    bool open_ = false;
};

/// Replays framed messages from a bag, in log-time order.
class McapRecordReader {
public:
    mcap::Status Open(const std::string& path) { return reader_.open(path); }

    /// Invoke `on_frame` for each recorded message, in log-time order.
    /// Returns the number of messages replayed.
    ///
    /// Problems are surfaced through `on_problem` rather than ignored: a
    /// truncated or corrupt bag must be visible, since silently replaying a
    /// prefix would produce a plausible but incomplete result.
    size_t ForEach(const std::function<void(const std::vector<uint8_t>&,
                                            uint64_t log_time)>& on_frame,
                   const std::function<void(const mcap::Status&)>& on_problem =
                       nullptr) {
        size_t count = 0;
        auto problem_handler = [&on_problem](const mcap::Status& status) {
            if (on_problem) on_problem(status);
        };
        auto view = reader_.readMessages(problem_handler);
        for (const auto& item : view) {
            std::vector<uint8_t> frame(
                reinterpret_cast<const uint8_t*>(item.message.data),
                reinterpret_cast<const uint8_t*>(item.message.data) +
                    item.message.dataSize);
            on_frame(frame, static_cast<uint64_t>(item.message.logTime));
            ++count;
        }
        return count;
    }

    void Close() { reader_.close(); }
    ~McapRecordReader() { Close(); }

private:
    mcap::McapReader reader_;
};

}  // namespace ariadne::runtime
