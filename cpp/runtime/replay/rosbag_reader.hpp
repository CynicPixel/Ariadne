#pragma once
// ---------------------------------------------------------------------------
// Reads a rosbag2 MCAP recording (the simulator's bags, design SIM_1.1-1.3 §7)
// without ROS: the Foxglove MCAP SDK walks the container (zstd chunks included)
// and ros_cdr.hpp decodes the messages. rosbag2 writes one Channel per topic
// with messageEncoding "cdr" and a Schema whose name is the ROS type
// ("sensor_msgs/msg/Image") and encoding "ros2msg".
//
// Messages are yielded in LOG-TIME order (explicitly requested: the SDK's
// default is file order, and rosbag2 interleaves concurrent subscriptions), ties
// broken by file position. Problems (truncation, CRC, unknown compression) are
// passed to on_problem and also make Replay() report failure: a partially read
// bag must never pass for a whole one.
// ---------------------------------------------------------------------------
#include <mcap/reader.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ariadne::runtime {

struct RosMessage {
    std::string_view topic;
    std::string_view type;          // e.g. "sensor_msgs/msg/Image"
    uint64_t log_time_ns = 0;       // recorder receive time (sim clock under use_sim_time)
    uint64_t publish_time_ns = 0;
    const uint8_t* data = nullptr;  // CDR bytes, valid only during the callback
    size_t size = 0;
};

class RosbagReader {
public:
    /// Opens the .mcap file and reads its summary (channels, schemas, statistics).
    void Open(const std::string& path) {
        const auto s = reader_.open(path);
        if (!s.ok()) throw std::runtime_error("rosbag: cannot open " + path + ": " + s.message);
        const auto r = reader_.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
        if (!r.ok()) throw std::runtime_error("rosbag: cannot read summary of " + path + ": " + r.message);
        topics_.clear();
        for (const auto& [id, ch] : reader_.channels()) {
            if (ch->messageEncoding != "cdr") {
                throw std::runtime_error("rosbag: topic " + ch->topic + " is not CDR-encoded");
            }
            const auto schema = reader_.schema(ch->schemaId);
            topics_[ch->topic] = schema ? schema->name : std::string();
        }
        open_ = true;
    }

    /// topic -> ROS type name, for every channel in the file.
    const std::map<std::string, std::string>& topics() const { return topics_; }

    /// Message count per the file's Statistics record (0 if the file has none).
    uint64_t message_count() const {
        const auto& st = reader_.statistics();
        return st ? st->messageCount : 0;
    }

    /// Calls on_message for every message on `only_topics` (all topics if empty), in
    /// log-time order. Returns the number delivered; throws if any problem occurred.
    size_t Replay(const std::function<void(const RosMessage&)>& on_message,
                  const std::set<std::string>& only_topics = {}) {
        if (!open_) throw std::logic_error("rosbag: Replay before Open");
        mcap::ReadMessageOptions opts;
        opts.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
        if (!only_topics.empty()) {
            opts.topicFilter = [&only_topics](std::string_view t) {
                return only_topics.count(std::string(t)) > 0;
            };
        }
        std::string problem;
        auto on_problem = [&problem](const mcap::Status& s) {
            if (problem.empty()) problem = s.message;
        };
        size_t n = 0;
        for (const auto& view : reader_.readMessages(on_problem, opts)) {
            RosMessage m;
            m.topic = view.channel->topic;
            m.type = view.schema ? std::string_view(view.schema->name) : std::string_view();
            m.log_time_ns = view.message.logTime;
            m.publish_time_ns = view.message.publishTime;
            m.data = reinterpret_cast<const uint8_t*>(view.message.data);
            m.size = view.message.dataSize;
            on_message(m);
            ++n;
        }
        if (!problem.empty()) throw std::runtime_error("rosbag: problem while reading: " + problem);
        return n;
    }

    void Close() {
        if (open_) reader_.close();
        open_ = false;
    }
    ~RosbagReader() { Close(); }

private:
    mcap::McapReader reader_;
    std::map<std::string, std::string> topics_;
    bool open_ = false;
};

}  // namespace ariadne::runtime
