// ---------------------------------------------------------------------------
// ariadne_gz_recorder: records gz-transport topics straight into a rosbag2 MCAP
// bag, with no DDS hop (DECISIONS D40, design SIM_1.1-1.3 §7).
//
//   ariadne_gz_recorder --out DIR --end-ns N --report FILE [--max-backlog-mb M]
//       [--connect-timeout-s S] --topic ROS_TOPIC,GZ_TOPIC,GZ_TYPE,ROS_TYPE,PERIOD_NS,FIRST_NS ...
//
// Why: three logged full recordings (SMOKE §4l) attributed every lost message to
// the DDS link between ros_gz_bridge and `ros2 bag record` (residual 0 elsewhere),
// including on queues 1000 deep. This tool subscribes to gz-transport itself,
// converts with ros_gz_bridge's own installed functions (convert_gz_to_ros, so the
// bytes are what the bridge would have published) and writes through rosbag2_cpp
// with NO write cache (synchronous: rosbag2's cache is a drop path).
//
// LOUD by design — every way a message could go missing is an error that is
// printed the moment it happens, counted in the JSON report, and makes the exit
// code non-zero:
//   * stamp continuity per topic: after the first message every interval must be
//     exactly the topic's period (a multiple k > 1 = k-1 messages LOST; anything
//     else = irregular; 0 = duplicate; < 0 = out of order). The one exception: repeats of stamp 0 while the world is
//     still paused (gz republishes /clock = 0) are counted as paused_repeats and
//     not written.
//   * start: the first stamp of every topic must be exactly FIRST_NS (measured:
//     0 for cameras, camera_info and /clock with a paused start; one period for
//     the mover's IMU/pose, which publishes from its first step).
//   * end: every topic must reach its last stamp floor(N/period)*period.
//   * backlog: the queue between gz-transport's callback and the writer is capped
//     (default 4 GiB); exceeding it ABORTS the recording instead of dropping.
//   * connection: the tool waits until every topic has a discovered publisher,
//     then prints "CONNECTED" — sim_record starts the world paused and unpauses
//     only after that line, so nothing is published before we listen.
// gz-transport's own ZeroMQ receive queue is unbounded (GZ_TRANSPORT_RCVHWM=0, set
// by sim_record) so the only queue that can fill is ours, and that one is loud.
// ---------------------------------------------------------------------------
#include <gz/msgs/camera_info.pb.h>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/imu.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/transport/Node.hh>
#include <ros_gz_bridge/convert.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};
void OnSignal(int) { g_stop = true; }

struct TopicSpec {
    std::string ros_topic, gz_topic, gz_type, ros_type;
    int64_t period_ns = 0;
    int64_t first_ns = 0;   // the stamp the first message must carry
};

struct TopicState {
    TopicSpec spec;
    int64_t count = 0, first_ns = -1, last_ns = -1;
    int64_t lost = 0, irregular = 0, duplicates = 0, out_of_order = 0;
    int64_t paused_repeats = 0;   // stamp-0 repeats while the world is still paused (not written)
    bool late_start = false;
    std::vector<std::pair<int64_t, int64_t>> gaps;   // (from, to) stamps, first 100
};

/// One received message on its way to the writer: conversion + serialization
/// run on the writer thread so the gz-transport callback only copies.
struct Item {
    size_t topic = 0;
    int64_t ns = 0;
    size_t bytes = 0;
    std::function<std::shared_ptr<rclcpp::SerializedMessage>()> serialize;
};

int64_t StampNs(const gz::msgs::Time& t) { return int64_t(t.sec()) * 1000000000 + t.nsec(); }
template <typename GzT>
int64_t StampOf(const GzT& m) { return StampNs(m.header().stamp()); }
template <>
int64_t StampOf<gz::msgs::Clock>(const gz::msgs::Clock& m) { return StampNs(m.sim()); }

/// ROS CDR into a zero-filled buffer: when the estimate covers the message (it is
/// generous) rmw serializes in place, so alignment padding is 0, not heap residue.
template <typename RosT>
std::shared_ptr<rclcpp::SerializedMessage> Serialize(const RosT& msg, size_t estimate) {
    auto out = std::make_shared<rclcpp::SerializedMessage>(estimate);
    auto& raw = out->get_rcl_serialized_message();
    std::memset(raw.buffer, 0, raw.buffer_capacity);
    static const rclcpp::Serialization<RosT> ser;
    ser.serialize_message(&msg, out.get());
    return out;
}

class Recorder {
public:
    Recorder(std::vector<TopicSpec> specs, int64_t end_ns, size_t max_backlog)
        : end_ns_(end_ns), max_backlog_(max_backlog) {
        for (auto& s : specs) {
            TopicState t;
            t.spec = std::move(s);
            topics_.push_back(std::move(t));
        }
    }

    void Subscribe() {
        for (size_t i = 0; i < topics_.size(); ++i) {
            const auto& s = topics_[i].spec;
            const std::string pair = s.gz_type + " -> " + s.ros_type;
            bool ok;
            if (s.gz_type == "gz.msgs.Clock" && s.ros_type == "rosgraph_msgs/msg/Clock") {
                ok = Sub<gz::msgs::Clock, rosgraph_msgs::msg::Clock>(i);
            } else if (s.gz_type == "gz.msgs.Image" && s.ros_type == "sensor_msgs/msg/Image") {
                ok = Sub<gz::msgs::Image, sensor_msgs::msg::Image>(i);
            } else if (s.gz_type == "gz.msgs.CameraInfo" && s.ros_type == "sensor_msgs/msg/CameraInfo") {
                ok = Sub<gz::msgs::CameraInfo, sensor_msgs::msg::CameraInfo>(i);
            } else if (s.gz_type == "gz.msgs.IMU" && s.ros_type == "sensor_msgs/msg/Imu") {
                ok = Sub<gz::msgs::IMU, sensor_msgs::msg::Imu>(i);
            } else if (s.gz_type == "gz.msgs.Pose" && s.ros_type == "geometry_msgs/msg/PoseStamped") {
                ok = Sub<gz::msgs::Pose, geometry_msgs::msg::PoseStamped>(i);
            } else {
                throw std::runtime_error("unsupported type pair " + pair + " on " + s.gz_topic);
            }
            if (!ok) throw std::runtime_error("cannot subscribe to " + s.gz_topic);
        }
    }

    /// Blocks until every topic has a discovered publisher (or the timeout).
    void AwaitPublishers(double timeout_s) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
        while (true) {
            std::vector<std::string> missing;
            for (const auto& t : topics_) {
                std::vector<gz::transport::MessagePublisher> pubs, subs;
                if (!node_.TopicInfo(t.spec.gz_topic, pubs, subs) || pubs.empty()) missing.push_back(t.spec.gz_topic);
            }
            if (missing.empty()) return;
            if (std::chrono::steady_clock::now() > deadline || g_stop) {
                std::string m = "no publisher discovered for:";
                for (const auto& t : missing) m += " " + t;
                throw std::runtime_error(m);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    /// Writer loop on the calling thread until every topic is complete and the
    /// queue is empty, or a stop signal / fatal error. Returns the exit code.
    int Run(const std::string& out_dir, const std::string& report_path) {
        rosbag2_storage::StorageOptions so;
        so.uri = out_dir;
        so.storage_id = "mcap";
        so.storage_preset_profile = "zstd_fast";
        so.max_cache_size = 0;                      // synchronous writes: no cache, no cache drops
        rosbag2_cpp::Writer writer;
        writer.open(so, {"cdr", "cdr"});
        const auto t0 = std::chrono::steady_clock::now();
        while (true) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(100), [&] { return !q_.empty() || fatal_; });
            if (fatal_) break;
            if (q_.empty()) {
                if (AllCompleteLocked() || g_stop) break;
                continue;
            }
            Item it = std::move(q_.front());
            q_.pop_front();
            backlog_ -= it.bytes;
            lk.unlock();
            try {
                const auto& s = topics_[it.topic].spec;
                writer.write(it.serialize(), s.ros_topic, s.ros_type, it.ns, it.ns);   // log time = stamp
                ++written_;
            } catch (const std::exception& e) {
                Error(std::string("write failed: ") + e.what());
                std::lock_guard<std::mutex> g(mu_);
                fatal_ = true;
            }
        }
        // Stop listening before closing so no callback races the final report.
        for (const auto& t : topics_) node_.Unsubscribe(t.spec.gz_topic);
        writer.close();
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> g(mu_);
        if (!fatal_) {
            for (const auto& t : topics_) {
                const int64_t want = (end_ns_ / t.spec.period_ns) * t.spec.period_ns;
                if (t.last_ns != want) {
                    ErrorLocked("incomplete: " + t.spec.ros_topic + " last stamp " + std::to_string(t.last_ns) +
                          " != " + std::to_string(want));
                }
            }
        }
        WriteReport(report_path, wall);
        return errors_.empty() ? 0 : 1;
    }

private:
    template <typename GzT, typename RosT>
    bool Sub(size_t i) {
        std::function<void(const GzT&)> cb = [this, i](const GzT& m) {
            auto copy = std::make_shared<GzT>(m);
            const size_t bytes = m.ByteSizeLong();
            Item it;
            it.topic = i;
            it.ns = StampOf(m);
            it.bytes = bytes;
            it.serialize = [copy, bytes]() {
                RosT ros;
                ros_gz_bridge::convert_gz_to_ros(*copy, ros);
                return Serialize(ros, bytes + 4096);
            };
            Enqueue(std::move(it));
        };
        return node_.Subscribe(topics_[i].spec.gz_topic, cb);
    }

    void Enqueue(Item it) {
        std::lock_guard<std::mutex> g(mu_);
        if (fatal_) return;
        TopicState& t = topics_[it.topic];
        // A paused world republishes sim time 0 (/clock does, ~140 Hz wall): before a
        // topic's first non-zero stamp, a repeat of stamp 0 is not a duplicate sample
        // but the same instant again. Counted, not written. Any other repeat is an error.
        if (it.ns == 0 && t.count > 0 && t.last_ns == 0) {
            ++t.paused_repeats;
            return;
        }
        Check(t, it.ns);
        if (backlog_ + it.bytes > max_backlog_) {
            ErrorLocked("backlog " + std::to_string((backlog_ + it.bytes) >> 20) + " MiB exceeds the cap of " +
                        std::to_string(max_backlog_ >> 20) + " MiB: the writer cannot keep up; ABORTING "
                        "(a dropped message would be worse)");
            fatal_ = true;
            cv_.notify_all();
            return;
        }
        backlog_ += it.bytes;
        peak_backlog_ = std::max(peak_backlog_, backlog_);
        q_.push_back(std::move(it));
        cv_.notify_one();
    }

    /// Stamp continuity, under mu_. Every anomaly is an error, printed now.
    void Check(TopicState& t, int64_t ns) {
        const int64_t p = t.spec.period_ns;
        const std::string& name = t.spec.ros_topic;
        if (t.count == 0) {
            t.first_ns = ns;
            if (ns != t.spec.first_ns) {
                t.late_start = true;
                ErrorLocked(name + ": first stamp " + std::to_string(ns) + " ns != expected " +
                            std::to_string(t.spec.first_ns) + " ns: the start was missed");
            }
        } else {
            const int64_t d = ns - t.last_ns;
            if (d == 0) {
                ++t.duplicates;
                ErrorLocked(name + ": duplicate stamp " + std::to_string(ns));
            } else if (d < 0) {
                ++t.out_of_order;
                ErrorLocked(name + ": out of order " + std::to_string(t.last_ns) + " -> " + std::to_string(ns));
            } else if (d % p == 0 && d > p) {
                t.lost += d / p - 1;
                if (t.gaps.size() < 100) t.gaps.emplace_back(t.last_ns, ns);
                ErrorLocked(name + ": LOST " + std::to_string(d / p - 1) + " message(s) between " +
                            std::to_string(t.last_ns) + " and " + std::to_string(ns) + " ns");
            } else if (d != p) {
                ++t.irregular;
                ErrorLocked(name + ": irregular interval " + std::to_string(d) + " ns (period " +
                            std::to_string(p) + ") at " + std::to_string(ns));
            }
        }
        if (ns > t.last_ns) t.last_ns = ns;
        ++t.count;
    }

    bool AllCompleteLocked() const {
        for (const auto& t : topics_) {
            if (t.last_ns != (end_ns_ / t.spec.period_ns) * t.spec.period_ns) return false;
        }
        return true;
    }

    void Error(const std::string& m) {
        std::lock_guard<std::mutex> g(mu_);
        ErrorLocked(m);
    }
    void ErrorLocked(const std::string& m) {
        std::cerr << "[gz_recorder] ERROR: " << m << std::endl;
        if (errors_.size() < 1000) errors_.push_back(m);
        ++n_errors_;
    }

    static std::string Json(const std::string& s) {
        std::string o = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') o += '\\';
            if (static_cast<unsigned char>(c) < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                o += b;
                continue;
            }
            o += c;
        }
        return o + "\"";
    }

    void WriteReport(const std::string& path, double wall) const {
        std::ostringstream o;
        o << "{\n \"tool\": \"ariadne_gz_recorder\", \"end_ns\": " << end_ns_ << ", \"written\": " << written_
          << ", \"peak_backlog_bytes\": " << peak_backlog_ << ", \"max_backlog_bytes\": " << max_backlog_
          << ", \"aborted\": " << (fatal_ ? "true" : "false") << ", \"wall_s\": " << wall
          << ", \"error_count\": " << n_errors_ << ",\n \"errors\": [";
        for (size_t i = 0; i < errors_.size(); ++i) o << (i ? ", " : "") << Json(errors_[i]);
        o << "],\n \"topics\": {";
        for (size_t i = 0; i < topics_.size(); ++i) {
            const auto& t = topics_[i];
            o << (i ? ",\n  " : "\n  ") << Json(t.spec.ros_topic) << ": {\"count\": " << t.count
              << ", \"first_ns\": " << t.first_ns << ", \"last_ns\": " << t.last_ns << ", \"period_ns\": "
              << t.spec.period_ns << ", \"lost\": " << t.lost << ", \"irregular\": " << t.irregular
              << ", \"duplicates\": " << t.duplicates << ", \"out_of_order\": " << t.out_of_order
              << ", \"paused_repeats\": " << t.paused_repeats
              << ", \"late_start\": " << (t.late_start ? "true" : "false") << ", \"gaps\": [";
            for (size_t k = 0; k < t.gaps.size(); ++k) {
                o << (k ? ", " : "") << "[" << t.gaps[k].first << ", " << t.gaps[k].second << "]";
            }
            o << "]}";
        }
        o << "\n }\n}\n";
        std::ofstream f(path);
        f << o.str();
        if (!f) std::cerr << "[gz_recorder] ERROR: cannot write report " << path << std::endl;
    }

    gz::transport::Node node_;
    std::vector<TopicState> topics_;
    const int64_t end_ns_;
    const size_t max_backlog_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> q_;
    size_t backlog_ = 0, peak_backlog_ = 0;
    bool fatal_ = false;
    int64_t written_ = 0, n_errors_ = 0;
    std::vector<std::string> errors_;
};

TopicSpec ParseTopic(const std::string& arg) {
    std::vector<std::string> f;
    std::stringstream ss(arg);
    for (std::string x; std::getline(ss, x, ',');) f.push_back(x);
    if (f.size() != 6) {
        throw std::invalid_argument("--topic needs ROS_TOPIC,GZ_TOPIC,GZ_TYPE,ROS_TYPE,PERIOD_NS,FIRST_NS: " + arg);
    }
    TopicSpec s{f[0], f[1], f[2], f[3], std::stoll(f[4]), std::stoll(f[5])};
    if (s.period_ns <= 0 || s.first_ns < 0) throw std::invalid_argument("period must be > 0, first >= 0: " + arg);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string out, report;
        int64_t end_ns = -1;
        size_t max_backlog = size_t(4096) << 20;
        double connect_timeout = 120.0;
        std::vector<TopicSpec> specs;
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
            const std::string v = argv[++i];
            if (k == "--out") out = v;
            else if (k == "--report") report = v;
            else if (k == "--end-ns") end_ns = std::stoll(v);
            else if (k == "--max-backlog-mb") max_backlog = size_t(std::stoull(v)) << 20;
            else if (k == "--connect-timeout-s") connect_timeout = std::stod(v);
            else if (k == "--topic") specs.push_back(ParseTopic(v));
            else throw std::invalid_argument("unknown argument " + k);
        }
        if (out.empty() || report.empty() || end_ns < 0 || specs.empty()) {
            throw std::invalid_argument("usage: ariadne_gz_recorder --out DIR --end-ns N --report FILE --topic ...");
        }
        if (std::filesystem::exists(out)) throw std::runtime_error(out + " exists; refusing to overwrite");
        std::signal(SIGINT, OnSignal);
        std::signal(SIGTERM, OnSignal);

        Recorder rec(std::move(specs), end_ns, max_backlog);
        rec.Subscribe();
        rec.AwaitPublishers(connect_timeout);
        // ZeroMQ "slow joiner": the SUB connection completes asynchronously after
        // discovery. The world is still paused, so waiting costs nothing.
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "CONNECTED" << std::endl;
        const int rc = rec.Run(out, report);
        std::cout << (rc == 0 ? "COMPLETE" : "FAILED") << std::endl;
        return rc;
    } catch (const std::exception& e) {
        std::cerr << "[gz_recorder] FATAL: " << e.what() << std::endl;
        return 2;
    }
}
