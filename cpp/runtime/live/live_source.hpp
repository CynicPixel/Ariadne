#pragma once
// ---------------------------------------------------------------------------
// The live runtime's input side (design LIVE_RUNTIME_L1 D152). ROS-free: messages arrive as (topic, CDR bytes)
// from the link (link_frame.hpp) or from a recording of a previous live run, and go through the SAME decoder and
// frame assembly as the bag replay (SimFrameAssembler).
//
//   StreamMessage     one ROS message: topic, type, header stamp, CDR bytes, release sequence number
//   MessageQueue      unbounded blocking FIFO between the reader and an agent's estimator thread
//   Reorderer         holds messages back until every timed input topic has delivered a stamp >= theirs
//                     (a watermark), then releases them in (stamp, topic, arrival) order: the order the
//                     estimator consumes and the recording stores. A topic silent past the hold-back timeout is
//                     skipped by the watermark, and that is logged (released_on_timeout).
//   ConsumedRecorder  appends every released message to a rosbag2-layout MCAP (RosbagReader reads it) with
//                     log time = release sequence number, strictly increasing, so a replay sees exactly the
//                     consumed order even where stamps tie (left/right images share a stamp). Also an FNV-1a digest
//                     over (topic, stamp, payload) in that order: the first replay check.
//   LiveSimSource     a StereoSource over one agent's image queue. Calibration and extrinsics from the mission
//                     JSON (a live run has no canonical bag and no /tf_static); lenient pairing (drops counted).
//                     Agents take turns per frame through a shared lock (D152a): with two estimators calling into
//                     OpenCV's one parallel_for_ pool at once, one pose in 501 differed between two replays of the same
//                     recording (PnP inliers 323 vs 322; same tracks and correspondences). [UNVERIFIED] cause: a busy
//                     pool runs the other caller's job serially, changing stripe partitioning and so the order of
//                     floating-point reductions. Turn-taking removes the contention whatever the cause.
// ---------------------------------------------------------------------------
#include "core/determinism/digests.hpp"
#include "runtime/datasets/stereo_source.hpp"

#include <mcap/writer.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ariadne::runtime::live {

struct StreamMessage {
    std::string topic, type;
    int64_t stamp_ns = -1;   // header stamp; -1 = untimed (no header: std_msgs/String)
    std::shared_ptr<const std::vector<uint8_t>> data;
    uint64_t seq = 0;        // release order (1-based), = log time in the recording
};

/// The header stamp of a CDR message that starts with std_msgs/Header (Image, CameraInfo, Imu, PoseStamped):
/// bytes 4..12 after the encapsulation header. -1 for untimed types.
inline int64_t PeekStamp(const std::string& type, const std::vector<uint8_t>& b) {
    if (type == "std_msgs/msg/String") return -1;
    if (b.size() < 12 || b[1] != 0x01) throw std::runtime_error("live: short or non-little-endian CDR message");
    int32_t sec;
    uint32_t nsec;
    std::memcpy(&sec, b.data() + 4, 4);
    std::memcpy(&nsec, b.data() + 8, 4);
    return int64_t(sec) * 1000000000 + nsec;
}

class MessageQueue {
public:
    /// capacity 0 = unbounded. Bounded: Push blocks while full (backpressure; order is unchanged, only pacing). The
    /// replay reader outruns the estimators (~0.85 s per keyframe), and an unbounded queue held a whole recording's
    /// decoded RGB: a 45 s replay was OOM-killed at 7.4 GB RSS (2026-10-23).
    explicit MessageQueue(size_t capacity = 0) : cap_(capacity) {}
    void Push(StreamMessage m) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (cap_) space_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
            q_.push_back(std::move(m));
            max_depth_ = std::max(max_depth_, q_.size());
        }
        cv_.notify_one();
    }
    /// Blocks; nullopt once closed and drained.
    std::optional<StreamMessage> Pop() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        StreamMessage m = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        space_.notify_one();
        return m;
    }
    void Close() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            closed_ = true;
        }
        cv_.notify_all();
        space_.notify_all();
    }
    size_t max_depth() const {
        std::lock_guard<std::mutex> lk(mu_);
        return max_depth_;
    }

private:
    size_t cap_ = 0, max_depth_ = 0;
    mutable std::mutex mu_;
    std::condition_variable cv_, space_;
    std::deque<StreamMessage> q_;
    bool closed_ = false;
};

class Reorderer {
public:
    explicit Reorderer(std::set<std::string> timed_topics, std::chrono::milliseconds hold_back = std::chrono::seconds(3))
        : timed_(std::move(timed_topics)), hold_back_(hold_back) {}

    /// Push one arrival; calls release(m) for every message now safe to release, in release order.
    template <typename F>
    void Push(StreamMessage m, F&& release) {
        const auto now = std::chrono::steady_clock::now();
        if (m.stamp_ns < 0) {   // untimed: released at once, in arrival order
            release(std::move(m));
            return;
        }
        if (timed_.count(m.topic)) {
            auto& t = latest_[m.topic];
            t.stamp = std::max(t.stamp, m.stamp_ns);
            t.wall = now;
        }
        held_.emplace(std::make_tuple(m.stamp_ns, m.topic, arrival_++), std::move(m));
        Drain(now, release);
    }

    /// End of input: release everything still held, in order.
    template <typename F>
    void Flush(F&& release) {
        for (auto& [k, m] : held_) release(std::move(m));
        held_.clear();
    }

    long released_on_timeout() const { return timeouts_; }

private:
    template <typename F>
    void Drain(std::chrono::steady_clock::time_point now, F&& release) {
        int64_t wm = std::numeric_limits<int64_t>::max();
        bool timed_out = false;
        for (const auto& t : timed_) {
            const auto it = latest_.find(t);
            const bool silent = it == latest_.end() || now - it->second.wall > hold_back_;
            if (silent) {
                // Never heard from (or stalled): the watermark waits for it up to hold_back from the first arrival.
                if (it == latest_.end() && now - first_wall(now) <= hold_back_) return;
                timed_out = true;
                continue;
            }
            wm = std::min(wm, it->second.stamp);
        }
        while (!held_.empty() && std::get<0>(held_.begin()->first) <= wm) {
            if (timed_out) ++timeouts_;
            release(std::move(held_.begin()->second));
            held_.erase(held_.begin());
        }
    }
    std::chrono::steady_clock::time_point first_wall(std::chrono::steady_clock::time_point now) {
        if (!first_) first_ = now;
        return *first_;
    }

    struct Latest {
        int64_t stamp = -1;
        std::chrono::steady_clock::time_point wall;
    };
    std::set<std::string> timed_;
    std::chrono::milliseconds hold_back_;
    std::map<std::string, Latest> latest_;
    std::map<std::tuple<int64_t, std::string, uint64_t>, StreamMessage> held_;
    uint64_t arrival_ = 0;
    long timeouts_ = 0;
    std::optional<std::chrono::steady_clock::time_point> first_;
};

class ConsumedRecorder {
public:
    void Open(const std::string& path) {
        mcap::McapWriterOptions o("ros2");
        o.compression = mcap::Compression::Zstd;
        const auto s = w_.open(path, o);
        if (!s.ok()) throw std::runtime_error("live: cannot open recording " + path + ": " + s.message);
        open_ = true;
    }
    /// Digest only (replay: the consumed sequence is re-derived and compared, nothing is written).
    void Absorb(const StreamMessage& m) {
        for (char c : m.topic) h_.AbsorbByte(uint8_t(c));
        h_.AbsorbU64(uint64_t(m.stamp_ns));
        h_.AbsorbU64(m.data->size());
        for (uint8_t b : *m.data) h_.AbsorbByte(b);
        ++count_;
    }
    void Write(const StreamMessage& m) {
        Absorb(m);
        if (!open_) return;
        auto it = channels_.find(m.topic);
        if (it == channels_.end()) {
            mcap::Schema sc(m.type, "ros2msg", "");
            w_.addSchema(sc);
            mcap::Channel ch(m.topic, "cdr", sc.id);
            w_.addChannel(ch);
            it = channels_.emplace(m.topic, ch.id).first;
        }
        mcap::Message r;
        r.channelId = it->second;
        r.sequence = uint32_t(m.seq);
        r.logTime = mcap::Timestamp(m.seq);   // the consumed order, not a clock (D152)
        r.publishTime = mcap::Timestamp(m.stamp_ns < 0 ? 0 : m.stamp_ns);
        r.dataSize = m.data->size();
        r.data = reinterpret_cast<const std::byte*>(m.data->data());
        const auto s = w_.write(r);
        if (!s.ok()) throw std::runtime_error("live: recording write failed: " + s.message);
    }
    void Close() {
        if (open_) w_.close();
        open_ = false;
    }
    ~ConsumedRecorder() { Close(); }
    uint64_t digest() const { return h_.value(); }
    uint64_t count() const { return count_; }

private:
    mcap::McapWriter w_;
    bool open_ = false;
    std::map<std::string, mcap::ChannelId> channels_;
    core::Fnv1a64 h_;
    uint64_t count_ = 0;
};

/// Thread-safe ground-truth store (scoring only; the estimator never reads it).
class GtStore {
public:
    void Add(const TimedPose& p) {
        std::lock_guard<std::mutex> lk(mu_);
        v_.push_back(p);
    }
    std::vector<TimedPose> Snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        auto v = v_;
        std::stable_sort(v.begin(), v.end(), [](const TimedPose& a, const TimedPose& b) { return a.t_ns < b.t_ns; });
        return v;
    }

private:
    mutable std::mutex mu_;
    std::vector<TimedPose> v_;
};

class LiveSimSource : public StereoSource {
public:
    /// `turn`: held while one frame runs through the estimator (D152a). Every agent's source shares it, so frames are
    /// processed one at a time, in whatever order, and each OpenCV call sees an idle thread pool.
    LiveSimSource(const std::string& scene, int agent, MessageQueue& images, const GtStore& gt, std::mutex& turn,
                  const std::string& mission = "live")
        : agent_(agent), images_(images), gt_(gt), turn_(turn) {
        const YAML::Node sc = YAML::LoadFile(scene + "/scene.json");
        const YAML::Node m = YAML::LoadFile(scene + "/mission_" + mission + ".json");
        scene_id_ = sc["scene_id"].as<std::string>();
        rig_ = m["rig"]["name"].as<std::string>();
        YAML::Node ag;
        for (const auto& x : m["agents"]) {
            if (x["agent"].as<int>() == agent) ag.reset(x);
        }
        if (!ag) throw std::runtime_error("live: agent " + std::to_string(agent) + " not in the mission");
        t_left_ = ag["topics"]["left_image"].as<std::string>();
        t_right_ = ag["topics"]["right_image"].as<std::string>();
        const auto L = m["lens"];
        const auto R = m["render_intrinsics_pixel_centre"];
        lens_.width = m["rig"]["camera"]["width"].as<int>();
        lens_.height = m["rig"]["camera"]["height"].as<int>();
        lens_.fx = L["fx"].as<double>();
        lens_.fy = L["fy"].as<double>();
        lens_.cx = L["cx"].as<double>();
        lens_.cy = L["cy"].as<double>();
        lens_.k1 = L["k1"].as<double>();
        lens_.k2 = L["k2"].as<double>();
        lens_.p1 = L["p1"].as<double>();
        lens_.p2 = L["p2"].as<double>();
        lens_.k3 = L["k3"].as<double>();
        lens_.render_fx = R["fx"].as<double>();
        lens_.render_fy = R["fy"].as<double>();
        lens_.render_cx = R["cx"].as<double>();
        lens_.render_cy = R["cy"].as<double>();
        if (lens_.k3 != 0.0) throw std::runtime_error("live: PinholeRadtan carries k1 k2 p1 p2 only; rig.lens.k3 != 0");
        map_ = perception::BuildLensMap(lens_);
        // T_B_optical from the mission manifest: the same numbers the canonical /tf_static is built from and
        // checked against (D40), which a live run does not have.
        for (const char* side : {"left", "right"}) {
            core::SE3 T;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) T.R(i, j) = m["R_B_optical_rows"][i][j].as<double>();
                T.t(i) = m["t_B_camera_m"][side][i].as<double>();
            }
            (std::string(side) == "left" ? T_B_left_ : T_B_right_) = T;
        }
    }

    std::string RunName() const override { return scene_id_ + "-agent" + std::to_string(agent_); }
    std::string SceneName() const override { return "sim-live/" + scene_id_ + "/agent" + std::to_string(agent_); }
    std::string CameraConfig() const override { return rig_ + "+lens(D41)"; }
    std::string BagChecksum() const override { return kUnknownField; }   // the run records its consumed digest
    perception::RawStereoCalibration Calibration() const override {
        perception::PinholeRadtan c;
        c.width = lens_.width;
        c.height = lens_.height;
        c.fu = lens_.fx;
        c.fv = lens_.fy;
        c.cu = lens_.cx;
        c.cv = lens_.cy;
        c.dist = {lens_.k1, lens_.k2, lens_.p1, lens_.p2};
        perception::RawStereoCalibration s;
        s.left = s.right = c;
        s.T_right_left = T_B_right_.inverse() * T_B_left_;
        return s;
    }
    core::SE3 T_B_C0() const override { return T_B_left_; }
    std::vector<TimedPose> GroundTruth() const override { return {}; }   // late: StereoVoArgs::late_ground_truth
    size_t FrameCount() const override { return 0; }                     // unknown ahead of time
    uint32_t Agent() const override { return uint32_t(agent_); }

    size_t ForEachFrame(size_t max_frames, const std::function<void(const StereoFrame&)>& on_frame) override {
        SimFrameAssembler assembler(map_, uint32_t(agent_), t_left_, t_right_, /*strict=*/false);
        while (auto m = images_.Pop()) {
            std::lock_guard<std::mutex> lk(turn_);
            if (assembler.Feed(m->topic, m->data->data(), m->data->size(), on_frame) && assembler.emitted() == max_frames) break;
        }
        unpaired_ = assembler.unpaired() + assembler.pending();
        return assembler.emitted();
    }

    const std::string& left_topic() const { return t_left_; }
    const std::string& right_topic() const { return t_right_; }
    size_t unpaired() const { return unpaired_; }

private:
    int agent_;
    MessageQueue& images_;
    const GtStore& gt_;
    std::mutex& turn_;
    std::string scene_id_, rig_, t_left_, t_right_;
    perception::LensModel lens_;
    perception::LensMap map_;
    core::SE3 T_B_left_, T_B_right_;
    size_t unpaired_ = 0;
};

}  // namespace ariadne::runtime::live
