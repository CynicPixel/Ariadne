// ---------------------------------------------------------------------------
// ariadne_bridge::RuntimeLink (design LIVE_RUNTIME_L1 D150). The only ROS code of the live runtime.
//
// Loaded into the SAME component container as ros_gz_bridge::RosGzBridge, with intra-process comms, so a bridged
// message reaches this node without a DDS hop (D40: every measured loss of the old recording path was on that hop).
//   GZ -> ROS topics of the bridge YAML (except the skipped ones, e.g. /clock): typed subscription -> CDR bytes
//     (rclcpp::Serialization, the bytes a rosbag2 MCAP would hold) -> one link frame on the Unix socket.
//   ROS -> GZ topics (the commands): link frame from the runtime -> deserialize -> typed publish -> the bridge.
// It never interprets a message. The runtime decodes them with its own ROS-free decoder (runtime/replay/ros_cdr).
//
// Parameters: bridge_config (the YAML the bridge loads), socket_path, ready_file, skip_topics (string array).
// Readiness: when every forwarded topic has a publisher and every command topic a subscriber, a #ready frame goes
// to the runtime and ready_file is written; the launcher unpauses the world only then.
// Lockstep (D165): a service entry of the YAML (ros_gz_interfaces/srv/ControlWorld, bridged to /world/<w>/control)
// gives a client; a "#run_to" frame from the runtime (int64 sim ns, little-endian) becomes a request with only
// run_to_sim_time set: gz runs to that time and pauses itself. The answer goes back as "#run_to_ack" JSON with the sim
// clock at the moment of the request (/clock, subscribed here, not forwarded): the time the world was paused at.
// ---------------------------------------------------------------------------
#include "runtime/live/link_frame.hpp"

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ros_gz_interfaces/srv/control_world.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace ariadne_bridge {

namespace live = ariadne::runtime::live;

class RuntimeLink : public rclcpp::Node {
public:
    explicit RuntimeLink(const rclcpp::NodeOptions& opts) : rclcpp::Node("ariadne_runtime_link", opts) {
        const std::string config = declare_parameter<std::string>("bridge_config");
        const std::string socket_path = declare_parameter<std::string>("socket_path");
        ready_file_ = declare_parameter<std::string>("ready_file", "");
        const auto skip = declare_parameter<std::vector<std::string>>("skip_topics", std::vector<std::string>{"/clock"});
        const std::set<std::string> skipped(skip.begin(), skip.end());

        fd_ = live::Connect(socket_path, std::chrono::seconds(120));
        RCLCPP_INFO(get_logger(), "connected to the runtime at %s", socket_path.c_str());

        for (const auto& e : YAML::LoadFile(config)) {
            if (e["service_name"]) {
                const std::string type = e["ros_type_name"].as<std::string>();
                if (type != "ros_gz_interfaces/srv/ControlWorld") {
                    throw std::runtime_error("RuntimeLink: unsupported service type " + type);
                }
                control_ = create_client<ros_gz_interfaces::srv::ControlWorld>(e["service_name"].as<std::string>());
                clock_sub_ = create_subscription<rosgraph_msgs::msg::Clock>(
                    "/clock", rclcpp::QoS(10), [this](const rosgraph_msgs::msg::Clock::SharedPtr m) {
                        clock_ns_ = int64_t(m->clock.sec) * 1000000000 + m->clock.nanosec;
                    });
                continue;
            }
            const std::string topic = e["ros_topic_name"].as<std::string>();
            const std::string type = e["ros_type_name"].as<std::string>();
            const std::string dir = e["direction"].as<std::string>();
            if (dir == "GZ_TO_ROS") {
                if (skipped.count(topic)) continue;
                Subscribe(topic, type);
            } else if (dir == "ROS_TO_GZ") {
                if (type != "geometry_msgs/msg/PoseArray") {
                    throw std::runtime_error("RuntimeLink: unsupported command type " + type + " on " + topic);
                }
                cmd_pubs_[topic] = create_publisher<geometry_msgs::msg::PoseArray>(topic, rclcpp::QoS(100).reliable());
            } else {
                throw std::runtime_error("RuntimeLink: unsupported direction " + dir);
            }
        }
        reader_ = std::thread([this] { ReadCommands(); });
        ready_timer_ = create_wall_timer(std::chrono::milliseconds(200), [this] { CheckReady(); });
        stats_timer_ = create_wall_timer(std::chrono::seconds(5), [this] { SendStats(); });
    }

    ~RuntimeLink() override {
        stop_ = true;
        if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR);
        if (reader_.joinable()) reader_.join();
        if (fd_ >= 0) ::close(fd_);
    }

private:
    template <typename T>
    void SubscribeTyped(const std::string& topic) {
        auto ser = std::make_shared<rclcpp::Serialization<T>>();
        // Deep enough that a slow socket write does not overflow the intra-process buffer during a stall;
        // any loss still shows as a stamp-grid gap in the runtime's per-topic accounting.
        subs_.push_back(create_subscription<T>(topic, rclcpp::QoS(500).reliable(),
                                               [this, topic, ser](const typename T::SharedPtr m) {
                                                   rclcpp::SerializedMessage sm;
                                                   ser->serialize_message(m.get(), &sm);
                                                   const auto& r = sm.get_rcl_serialized_message();
                                                   Send(topic, r.buffer, r.buffer_length);
                                               }));
        received_[topic] = 0;
    }

    void Subscribe(const std::string& topic, const std::string& type) {
        if (type == "sensor_msgs/msg/Image") SubscribeTyped<sensor_msgs::msg::Image>(topic);
        else if (type == "sensor_msgs/msg/CameraInfo") SubscribeTyped<sensor_msgs::msg::CameraInfo>(topic);
        else if (type == "sensor_msgs/msg/Imu") SubscribeTyped<sensor_msgs::msg::Imu>(topic);
        else if (type == "geometry_msgs/msg/PoseStamped") SubscribeTyped<geometry_msgs::msg::PoseStamped>(topic);
        else if (type == "rosgraph_msgs/msg/Clock") SubscribeTyped<rosgraph_msgs::msg::Clock>(topic);
        else if (type == "std_msgs/msg/String") SubscribeTyped<std_msgs::msg::String>(topic);
        else throw std::runtime_error("RuntimeLink: unsupported sensor type " + type + " on " + topic);
        forwarded_.push_back(topic);
    }

    void Send(const std::string& topic, const uint8_t* data, size_t n) {
        std::lock_guard<std::mutex> lk(write_mu_);
        if (topic[0] != '#') ++received_[topic];
        if (!live::WriteFrame(fd_, topic, data, n)) {
            if (!lost_peer_.exchange(true)) RCLCPP_ERROR(get_logger(), "runtime link closed: messages are now dropped");
            ++dropped_[topic];
        }
    }

    void ReadCommands() {
        rclcpp::Serialization<geometry_msgs::msg::PoseArray> ser;
        live::LinkFrame f;
        try {
            while (!stop_ && live::ReadFrame(fd_, f)) {
                if (f.topic == "#run_to") {
                    RunTo(f);
                    continue;
                }
                const auto it = cmd_pubs_.find(f.topic);
                if (it == cmd_pubs_.end()) {
                    RCLCPP_ERROR(get_logger(), "runtime sent a frame for unknown topic '%s'", f.topic.c_str());
                    continue;
                }
                rclcpp::SerializedMessage sm(f.payload.size());
                auto& r = sm.get_rcl_serialized_message();
                std::memcpy(r.buffer, f.payload.data(), f.payload.size());
                r.buffer_length = f.payload.size();
                geometry_msgs::msg::PoseArray msg;
                ser.deserialize_message(&sm, &msg);
                it->second->publish(msg);
                ++commands_;
            }
        } catch (const std::exception& e) {
            if (!stop_) RCLCPP_ERROR(get_logger(), "command link: %s", e.what());
        }
    }

    void RunTo(const live::LinkFrame& f) {
        int64_t t = 0;
        if (!control_ || f.payload.size() != sizeof t) {
            RCLCPP_ERROR(get_logger(), "#run_to without a control service, or a bad payload");
            return;
        }
        std::memcpy(&t, f.payload.data(), sizeof t);
        auto req = std::make_shared<ros_gz_interfaces::srv::ControlWorld::Request>();
        req->world_control.run_to_sim_time.sec = int32_t(t / 1000000000);
        req->world_control.run_to_sim_time.nanosec = uint32_t(t % 1000000000);
        const int64_t paused = clock_ns_.load();
        control_->async_send_request(req, [this, t, paused](rclcpp::Client<ros_gz_interfaces::srv::ControlWorld>::SharedFuture r) {
            const std::string s = "{\"run_to_ns\":" + std::to_string(t) + ",\"paused_clock_ns\":" + std::to_string(paused) +
                                  ",\"accepted\":" + (r.get()->success ? "true" : "false") + "}";
            Send("#run_to_ack", reinterpret_cast<const uint8_t*>(s.data()), s.size());
        });
    }

    void CheckReady() {
        if (ready_) return;
        if (control_ && !control_->service_is_ready()) return;
        std::ostringstream js;
        js << "{\"forwarded\":[";
        for (size_t i = 0; i < forwarded_.size(); ++i) {
            if (count_publishers(forwarded_[i]) == 0) return;
            js << (i ? "," : "") << "\"" << forwarded_[i] << "\"";
        }
        js << "],\"commands\":[";
        bool first = true;
        for (const auto& [t, p] : cmd_pubs_) {
            if (p->get_subscription_count() == 0) return;
            js << (first ? "" : ",") << "\"" << t << "\"";
            first = false;
        }
        js << "]}";
        const std::string s = js.str();
        Send("#ready", reinterpret_cast<const uint8_t*>(s.data()), s.size());
        if (!ready_file_.empty()) std::ofstream(ready_file_) << s << "\n";
        ready_ = true;
        RCLCPP_INFO(get_logger(), "ready: %s", s.c_str());
    }

    void SendStats() {
        std::ostringstream js;
        std::unique_lock<std::mutex> lk(write_mu_);   // the counters are written under it (Send)
        js << "{\"commands_published\":" << commands_.load() << ",\"received\":{";
        bool first = true;
        for (const auto& [t, n] : received_) {
            js << (first ? "" : ",") << "\"" << t << "\":" << n;
            first = false;
        }
        js << "},\"dropped_link_closed\":{";
        first = true;
        for (const auto& [t, n] : dropped_) {
            js << (first ? "" : ",") << "\"" << t << "\":" << n;
            first = false;
        }
        js << "}}";
        const std::string s = js.str();
        lk.unlock();
        Send("#stats", reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    int fd_ = -1;
    std::string ready_file_;
    std::mutex write_mu_;
    std::atomic<bool> stop_{false}, lost_peer_{false};
    bool ready_ = false;
    std::atomic<long> commands_{0};
    std::vector<std::string> forwarded_;
    std::map<std::string, long> received_, dropped_;   // guarded by write_mu_
    std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr> cmd_pubs_;
    std::thread reader_;
    rclcpp::Client<ros_gz_interfaces::srv::ControlWorld>::SharedPtr control_;   // lockstep only
    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr clock_sub_;
    std::atomic<int64_t> clock_ns_{-1};
    rclcpp::TimerBase::SharedPtr ready_timer_, stats_timer_;
};

}  // namespace ariadne_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(ariadne_bridge::RuntimeLink)
