// ---------------------------------------------------------------------------
// ariadne_live: the live closed-loop runtime (design LIVE_RUNTIME_L1, D150-D154). ROS-free.
//
//   ariadne_live --scene DIR --perception-config P --slam-config S --mission-config M --live-config L --out DIR
//                (--link SOCKET | --replay INPUTS.mcap)
//
// --link: listens on SOCKET for the ROS edge (ros2/ariadne_bridge RuntimeLink), takes (topic, CDR) frames, reorders
//   them by stamp behind a watermark, records exactly what it consumed (inputs.mcap) and sends commands back.
// --replay: feeds a previous run's inputs.mcap through the same code in the recorded order; no command leaves.
//
// Per agent, one thread runs the replay estimator path unchanged (RunStereoVo over a LiveSimSource: the same
// decoder, lens, gray conversion and pairing as SimSource). The mission thread integrates keyframe depth into the
// 1.8 map (planning/mission/frontier_mission.hpp) and replans every replan_period_s of SIM time, only after BOTH
// agents' estimators have passed that time, so a replan sees the same inputs live and in replay. At each replan the
// deterministic layer lists the reachable frontier clusters (the candidates), the chooser picks one (classical =
// nearest), and the choice becomes a follow() command along the A* path (D151). Every decision is a trace record;
// live records are executed: true and carry the command id (agent, stamp) the mover answers on mover_state.
//
// Outputs (run dir): trace.jsonl, commands.jsonl, mover_state.jsonl, telemetry.jsonl (console), snapshots/,
// raw/agent_<k>/... (the runner's own outputs, estimate_body.tum included), inputs.mcap (live), results.yaml,
// manifest.yaml.
// ---------------------------------------------------------------------------
#include "planning/mission/frontier_mission.hpp"
#include "runtime/live/link_frame.hpp"
#include "runtime/live/live_source.hpp"
#include "runtime/manifest/build_identity.hpp"
#include "runtime/manifest/determinism.hpp"
#include "runtime/manifest/opencv_identity.hpp"
#include "runtime/manifest/run_manifest.hpp"
#include "runtime/replay/ros_cdr_writer.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace ariadne;
using planning::Cell;
namespace live = runtime::live;
namespace mis = planning::mission;

namespace {

struct LiveConfig {
    std::string name, chooser;
    double end_s = 0, max_command_m = 0, eps_cells = 1, telemetry_hz = 5, min_goal_m = 0, escape_m = 0,
           max_standoff_m = 0, refused_radius_m = 0;
    double autonomous_threshold = 1.01;   // §23 risk gate (AGENTIC_7 D158); > 1 = no gate
    double llm_timeout_s = 0;              // chooser llm: wall seconds to wait for the orchestrator
    bool face_frontier = true;
    int hold_back_ms = 3000;
};

LiveConfig LoadLive(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    LiveConfig c;
    c.name = n["name"].as<std::string>();
    c.end_s = n["end_s"].as<double>();
    c.chooser = n["chooser"].as<std::string>();
    c.max_command_m = n["command"]["max_length_m"].as<double>();
    c.eps_cells = n["command"]["simplify_eps_cells"].as<double>();
    c.face_frontier = n["command"]["face_frontier"].as<bool>();
    c.min_goal_m = n["command"]["min_goal_m"].as<double>();
    c.escape_m = n["command"]["escape_m"].as<double>();
    c.max_standoff_m = n["command"]["max_standoff_m"].as<double>();
    c.refused_radius_m = n["command"]["refused_radius_m"].as<double>();
    c.hold_back_ms = n["reorder"]["hold_back_ms"].as<int>();
    c.telemetry_hz = n["telemetry_hz"].as<double>();
    if (c.chooser != "classical" && c.chooser != "llm") throw std::invalid_argument("live config: chooser classical|llm");
    c.autonomous_threshold = n["risk_gate"]["autonomous_threshold"].as<double>();
    if (c.chooser == "llm") c.llm_timeout_s = n["llm"]["timeout_s"].as<double>();
    if (!(c.end_s > 0) || !(c.max_command_m > 0) || !(c.telemetry_hz > 0)) throw std::invalid_argument("live config: bad values");
    return c;
}

/// One keyframe for the map, at the spool's scale (the 1.8 carving values assume 640x360 depth, D153).
struct KfRec {
    int agent = 0;
    uint64_t id = 0;
    int64_t t_ns = 0;
    core::SE3 T_O_C;
    cv::Mat Z, S;   // CV_32F, half resolution, fp16-rounded exactly as the spool stores them
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

cv::Mat HalfAsSpool(const cv::Mat& f32) {   // KeyframeSpool::WriteHalf + ReadHalf, without the PNG round trip
    cv::Mat d, h, back;
    cv::resize(f32, d, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    d.convertTo(h, CV_16F);
    h.convertTo(back, CV_32F);
    return back;
}

struct AgentLane {
    live::MessageQueue images;
    live::GtStore gt;
    std::mutex mu;
    std::deque<KfRec> kfs;                               // delivered by the VO thread, consumed by the mission
    std::vector<std::pair<int64_t, core::SE3>> cams;     // every frame's T_O_C (rectified left camera)
    std::vector<int64_t> kf_times;                       // every keyframe's stamp (estimator facts for the chooser)
    cv::Mat last_rgb;                                    // the latest keyframe's rectified RGB (observe_image)
    int64_t progress = -1;                               // stamp of the last frame the estimator finished
    bool done = false;
    size_t frames = 0, keyframes = 0, unpaired = 0;
    int64_t last_telemetry_ns = -1;
};

class Lines {   // a thread-safe JSONL file
public:
    explicit Lines(const fs::path& p) : f_(p) { f_.precision(10); }
    void Put(const std::string& s) {
        std::lock_guard<std::mutex> lk(mu_);
        f_ << s << "\n";
        f_.flush();
    }

private:
    std::mutex mu_;
    std::ofstream f_;
};

std::string Xy(double x, double y) {
    std::ostringstream o;
    o.precision(10);
    o << "[" << x << "," << y << "]";
    return o.str();
}

double YawOfCamera(const core::SE3& T_W_C) {   // heading of the optical axis (z) in the horizontal plane
    return std::atan2(T_W_C.R(1, 2), T_W_C.R(0, 2));
}

/// The A* path to `goal`, simplified (approxPolyDP, every chord re-checked on the traversable grid), in W, without
/// the start cell, cut at max_m. Empty if no path.
std::vector<std::pair<double, double>> CommandPath(const mis::Briefing& b, const cv::Mat& trav, const Cell& from,
                                                   const Cell& goal, double eps_cells, double max_m) {
    const planning::GridPath p = planning::AStar(trav, from, {goal});
    if (!p.found || p.cells.size() < 2) return {};
    std::vector<cv::Point> pts, simp;
    for (const auto& c : p.cells) pts.emplace_back(c.c, c.r);
    cv::approxPolyDP(pts, simp, eps_cells, false);
    bool chords_ok = true;
    for (size_t i = 1; i < simp.size() && chords_ok; ++i) {
        cv::LineIterator it(trav, simp[i - 1], simp[i], 8);
        for (int k = 0; k < it.count; ++k, ++it) {
            if (trav.at<uint8_t>(it.pos()) == 0) {
                chords_ok = false;
                break;
            }
        }
    }
    if (!chords_ok) simp = pts;   // never shortcut through a non-traversable cell
    std::vector<std::pair<double, double>> out;
    auto [px, py] = b.CellCentre(Cell{simp[0].y, simp[0].x});
    double left = max_m;
    for (size_t i = 1; i < simp.size() && left > 1e-9; ++i) {
        auto [x, y] = b.CellCentre(Cell{simp[i].y, simp[i].x});
        const double d = std::hypot(x - px, y - py);
        if (d > left) {
            x = px + (x - px) * left / d;
            y = py + (y - py) * left / d;
        }
        out.emplace_back(x, y);
        left -= std::min(d, left);
        px = x, py = y;
    }
    return out;
}

/// chooser: llm, live (AGENTIC_7 D156): one JSON line per decision request to the orchestrator's Unix socket; answers
/// come back as JSON lines keyed by decision_id, in any order.
class ChooserLink {
public:
    explicit ChooserLink(const std::string& path) : fd_(live::Connect(path, std::chrono::seconds(60))) {
        reader_ = std::thread([this] {
            std::string buf;
            char tmp[4096];
            for (;;) {
                const ssize_t n = ::recv(fd_, tmp, sizeof tmp, 0);
                if (n <= 0) break;
                buf.append(tmp, size_t(n));
                size_t nl;
                while ((nl = buf.find('\n')) != std::string::npos) {
                    std::string line = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    try {
                        const std::string id = YAML::Load(line)["decision_id"].as<std::string>();
                        std::lock_guard<std::mutex> lk(mu_);
                        answers_[id] = std::move(line);
                    } catch (const std::exception&) {
                    }
                    cv_.notify_all();
                }
            }
            std::lock_guard<std::mutex> lk(mu_);
            closed_ = true;
            cv_.notify_all();
        });
    }
    ~ChooserLink() {
        ::shutdown(fd_, SHUT_RDWR);
        reader_.join();
        ::close(fd_);
    }
    void Send(const std::string& line) {
        const std::string l = line + "\n";
        if (::send(fd_, l.data(), l.size(), MSG_NOSIGNAL) != ssize_t(l.size())) throw std::runtime_error("chooser: send");
    }
    /// The raw answer line, or "" on timeout / closed link.
    std::string Wait(const std::string& id, std::chrono::steady_clock::time_point until) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait_until(lk, until, [&] { return answers_.count(id) || closed_; });
        const auto it = answers_.find(id);
        return it == answers_.end() ? "" : it->second;
    }

private:
    int fd_;
    std::thread reader_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<std::string, std::string> answers_;
    bool closed_ = false;
};

/// chooser: llm, replay (AGENTIC_7 D161): the live trace's choices, keyed by (t_s text, agent): the chosen index and the
/// raw `"choice":{...}` tail, re-emitted verbatim. The model is never called in replay.
std::map<std::pair<std::string, int>, std::pair<int, std::string>> LoadRecordedChoices(const std::string& trace) {
    std::map<std::pair<std::string, int>, std::pair<int, std::string>> out;
    std::ifstream f(trace);
    if (!f) throw std::runtime_error("replay: cannot open the live trace " + trace);
    std::string line;
    while (std::getline(f, line)) {
        const YAML::Node n = YAML::Load(line);
        if (!n["choice"]) continue;
        const auto p = line.find(",\"choice\":");
        std::ostringstream t;
        t.precision(10);
        t << n["t_s"].as<double>();
        out[{t.str(), n["agent"].as<int>()}] = {n["chosen"].as<int>(), line.substr(p, line.size() - 1 - p)};
    }
    return out;
}

std::string JsonEscape(const std::string& s) {
    std::string o;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') o += '\\';
        if (ch == '\n') o += "\\n";
        else o += ch;
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string scene, mission_cfg_path, live_cfg_path, out, link_path, replay_path, chooser_socket, decisions_path;
        tools::StereoVoArgs va;
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
            const std::string v = argv[++i];
            if (k == "--scene") scene = v;
            else if (k == "--perception-config") va.perception_config = v;
            else if (k == "--slam-config") va.slam_config = v;
            else if (k == "--mission-config") mission_cfg_path = v;
            else if (k == "--live-config") live_cfg_path = v;
            else if (k == "--out") out = v;
            else if (k == "--link") link_path = v;
            else if (k == "--replay") replay_path = v;
            else if (k == "--chooser-socket") chooser_socket = v;   // chooser llm, live
            else if (k == "--decisions") decisions_path = v;        // chooser llm, replay: the live trace.jsonl
            else if (k == "--seed") va.seed = std::stoull(v);
            else throw std::invalid_argument("unknown argument " + k);
        }
        if (scene.empty() || mission_cfg_path.empty() || live_cfg_path.empty() || out.empty() ||
            va.perception_config.empty() || va.slam_config.empty() || link_path.empty() == replay_path.empty()) {
            throw std::invalid_argument("usage: ariadne_live --scene S --perception-config P --slam-config S "
                                        "--mission-config M --live-config L --out DIR (--link SOCK | --replay MCAP)");
        }
        const bool is_live = !link_path.empty();
        const LiveConfig lc = LoadLive(live_cfg_path);
        if (lc.chooser == "llm" && is_live && chooser_socket.empty()) throw std::invalid_argument("chooser llm needs --chooser-socket");
        if (lc.chooser == "llm" && !is_live && decisions_path.empty() == chooser_socket.empty())
            throw std::invalid_argument("chooser llm replay needs --decisions (the live trace) or --chooser-socket (a test)");
        const auto recorded = lc.chooser == "llm" && !decisions_path.empty() ? LoadRecordedChoices(decisions_path)
                                                              : std::map<std::pair<std::string, int>, std::pair<int, std::string>>{};
        const mis::MissionConfig mc = mis::LoadMissionConfig(mission_cfg_path);
        const mis::Briefing brief = mis::LoadBriefing(scene, mc, 2, "live");
        mis::FrontierMap map(mc, brief);
        const int64_t end_ns = int64_t(std::llround(lc.end_s * 1e9));
        const int64_t replan_ns = int64_t(std::llround(mc.replan_s * 1e9));

        const fs::path run_dir = fs::path(out) / (brief.scene_id + (is_live ? "-live-" : "-replay-") + tools::UtcNow("%Y%m%dT%H%M%SZ"));
        fs::create_directories(run_dir / "snapshots");
        fs::create_directories(run_dir / "raw");
        Lines trace(run_dir / "trace.jsonl"), commands(run_dir / "commands.jsonl"),
            mover_state(run_dir / "mover_state.jsonl"), telemetry(run_dir / "telemetry.jsonl"),
            link_log(run_dir / "link.jsonl");

        // --- topics (mission_live.json) -------------------------------------------------------------------
        const YAML::Node mj = YAML::LoadFile(scene + "/mission_live.json");
        std::map<std::string, std::pair<int, std::string>> route;   // topic -> (agent, role)
        std::map<int, std::string> cmd_topic;
        std::set<std::string> timed;
        for (const auto& a : mj["agents"]) {
            const int k = a["agent"].as<int>();
            const auto t = a["topics"];
            route[t["left_image"].as<std::string>()] = {k, "image"};
            route[t["right_image"].as<std::string>()] = {k, "image"};
            route[t["gt_pose"].as<std::string>()] = {k, "gt"};
            route[t["imu"].as<std::string>()] = {k, "imu"};
            route[t["left_info"].as<std::string>()] = {k, "info"};
            route[t["right_info"].as<std::string>()] = {k, "info"};
            route[a["live_topics"]["mover_state"].as<std::string>()] = {k, "mover_state"};
            cmd_topic[k] = a["live_topics"]["cmd"].as<std::string>();
            for (const char* r : {"left_image", "right_image", "gt_pose", "imu"}) timed.insert(t[r].as<std::string>());
        }
        if (cmd_topic.size() != 2) throw std::runtime_error("live: expected 2 agents in mission_live.json");

        AgentLane lanes[2];
        // The mover's answers (D153b): (agent, command stamp) -> accepted. Part of the recorded input, so the wait for
        // them below is deterministic in replay.
        std::map<std::pair<int, int64_t>, bool> answers;
        bool input_done = false;
        std::mutex progress_mu;
        std::condition_variable progress_cv;

        // --- the link (live) ------------------------------------------------------------------------------
        int fd = -1;
        std::mutex send_mu;
        if (is_live) {
            std::printf("ariadne_live: waiting for the ROS edge on %s\n", link_path.c_str());
            std::fflush(stdout);
            fd = live::ListenAndAccept(link_path);
            std::printf("ariadne_live: link up\n");
        }
        auto SendCommand = [&](int agent, const runtime::ros_cdr::PoseArray& m) {
            if (!is_live) return;
            const auto bytes = runtime::ros_cdr::EncodePoseArray(m);
            std::lock_guard<std::mutex> lk(send_mu);
            if (!live::WriteFrame(fd, cmd_topic.at(agent), bytes.data(), bytes.size())) {
                throw std::runtime_error("live: the ROS edge closed the link");
            }
        };

        // --- dispatch: every consumed message, in release order ----------------------------------------------
        live::ConsumedRecorder rec;
        if (is_live) rec.Open((run_dir / "inputs.mcap").string());
        std::atomic<bool> past_end{false};
        uint64_t seq = 0;
        long dropped_untyped = 0;
        auto Dispatch = [&](live::StreamMessage m) {
            if (past_end) return;
            const auto it = route.find(m.topic);
            if (it == route.end()) {
                ++dropped_untyped;
                return;
            }
            const auto [agent, role] = it->second;
            if (role == "image" && m.stamp_ns > end_ns) {   // the end of the run: identical cut live and in replay
                past_end = true;
                for (auto& l : lanes) l.images.Close();
                return;
            }
            m.seq = ++seq;
            rec.Write(m);
            if (role == "image") {
                lanes[agent].images.Push(std::move(m));
            } else if (role == "gt") {
                const auto p = runtime::ros_cdr::DecodePoseStamped(m.data->data(), m.data->size());
                runtime::TimedPose tp{p.header.stamp.ns(), {}};
                tp.T.R = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)
                             .normalized()
                             .toRotationMatrix();
                tp.T.t = core::Vec3(p.position.x, p.position.y, p.position.z);
                lanes[agent].gt.Add(tp);
            } else if (role == "mover_state") {
                const std::string js = runtime::ros_cdr::DecodeString(m.data->data(), m.data->size());
                const auto num = [&js](const std::string& key) -> long long {
                    const auto p = js.find("\"" + key + "\":");
                    return p == std::string::npos ? -1 : std::stoll(js.substr(p + key.size() + 3));
                };
                {
                    std::lock_guard<std::mutex> lk(progress_mu);
                    answers[{agent, int64_t(num("cmd_stamp_ns"))}] = js.find("\"accepted\":true") != std::string::npos;
                }
                progress_cv.notify_all();
                mover_state.Put("{\"agent\":" + std::to_string(agent) + ",\"seq\":" + std::to_string(m.seq) +
                                ",\"state\":" + js + "}");
                telemetry.Put("{\"type\":\"mover_state\",\"agent\":" + std::to_string(agent) + ",\"state\":" + js + "}");
            }
        };

        std::thread reader;
        long reorder_timeouts = 0;
        if (is_live) {
            reader = std::thread([&] {
                live::Reorderer ro(timed, std::chrono::milliseconds(lc.hold_back_ms));
                live::LinkFrame f;
                try {
                    while (live::ReadFrame(fd, f)) {
                        if (f.topic[0] == '#') {
                            link_log.Put("{\"control\":\"" + f.topic + "\",\"payload\":" +
                                         std::string(f.payload.begin(), f.payload.end()) + "}");
                            continue;
                        }
                        live::StreamMessage m;
                        m.topic = f.topic;
                        const auto it = route.find(f.topic);
                        const std::string role = it == route.end() ? "" : it->second.second;
                        m.type = role == "image" ? "sensor_msgs/msg/Image"
                                 : role == "gt" ? "geometry_msgs/msg/PoseStamped"
                                 : role == "imu" ? "sensor_msgs/msg/Imu"
                                 : role == "info" ? "sensor_msgs/msg/CameraInfo"
                                 : role == "mover_state" ? "std_msgs/msg/String" : "unknown";
                        auto data = std::make_shared<std::vector<uint8_t>>(std::move(f.payload));
                        m.stamp_ns = live::PeekStamp(m.type, *data);
                        m.data = std::move(data);
                        ro.Push(std::move(m), Dispatch);
                        if (past_end) break;
                    }
                    ro.Flush(Dispatch);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "ariadne_live: link reader: %s\n", e.what());
                }
                reorder_timeouts = ro.released_on_timeout();
                for (auto& l : lanes) l.images.Close();
                {
                    std::lock_guard<std::mutex> lk(progress_mu);
                    input_done = true;
                }
                progress_cv.notify_all();
            });
        } else {
            reader = std::thread([&] {
                runtime::RosbagReader bag;
                bag.Open(replay_path);
                const auto types = bag.topics();
                bag.Replay([&](const runtime::RosMessage& msg) {
                    live::StreamMessage m;
                    m.topic = std::string(msg.topic);
                    m.type = types.at(m.topic);
                    auto data = std::make_shared<std::vector<uint8_t>>(msg.data, msg.data + msg.size);
                    m.stamp_ns = live::PeekStamp(m.type, *data);
                    m.data = std::move(data);
                    Dispatch(std::move(m));
                });
                for (auto& l : lanes) l.images.Close();
                {
                    std::lock_guard<std::mutex> lk(progress_mu);
                    input_done = true;
                }
                progress_cv.notify_all();
            });
        }

        // --- per-agent estimators: the replay path, unchanged ---------------------------------------------------
        std::thread vo[2];
        std::unique_ptr<live::LiveSimSource> sources[2];
        std::mutex estimator_turn;   // D152a: one frame at a time through the estimators
        for (int k = 0; k < 2; ++k) {
            sources[k] = std::make_unique<live::LiveSimSource>(scene, k, lanes[k].images, lanes[k].gt, estimator_turn, "live");
            vo[k] = std::thread([&, k] {
                AgentLane& L = lanes[k];
                tools::StereoVoArgs a = va;
                a.out = (run_dir / "raw").string();
                a.tool = std::string("ariadne_live/agent") + std::to_string(k);
                a.late_ground_truth = [&L] { return L.gt.Snapshot(); };
                a.on_keyframe = [&, k](const tools::KeyframeView& kv) {
                    const runtime::KeyframeBundle& b = kv.bundle;
                    KfRec r;
                    r.agent = k;
                    r.id = kv.keyframe.id;
                    r.t_ns = kv.keyframe.t_ns;
                    r.T_O_C = b.T_WiC;
                    r.Z = HalfAsSpool(runtime::MaskedDepth32(b.depth, b.mask));
                    r.S = HalfAsSpool(runtime::MaskedDepth32(b.depth_sigma, b.mask));
                    // Spool camera: u' + 0.5 = 0.5 (u + 0.5).
                    r.fx = b.K.fx * 0.5;
                    r.fy = b.K.fy * 0.5;
                    r.cx = 0.5 * (b.K.cx + 0.5) - 0.5;
                    r.cy = 0.5 * (b.K.cy + 0.5) - 0.5;
                    std::lock_guard<std::mutex> lk(L.mu);
                    L.kf_times.push_back(r.t_ns);
                    L.last_rgb = b.rgb.clone();
                    L.kfs.push_back(std::move(r));
                    ++L.keyframes;
                };
                a.on_frame = [&, k](const tools::FrameView& fv) {
                    {
                        std::lock_guard<std::mutex> lk(L.mu);
                        L.cams.emplace_back(fv.t_ns, fv.T_O_C);
                        ++L.frames;
                    }
                    {
                        std::lock_guard<std::mutex> lk(progress_mu);
                        L.progress = fv.t_ns;
                    }
                    progress_cv.notify_all();
                    if (L.last_telemetry_ns < 0 || fv.t_ns - L.last_telemetry_ns >= int64_t(1e9 / lc.telemetry_hz)) {
                        L.last_telemetry_ns = fv.t_ns;
                        const core::SE3 T = brief.T_W_O[size_t(k)] * fv.T_O_C;
                        std::ostringstream o;
                        o.precision(8);
                        o << "{\"type\":\"pose\",\"t_s\":" << fv.t_ns * 1e-9 << ",\"agent\":" << k << ",\"xy\":"
                          << Xy(T.t.x(), T.t.y()) << ",\"yaw\":" << YawOfCamera(T) << "}";
                        telemetry.Put(o.str());
                    }
                };
                try {
                    tools::RunStereoVo(*sources[k], a, argc, argv);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "ariadne_live: agent %d estimator: %s\n", k, e.what());
                }
                {
                    std::lock_guard<std::mutex> lk(progress_mu);
                    L.done = true;
                    L.unpaired = sources[k]->unpaired();
                }
                progress_cv.notify_all();
            });
        }

        // --- the mission: barrier -> integrate -> replan -> candidates -> choose -> command ----------------------
        std::unique_ptr<ChooserLink> chooser;
        if (lc.chooser == "llm" && !chooser_socket.empty()) {   // live; or replay against a test orchestrator
            fs::create_directories(run_dir / "approvals");
            chooser = std::make_unique<ChooserLink>(chooser_socket);
        }
        int replans = 0, commands_sent = 0, answer_timeouts = 0;
        size_t integrated[2] = {0, 0}, rays = 0;
        std::vector<std::pair<int, std::pair<double, double>>> refused_goals;   // (agent, goal xy): never re-chosen
        std::map<int, std::pair<double, double>> last_goal;                      // per agent, the last commanded goal
        for (int64_t T = replan_ns; T <= end_ns; T += replan_ns) {
            {
                std::unique_lock<std::mutex> lk(progress_mu);
                progress_cv.wait(lk, [&] {
                    return (lanes[0].done || lanes[0].progress >= T) && (lanes[1].done || lanes[1].progress >= T);
                });
                if (lanes[0].done && lanes[1].done && lanes[0].progress < T && lanes[1].progress < T) break;
                // The previous decision's outcome is an input to this one: wait for both answers to the commands sent
                // at T - replan (live: bounded wait, a timeout is counted; replay: they are in the recording).
                if (T > replan_ns) {
                    const auto have = [&] {
                        return input_done || (answers.count({0, T - replan_ns}) && answers.count({1, T - replan_ns}));
                    };
                    if (is_live) {
                        if (!progress_cv.wait_for(lk, std::chrono::seconds(30), have)) ++answer_timeouts;
                    } else {
                        progress_cv.wait(lk, have);
                    }
                    for (int k = 0; k < 2; ++k) {
                        const auto a = answers.find({k, T - replan_ns});
                        if (a != answers.end() && !a->second && last_goal.count(k)) refused_goals.push_back({k, last_goal[k]});
                    }
                }
            }
            // Keyframes stamped before T, merged across agents in (t, agent, id) order.
            std::vector<KfRec> batch;
            for (auto& L : lanes) {
                std::lock_guard<std::mutex> lk(L.mu);
                while (!L.kfs.empty() && L.kfs.front().t_ns < T) {
                    batch.push_back(std::move(L.kfs.front()));
                    L.kfs.pop_front();
                }
            }
            std::sort(batch.begin(), batch.end(), [](const KfRec& a, const KfRec& b) {
                return std::tie(a.t_ns, a.agent, a.id) < std::tie(b.t_ns, b.agent, b.id);
            });
            for (const KfRec& r : batch) {
                rays += map.Integrate(brief.T_W_O[size_t(r.agent)] * r.T_O_C, r.Z, r.S, r.fx, r.fy, r.cx, r.cy);
                ++integrated[r.agent];
            }
            const mis::ReplanMaps maps = map.Replan();
            const double ts = T * 1e-9;
            const std::string snap = "snapshots/slab_" + std::to_string(int64_t(std::llround(ts * 1000))) + ".png";
            cv::imwrite((run_dir / snap).string(), maps.slab);
            {
                std::ostringstream o;
                o.precision(6);
                o << "{\"type\":\"snapshot\",\"t_s\":" << ts << ",\"png\":\"" << snap << "\",\"egress_xy\":";
                if (maps.egress.found) {   // the egress route at rho_responder (invariant 4), for the console
                    o << "[";
                    for (size_t i = 0; i < maps.egress.cells.size(); ++i) {
                        const auto [x, y] = brief.CellCentre(maps.egress.cells[i]);
                        o << (i ? "," : "") << Xy(x, y);
                    }
                    o << "]";
                } else {
                    o << "null";
                }
                o << "}";
                telemetry.Put(o.str());
            }
            ++replans;
            // Phase 1, per agent: the deterministic layer's candidate set, its risk (§23), the classical choice.
            struct Pending {
                bool active = false;
                std::string head;                  // the trace record up to (not including) "chosen"
                Cell ac;
                cv::Mat plan;
                std::vector<mis::Candidate> cands;   // ELIGIBLE only (path >= min_goal_m), nearest first
                std::vector<double> risk;
                int classical = -1;
                std::string agent_json;            // this agent's part of the joint request (without the closing brace)
            } P[2];
            const double rho_a = mc.rho_a, rho_r = mc.rho_r;
            const int free_cells = cv::countNonZero(maps.slab == planning::kFree),
                      occ_cells = cv::countNonZero(maps.slab == planning::kOccupied),
                      unknown_cells = int(maps.slab.total()) - free_cells - occ_cells;
            for (int k = 0; k < 2; ++k) {
                std::ostringstream o;
                o.precision(10);
                o << "{\"t_s\":" << ts << ",\"agent\":" << k << ",\"mode\":\"" << (is_live ? "live" : "replay")
                  << "\",\"executed\":" << (is_live ? "true" : "false") << ",\"decision\":\"frontier\",\"chooser\":\""
                  << lc.chooser << "\",";
                // The vision-derived inputs of this decision: the map built from keyframe depth, the agent's estimate.
                bool have_pose = false;
                core::SE3 T_O_C;
                size_t frames = 0, kfs = 0;
                int64_t last_kf = -1;
                {
                    std::lock_guard<std::mutex> lk(lanes[k].mu);
                    for (auto it = lanes[k].cams.rbegin(); it != lanes[k].cams.rend(); ++it) {
                        if (it->first < T) {
                            T_O_C = it->second;
                            have_pose = true;
                            break;
                        }
                    }
                    for (const auto& [t, Tc] : lanes[k].cams) frames += t < T;
                    for (int64_t t : lanes[k].kf_times) {
                        if (t < T) ++kfs, last_kf = t;
                    }
                }
                std::ostringstream vision;
                vision.precision(10);
                vision << "{\"keyframes_integrated\":[" << integrated[0] << "," << integrated[1] << "],\"rays\":" << rays
                       << ",\"free_cells\":" << free_cells << ",\"occupied_cells\":" << occ_cells << ",\"unknown_cells\":"
                       << unknown_cells << ",\"frontier_clusters\":" << maps.frontiers.clusters.size()
                       << ",\"rejected_small\":" << maps.frontiers.rejected_small << ",\"egress_found\":"
                       << (maps.egress.found ? "true" : "false") << ",\"snapshot_png\":\"" << snap << "\"";
                if (!have_pose) {
                    o << "\"vision\":" << vision.str() << "},\"result\":\"no decision: no frame yet\"}";
                    trace.Put(o.str());
                    continue;
                }
                const core::SE3 T_W_C = brief.T_W_O[size_t(k)] * T_O_C;
                const Cell ac = brief.ToCell(T_W_C.t.x(), T_W_C.t.y());
                vision << ",\"agent_xy_m\":" << Xy(T_W_C.t.x(), T_W_C.t.y()) << ",\"agent_cell\":[" << ac.r << ","
                       << ac.c << "]}";
                o << "\"vision\":" << vision.str() << ",";
                Pending& pd = P[k];
                pd.active = true;
                pd.ac = ac;
                // Standoff candidates over the conservative plan space (D153b), minus goals the mover refused before,
                // minus those nearer than min_goal_m (never presented: AGENTIC_7 review).
                pd.plan = map.PlanSpace(maps, ac, lc.escape_m);
                int excluded_refused = 0, excluded_near = 0;
                for (const auto& c : map.StandoffCandidates(maps, pd.plan, ac, lc.max_standoff_m)) {
                    const auto [gx, gy] = brief.CellCentre(c.goal);
                    bool refused = false;
                    for (const auto& [ra, rg] : refused_goals) {
                        refused = refused || (ra == k && std::hypot(rg.first - gx, rg.second - gy) < lc.refused_radius_m);
                    }
                    if (refused) ++excluded_refused;
                    else if (c.path_cells * brief.gs.cell < lc.min_goal_m) ++excluded_near;
                    else pd.cands.push_back(c);
                }
                // §23 risk (D158): the narrowest gap the agent cannot avoid on the way, against what a responder needs.
                const cv::Mat B = map.Bottleneck(maps, pd.plan, ac, lc.escape_m);
                std::ostringstream cj;
                cj.precision(6);
                cj << "[";
                for (size_t i = 0; i < pd.cands.size(); ++i) {
                    const auto& c = pd.cands[i];
                    // The widest way to ARRIVE within escape_m of the goal: the standoff goal itself sits at rho_agent
                    // from unknown space BY CONSTRUCTION, so its own clearance measures the placement rule, not a gap
                    // (first check, L1 45 s recording: all 58 candidates 0.875-0.925, bottleneck = the goal cell).
                    double bott = -1;
                    const int rad = int(std::ceil(lc.escape_m / brief.gs.cell));
                    for (int dy = -rad; dy <= rad; ++dy) {
                        for (int dx = -rad; dx <= rad; ++dx) {
                            const int y = c.goal.r + dy, x = c.goal.c + dx;
                            if (dx * dx + dy * dy > rad * rad || y < 0 || x < 0 || y >= B.rows || x >= B.cols) continue;
                            bott = std::max(bott, B.at<double>(y, x));
                        }
                    }
                    const double r = std::isfinite(bott) ? std::clamp((rho_r - bott) / (rho_r - rho_a), 0.0, 1.0) : 0.0;
                    pd.risk.push_back(r);
                    if (pd.classical < 0 && r <= lc.autonomous_threshold) pd.classical = int(i);
                    const auto [gx, gy] = brief.CellCentre(c.goal);
                    std::ostringstream why;
                    why.precision(3);
                    if (std::isfinite(bott)) why << "narrowest unavoidable gap " << bott << " m; a responder needs " << rho_r << " m";
                    else why << "goal inside the agent's own escape disc";
                    cj << (i ? "," : "") << "{\"index\":" << i << ",\"cluster\":" << c.cluster << ",\"path_m\":"
                       << c.path_cells * brief.gs.cell << ",\"goal_xy_m\":" << Xy(gx, gy) << ",\"standoff_m\":"
                       << c.standoff_m << ",\"centroid_xy_m\":" << Xy(c.centroid_x, c.centroid_y) << ",\"cells\":"
                       << c.cells << ",\"risk\":" << r << ",\"risk_reason\":\"" << why.str() << "\",\"needs_approval\":"
                       << (r > lc.autonomous_threshold ? "true" : "false") << "}";
                }
                cj << "]";
                o << "\"candidates\":" << cj.str() << ",\"candidates_total\":" << pd.cands.size()
                  << ",\"excluded_refused\":" << excluded_refused << ",\"excluded_near\":" << excluded_near
                  << ",\"autonomous_threshold\":" << lc.autonomous_threshold << ",\"classical_choice\":" << pd.classical
                  << ",";
                pd.head = o.str();
                {
                    std::ostringstream q;
                    q.precision(10);
                    q << "{\"agent\":" << k << ",\"vision\":" << vision.str() << ",\"estimator\":{\"frames\":"
                      << frames << ",\"keyframes\":" << kfs << ",\"since_last_keyframe_s\":"
                      << (last_kf < 0 ? -1.0 : (T - last_kf) * 1e-9) << "},\"candidates\":" << cj.str()
                      << ",\"classical_choice\":" << pd.classical;
                    pd.agent_json = q.str();
                }
            }
            // Phase 2: choose, jointly for both agents (AGENTIC_7 §0: Co-NavGPT's joint assignment; exactly-once).
            int picks[2] = {P[0].classical, P[1].classical};
            std::string choice[2];
            // Exactly-once (ALGORITHMS line 1165): agent 1 does not take agent 0's cluster while another eligible one
            // below the threshold exists. Applied to the classical pick, and to the LLM's in validation below.
            const auto other_cluster = [&](int k, int taken_cluster) {
                for (size_t i = 0; i < P[k].cands.size(); ++i) {
                    if (P[k].risk[i] <= lc.autonomous_threshold && P[k].cands[i].cluster != taken_cluster) return int(i);
                }
                return -1;
            };
            const auto cluster_of = [&](int k, int i) { return i >= 0 ? P[k].cands[size_t(i)].cluster : -1; };
            if (P[0].active && P[1].active && picks[0] >= 0 && picks[1] >= 0 &&
                cluster_of(0, picks[0]) == cluster_of(1, picks[1])) {
                const int alt = other_cluster(1, cluster_of(0, picks[0]));
                if (alt >= 0) picks[1] = alt;
            }
            if (lc.chooser == "classical") {
                for (auto& c : choice) c = ",\"choice\":{\"source\":\"classical\"}";
            } else if (!is_live && chooser_socket.empty()) {
                std::ostringstream t;
                t.precision(10);
                t << ts;
                for (int k = 0; k < 2; ++k) {
                    if (!P[k].active) continue;
                    const auto it = recorded.find({t.str(), k});
                    if (it == recorded.end()) throw std::runtime_error("replay: no recorded choice at t=" + t.str());
                    picks[k] = it->second.first;
                    choice[k] = it->second.second;
                }
            } else {
                // The joint request: both agents' facts and candidates, plus the images the model may look at, all
                // rendered here with OpenCV (the orchestrator image has no OpenCV).
                const std::string did = "t" + std::to_string(T / 1000000) + "ms";
                const fs::path ddir = run_dir / "decisions";
                fs::create_directories(ddir);
                cv::Mat img;
                cv::Mat lut(1, 256, CV_8UC3, cv::Scalar(128, 128, 128));
                lut.at<cv::Vec3b>(planning::kFree) = {255, 255, 255};
                lut.at<cv::Vec3b>(planning::kOccupied) = {0, 0, 0};
                cv::Mat slab3;
                cv::cvtColor(maps.slab, slab3, cv::COLOR_GRAY2BGR);
                cv::LUT(slab3, lut, img);
                cv::flip(img, img, 0);   // row 0 = south in the grid; north up in the picture
                const int scale = 3;
                cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_NEAREST);
                const auto px = [&](const Cell& c) { return cv::Point(c.c * scale + scale / 2, (maps.slab.rows - 1 - c.r) * scale + scale / 2); };
                for (size_t i = 1; i < maps.egress.cells.size(); ++i)
                    cv::line(img, px(maps.egress.cells[i - 1]), px(maps.egress.cells[i]), {255, 80, 0}, 2);
                const cv::Scalar col[2] = {{0, 0, 230}, {0, 160, 0}};
                for (int k = 0; k < 2; ++k) {
                    if (!P[k].active) continue;
                    for (size_t i = 0; i < P[k].cands.size(); ++i) {
                        const cv::Point g = px(P[k].cands[i].goal);
                        cv::drawMarker(img, g, col[k], cv::MARKER_TILTED_CROSS, 10, 2);
                        cv::putText(img, "a" + std::to_string(k) + ":c" + std::to_string(i), g + cv::Point(5, -5),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.4, col[k], 1, cv::LINE_AA);
                    }
                    cv::circle(img, px(P[k].ac), 7, col[k], cv::FILLED);
                    cv::putText(img, "agent " + std::to_string(k), px(P[k].ac) + cv::Point(9, 4), cv::FONT_HERSHEY_SIMPLEX,
                                0.45, col[k], 1, cv::LINE_AA);
                }
                const fs::path map_png = fs::absolute(ddir / (did + "_map.png"));
                cv::imwrite(map_png.string(), img);
                std::ostringstream q;
                q.precision(10);
                q << "{\"decision_id\":\"" << did << "\",\"t_s\":" << ts << ",\"map_png\":\"" << map_png.string()
                  << "\",\"egress\":{\"found\":" << (maps.egress.found ? "true" : "false") << ",\"length_m\":"
                  << maps.egress.length_cells * brief.gs.cell << ",\"exit\":" << maps.egress.goal_index
                  << ",\"clearance_m\":" << rho_r << "},\"autonomous_threshold\":" << lc.autonomous_threshold
                  << ",\"approvals_dir\":\"" << fs::absolute(run_dir / "approvals").string() << "\",\"agents\":[";
                bool first = true;
                for (int k = 0; k < 2; ++k) {
                    if (!P[k].active) continue;
                    std::string kf_png;
                    {
                        std::lock_guard<std::mutex> lk(lanes[k].mu);
                        if (!lanes[k].last_rgb.empty()) {
                            kf_png = fs::absolute(ddir / (did + "_a" + std::to_string(k) + "_kf.png")).string();
                            cv::Mat half, bgr;
                            cv::resize(lanes[k].last_rgb, half, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                            if (half.channels() == 3) cv::cvtColor(half, bgr, cv::COLOR_RGB2BGR);
                            else bgr = half;
                            cv::imwrite(kf_png, bgr);
                        }
                    }
                    q << (first ? "" : ",") << P[k].agent_json << ",\"keyframe_png\":"
                      << (kf_png.empty() ? "null" : "\"" + kf_png + "\"") << "}";
                    first = false;
                }
                q << "]}";
                chooser->Send(q.str());
                const std::string ans = chooser->Wait(did, std::chrono::steady_clock::now() +
                                                                std::chrono::milliseconds(int64_t(lc.llm_timeout_s * 1000)));
                YAML::Node assigned;
                std::string top_problem;
                if (ans.empty()) {
                    top_problem = "timeout or no orchestrator";
                } else {
                    try {
                        assigned = YAML::Load(ans)["assignments"];
                        if (!assigned || assigned.IsNull()) top_problem = "no assignment recorded";
                    } catch (const std::exception& e) {
                        top_problem = std::string("malformed answer: ") + e.what();
                    }
                }
                // Per agent: a bad pick for one agent does not discard the other's.
                int want[2] = {-2, -2};
                std::string approval_id[2], validation[2], source[2], approval_raw[2];
                for (int k = 0; k < 2; ++k) {
                    if (!P[k].active) continue;
                    source[k] = "orchestrator", validation[k] = "ok";
                    if (!top_problem.empty()) {
                        source[k] = "fallback", validation[k] = top_problem;
                        continue;
                    }
                    for (const auto& e : assigned) {
                        if (e["agent"] && e["agent"].as<int>() == k) {
                            want[k] = e["candidate"] ? e["candidate"].as<int>() : -2;
                            approval_id[k] = e["approval_id"] ? e["approval_id"].as<std::string>() : "";
                        }
                    }
                    if (want[k] < -1 || want[k] >= int(P[k].cands.size())) {
                        source[k] = "fallback", validation[k] = "missing or out of the candidate set";
                    } else if (want[k] >= 0 && P[k].risk[size_t(want[k])] > lc.autonomous_threshold) {
                        const std::string expect = did + "-a" + std::to_string(k) + "-c" + std::to_string(want[k]);
                        std::ifstream f(run_dir / "approvals" / (expect + ".json"));
                        approval_raw[k].assign(std::istreambuf_iterator<char>(f), {});
                        bool approved = false;
                        try {
                            approved = !approval_raw[k].empty() && approval_id[k] == expect &&
                                       YAML::Load(approval_raw[k])["verdict"].as<std::string>() == "approved";
                        } catch (const std::exception&) {
                        }
                        if (!approved) source[k] = "fallback", validation[k] = "above the risk threshold without an approval";
                    }
                    if (source[k] == "orchestrator") picks[k] = want[k];
                }
                if (P[0].active && P[1].active && picks[0] >= 0 && picks[1] >= 0 && source[1] == "orchestrator" &&
                    cluster_of(0, picks[0]) == cluster_of(1, picks[1])) {
                    const int alt = other_cluster(1, cluster_of(0, picks[0]));
                    if (alt >= 0) {
                        picks[1] = alt;
                        source[1] = "fallback", validation[1] = "same cluster as agent 0 while another was eligible";
                    }
                }
                for (int k = 0; k < 2; ++k) {
                    if (!P[k].active) continue;
                    std::ostringstream c;
                    c << ",\"choice\":{\"source\":\"" << source[k] << "\",\"decision_id\":\"" << did
                      << "\",\"validation\":\"" << JsonEscape(validation[k]) << "\",\"approval\":"
                      << (approval_raw[k].empty() ? "null" : "\"" + JsonEscape(approval_raw[k]) + "\"")
                      << (k == 0 ? std::string(",\"answer\":") + (ans.empty() ? "null" : ans) : std::string(",\"answer\":\"see agent 0's record\""))
                      << "}";
                    choice[k] = c.str();
                }
            }
            // Phase 3: the command for each pick (the same code for every chooser).
            for (int k = 0; k < 2; ++k) {
                Pending& pd = P[k];
                if (!pd.active) continue;
                const int pick = picks[k];
                std::ostringstream o;
                o.precision(10);
                o << pd.head;
                runtime::ros_cdr::PoseArray cmd;
                cmd.header.stamp.sec = int32_t(T / 1000000000);
                cmd.header.stamp.nanosec = uint32_t(T % 1000000000);
                cmd.header.frame_id = "world";
                if (pick >= 0) {
                    const auto& c = pd.cands[size_t(pick)];
                    const auto path = CommandPath(brief, pd.plan, pd.ac, c.goal, lc.eps_cells, lc.max_command_m);
                    if (!path.empty()) last_goal[k] = path.back();
                    for (const auto& [x, y] : path) {
                        runtime::ros_cdr::Pose p;
                        p.position = {x, y, brief.entry_z[size_t(k)]};
                        p.orientation = {0, 0, 0, 0};   // no yaw: face the next waypoint (D151)
                        cmd.poses.push_back(p);
                    }
                    if (!cmd.poses.empty() && lc.face_frontier) {
                        const auto& last = cmd.poses.back().position;
                        const double yaw = std::atan2(c.centroid_y - last.y, c.centroid_x - last.x);
                        cmd.poses.back().orientation = {0, 0, std::sin(0.5 * yaw), std::cos(0.5 * yaw)};
                    }
                }
                const std::string result = cmd.poses.empty() ? "hold" : "go";
                o << "\"chosen\":" << pick << ",\"min_goal_m\":" << lc.min_goal_m << ",\"result\":\"" << result
                  << "\",\"command\":{\"stamp_ns\":" << T << ",\"topic\":\"" << cmd_topic.at(k) << "\",\"kind\":\""
                  << (cmd.poses.empty() ? "hold" : cmd.poses.size() == 1 ? "goto" : "follow") << "\",\"waypoints_xy_m\":[";
                for (size_t i = 0; i < cmd.poses.size(); ++i) o << (i ? "," : "") << Xy(cmd.poses[i].position.x, cmd.poses[i].position.y);
                o << "]}" << choice[k] << "}";
                trace.Put(o.str());
                telemetry.Put("{\"type\":\"decision\",\"record\":" + o.str() + "}");
                commands.Put("{\"agent\":" + std::to_string(k) + ",\"stamp_ns\":" + std::to_string(T) + ",\"poses\":" +
                             std::to_string(cmd.poses.size()) + "}");
                SendCommand(k, cmd);
                ++commands_sent;
            }
        }
        // End of the run: every agent holds.
        if (is_live) {
            for (int k = 0; k < 2; ++k) {
                runtime::ros_cdr::PoseArray hold;
                hold.header.stamp.sec = int32_t(end_ns / 1000000000);
                hold.header.stamp.nanosec = uint32_t(end_ns % 1000000000);
                hold.header.frame_id = "world";
                SendCommand(k, hold);
            }
        }
        // The estimators stop when the READER closes their queues (the first image past end_s, or the end of the
        // input): never earlier, or live and replay would consume different tails.
        for (auto& t : vo) t.join();
        if (is_live) ::shutdown(fd, SHUT_RDWR);
        reader.join();
        rec.Close();
        if (is_live) ::close(fd);

        std::ofstream res(run_dir / "results.yaml");
        res << "# ariadne_live results (design LIVE_RUNTIME_L1). Generated; do not hand-edit.\n"
            << "mode: " << (is_live ? "live" : "replay") << "\nscene_id: " << brief.scene_id
            << "\nlive_config: " << lc.name << "\nmission_config: " << mc.name << "\nend_s: " << lc.end_s
            << "\nconsumed: {messages: " << rec.count() << ", digest_fnv1a64: '" << std::hex << rec.digest() << std::dec
            << "'}\nreplans: " << replans << "\ncommands_sent: " << (is_live ? commands_sent : 0)
            << "\ncommands_decided: " << commands_sent << "\nreorder_released_on_timeout: " << reorder_timeouts
            << "\nunrouted_messages: " << dropped_untyped << "\nanswer_timeouts: " << answer_timeouts
            << "\nrefused_goals: " << refused_goals.size() << "\nagents:\n";
        for (int k = 0; k < 2; ++k) {
            res << "  - {agent: " << k << ", frames: " << lanes[k].frames << ", keyframes: " << lanes[k].keyframes
                << ", unpaired_frames_dropped: " << lanes[k].unpaired << ", keyframes_integrated: " << integrated[k] << "}\n";
        }
        res << "rays: " << rays << "\n";
        res.close();

        runtime::RunManifest man;
        man.run_id = run_dir.filename().string();
        man.created_utc = tools::UtcNow("%Y-%m-%dT%H:%M:%SZ");
        man.git_commit = std::getenv("ARIADNE_GIT_COMMIT") ? std::getenv("ARIADNE_GIT_COMMIT") : runtime::kUnknownField;
        man.git_dirty = std::getenv("ARIADNE_GIT_DIRTY") && std::string(std::getenv("ARIADNE_GIT_DIRTY")) == "true";
        man.image_digest = std::getenv("ARIADNE_IMAGE_DIGEST") ? std::getenv("ARIADNE_IMAGE_DIGEST") : runtime::kUnknownField;
        man.scene = "sim-live/" + brief.scene_id;
        man.seed_recorded = true;
        man.seed = va.seed;
        man.agent_count = 2;
        man.execution_mode = is_live ? "live" : "replay";
        man.planner_config = mc.name;
        man.extra["planner"] = "frontier candidates (1.8) + chooser " + lc.chooser;
        man.extra["live_config"] = live_cfg_path;
        man.extra["perception_config"] = va.perception_config;
        man.extra["slam_config"] = va.slam_config;
        if (!replay_path.empty()) man.extra["replay_of"] = replay_path;
        std::string cmd;
        for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
        man.extra["command"] = cmd;
        runtime::RecordBuildIdentity(man);
        runtime::RecordOpenCvIdentity(man);
        runtime::RecordDeterminismDigests(man);
        std::ofstream(run_dir / "manifest.yaml") << runtime::ToYaml(man);
        std::printf("ariadne_live[%s]: %d replans, %d decisions, consumed %llu messages (digest %016llx) -> %s\n",
                    is_live ? "live" : "replay", replans, commands_sent, (unsigned long long)rec.count(),
                    (unsigned long long)rec.digest(), run_dir.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_live: %s\n", e.what());
        return 1;
    }
}
