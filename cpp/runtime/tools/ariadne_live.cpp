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
    if (c.chooser != "classical") throw std::invalid_argument("live config: chooser must be classical (7.1 adds more)");
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

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string scene, mission_cfg_path, live_cfg_path, out, link_path, replay_path;
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
            for (int k = 0; k < 2; ++k) {
                std::ostringstream o;
                o.precision(10);
                o << "{\"t_s\":" << ts << ",\"agent\":" << k << ",\"mode\":\"" << (is_live ? "live" : "replay")
                  << "\",\"executed\":" << (is_live ? "true" : "false") << ",\"decision\":\"frontier\",\"chooser\":\""
                  << lc.chooser << "\",";
                // The vision-derived inputs of this decision: the map built from keyframe depth, the agent's estimate.
                bool have_pose = false;
                core::SE3 T_O_C;
                {
                    std::lock_guard<std::mutex> lk(lanes[k].mu);
                    for (auto it = lanes[k].cams.rbegin(); it != lanes[k].cams.rend(); ++it) {
                        if (it->first < T) {
                            T_O_C = it->second;
                            have_pose = true;
                            break;
                        }
                    }
                }
                o << "\"vision\":{\"keyframes_integrated\":[" << integrated[0] << "," << integrated[1] << "],\"rays\":"
                  << rays << ",\"frontier_clusters\":" << maps.frontiers.clusters.size() << ",\"rejected_small\":"
                  << maps.frontiers.rejected_small << ",\"egress_found\":" << (maps.egress.found ? "true" : "false");
                if (!have_pose) {
                    o << "},\"result\":\"no decision: no frame yet\"}";
                    trace.Put(o.str());
                    continue;
                }
                const core::SE3 T_W_C = brief.T_W_O[size_t(k)] * T_O_C;
                const Cell ac = brief.ToCell(T_W_C.t.x(), T_W_C.t.y());
                o << ",\"agent_xy_m\":" << Xy(T_W_C.t.x(), T_W_C.t.y()) << ",\"agent_cell\":[" << ac.r << "," << ac.c
                  << "]},";
                // Standoff candidates over the conservative plan space (D153b), minus goals the mover refused before.
                const cv::Mat plan = map.PlanSpace(maps, ac, lc.escape_m);
                std::vector<mis::Candidate> cands;
                int excluded_refused = 0;
                for (const auto& c : map.StandoffCandidates(maps, plan, ac, lc.max_standoff_m)) {
                    const auto [gx, gy] = brief.CellCentre(c.goal);
                    bool refused = false;
                    for (const auto& [ra, rg] : refused_goals) {
                        refused = refused || (ra == k && std::hypot(rg.first - gx, rg.second - gy) < lc.refused_radius_m);
                    }
                    if (refused) ++excluded_refused;
                    else cands.push_back(c);
                }
                o << "\"candidates\":[";
                for (size_t i = 0; i < cands.size() && i < 5; ++i) {
                    const auto& c = cands[i];
                    const auto [gx, gy] = brief.CellCentre(c.goal);
                    o << (i ? "," : "") << "{\"cluster\":" << c.cluster << ",\"path_m\":" << c.path_cells * brief.gs.cell
                      << ",\"goal_xy_m\":" << Xy(gx, gy) << ",\"standoff_m\":" << c.standoff_m
                      << ",\"centroid_xy_m\":" << Xy(c.centroid_x, c.centroid_y) << ",\"cells\":" << c.cells << "}";
                }
                o << "],\"candidates_total\":" << cands.size() << ",\"excluded_refused\":" << excluded_refused << ",";
                // Chooser (classical): the nearest reachable cluster; none -> hold.
                runtime::ros_cdr::PoseArray cmd;
                cmd.header.stamp.sec = int32_t(T / 1000000000);
                cmd.header.stamp.nanosec = uint32_t(T % 1000000000);
                cmd.header.frame_id = "world";
                std::string result = "hold";
                // Classical chooser: the nearest candidate at least min_goal_m away (the nearest one overall is the
                // band under the agent; live_v1.yaml).
                int pick = -1;
                for (size_t i = 0; i < cands.size(); ++i) {
                    if (cands[i].path_cells * brief.gs.cell >= lc.min_goal_m) {
                        pick = int(i);
                        break;
                    }
                }
                if (pick >= 0) {
                    const auto& c = cands[size_t(pick)];
                    const auto path = CommandPath(brief, plan, ac, c.goal, lc.eps_cells, lc.max_command_m);
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
                    result = cmd.poses.empty() ? "hold" : "go";
                    o << "\"chosen\":" << pick << ",\"min_goal_m\":" << lc.min_goal_m << ",";
                } else {
                    o << "\"chosen\":-1,";
                }
                o << "\"result\":\"" << result << "\",\"command\":{\"stamp_ns\":" << T << ",\"topic\":\""
                  << cmd_topic.at(k) << "\",\"kind\":\""
                  << (cmd.poses.empty() ? "hold" : cmd.poses.size() == 1 ? "goto" : "follow") << "\",\"waypoints_xy_m\":[";
                for (size_t i = 0; i < cmd.poses.size(); ++i) o << (i ? "," : "") << Xy(cmd.poses[i].position.x, cmd.poses[i].position.y);
                o << "]}}";
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
