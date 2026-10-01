// ---------------------------------------------------------------------------
// ariadne_sim_calib: the 1.1 acceptance test (design SIM_1.1-1.3 §6). Recovers
// the simulated camera rig from a recording's ChArUco frames and compares every
// convention against ground truth, so that a failure points at its cause.
//
//   ariadne_sim_calib --rec RECDIR --scene SCENEDIR --out OUTROOT [--agent 0]
//                     [--mission default] [--max-calib-views 80]
//
// Reads the two-bag recording ROS-free (D39/D40): sensors/ for the images and
// camera_info, canonical/ for GT pose and /tf_static. Only frames inside the
// agent's "calib:*" script segments are decoded. Five independent checks:
//
//  A. GT REPROJECTION (no calibration involved). Every detected ChArUco corner
//     is compared with the projection of its world position (scene.json board
//     corners) through the GT camera pose (canonical pose at the image stamp
//     composed with /tf_static) and camera_info's K, distortion OFF. Residual
//     vs normalized radius exposes distortion (with k1 = -0.05 it would grow to
//     tens of pixels at the corners); its mean exposes the principal point; its
//     correlation with image motion exposes a render/pose time offset.
//  B. FREE CALIBRATION per camera (k1, k2, p1, p2 free, k3 fixed): recovered K
//     and D with standard deviations, the radius the corners reached, and the
//     RMS of a pinhole-only calibration for comparison.
//  C. STEREO: stereoCalibrate (intrinsics fixed from B) vs /tf_static.
//  D. PnP vs GT per frame: rotation/translation error of the board pose, with a
//     check for the one wrong-but-consistent failure (a 180-degree-rotated board
//     texture, which matchImagePoints would silently accept).
//  E. BOARD FRAME self-check on the asset texture itself: detected corners on
//     the PGM must sit where getChessboardCorners() says (x right, y down).
//
// Writes OUTROOT/<scene_id>-<UTC>/{results.yaml, manifest.yaml, corners.csv}.
// Reports measurements; pass thresholds are set from the first measurement
// (design §6) and recorded separately, never tuned silently here.
// ---------------------------------------------------------------------------
#include "runtime/manifest/build_identity.hpp"
#include "runtime/manifest/determinism.hpp"
#include "runtime/manifest/opencv_identity.hpp"
#include "runtime/manifest/run_manifest.hpp"
#include "runtime/replay/ros_cdr.hpp"
#include "runtime/replay/rosbag_reader.hpp"

#include <Eigen/Dense>
#include <opencv2/geometry.hpp>
#include <opencv2/calib.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace cdr = ariadne::runtime::ros_cdr;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

namespace {

struct Args {
    std::string rec, scene, out, mission = "default";
    int agent = 0;
    int max_views = 80;
};

std::string UtcNow(const char* fmt) {
    char buf[64];
    const std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof buf, fmt, std::gmtime(&t));
    return buf;
}

struct Pose {               // T_W_X: p_W = R p_X + t
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
    Pose inverse() const { return {R.transpose(), -R.transpose() * t}; }
    Pose operator*(const Pose& o) const { return {R * o.R, R * o.t + t}; }
    Vec3 operator*(const Vec3& p) const { return R * p + t; }
};

double RotAngle(const Mat3& R) {
    return std::acos(std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0));
}

/// One calibration segment of the script: [t0, t1] in ns and its tag.
struct Window {
    int64_t t0, t1;
    std::string tag;
};

/// Board geometry: the ChArUco object frame (x right, y down, origin at the
/// squares' top-left, as getChessboardCorners() defines it) placed in the world
/// from scene.json's four square-area corners.
struct BoardInWorld {
    cv::aruco::CharucoBoard board;
    std::vector<cv::Point3f> obj;   // chessboard corners in the board frame, by id
    Pose T_W_board;                 // board frame -> world
    int first_id = 0, last_id = 0;
    double width_m = 0, height_m = 0;
};

struct Detection {
    int64_t ns = 0;
    std::string tag;
    std::vector<cv::Point2f> px;
    std::vector<int> ids;
};

struct CameraInfoK {
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<double> d;
    std::string model;
};

BoardInWorld LoadBoard(const YAML::Node& b) {
    const auto& sq = b["squares_xy"];
    const cv::Size size(sq[0].as<int>(), sq[1].as<int>());
    const float square = b["square_m"].as<float>(), marker = b["marker_m"].as<float>();
    if (b["dictionary"].as<std::string>() != "DICT_5X5_250") throw std::runtime_error("unexpected dictionary");
    const cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    BoardInWorld out{cv::aruco::CharucoBoard(size, square, marker, dict), {}, {}, 0, 0, 0, 0};
    out.first_id = b["first_id"].as<int>();
    out.last_id = b["last_id"].as<int>();
    std::vector<int> ids(out.last_id - out.first_id + 1);
    std::iota(ids.begin(), ids.end(), out.first_id);
    out.board = cv::aruco::CharucoBoard(size, square, marker, dict, ids);
    if (int(out.board.getIds().size()) != int(ids.size())) throw std::runtime_error("board id count mismatch");
    out.obj = out.board.getChessboardCorners();
    out.width_m = size.width * double(square);
    out.height_m = size.height * double(square);
    const auto& c = b["squares_corners_world_tl_tr_br_bl"];
    auto P = [&](int i) { return Vec3(c[i][0].as<double>(), c[i][1].as<double>(), c[i][2].as<double>()); };
    const Vec3 tl = P(0), tr = P(1), bl = P(3);
    const Vec3 ex = (tr - tl) / out.width_m, ey = (bl - tl) / out.height_m;
    if (std::abs(ex.norm() - 1.0) > 1e-6 || std::abs(ey.norm() - 1.0) > 1e-6 || std::abs(ex.dot(ey)) > 1e-9) {
        throw std::runtime_error("scene.json board corners do not span a " + std::to_string(out.width_m) + " x " +
                                 std::to_string(out.height_m) + " m rectangle");
    }
    out.T_W_board.R.col(0) = ex;
    out.T_W_board.R.col(1) = ey;
    out.T_W_board.R.col(2) = ex.cross(ey);
    out.T_W_board.t = tl;
    return out;
}

/// E: detect the asset texture itself; every corner must land where the board
/// frame (x right, y down, origin top-left of the squares) puts it.
std::string CheckBoardFrame(const BoardInWorld& b, const std::string& pgm, int px_per_square, int margin_px,
                            double& worst_px) {
    const cv::Mat img = cv::imread(pgm, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("cannot read " + pgm);
    const cv::aruco::CharucoDetector det(b.board);
    std::vector<cv::Point2f> px;
    std::vector<int> ids;
    det.detectBoard(img, px, ids);
    if (ids.size() != b.obj.size()) {
        return "detected " + std::to_string(ids.size()) + " of " + std::to_string(b.obj.size()) + " corners";
    }
    const double s = px_per_square / double(b.board.getSquareLength());   // px per metre
    worst_px = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        const cv::Point3f o = b.obj[ids[i]];
        // generateImage maps pixel centres: corner at (margin + x*s - 0.5, ...)
        const double eu = margin_px + o.x * s - 0.5, ev = margin_px + o.y * s - 0.5;
        worst_px = std::max(worst_px, std::hypot(px[i].x - eu, px[i].y - ev));
    }
    return worst_px < 1.0 ? "ok" : "corners off the x-right/y-down frame by " + std::to_string(worst_px) + " px";
}

Pose FromQuat(const cdr::Vector3& p, const cdr::Quaternion& q) {
    Pose T;
    T.R = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
    T.t = Vec3(p.x, p.y, p.z);
    return T;
}

struct Stats {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double mean() const { return v.empty() ? NAN : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }
    double rms() const {
        if (v.empty()) return NAN;
        double s = 0;
        for (double x : v) s += x * x;
        return std::sqrt(s / v.size());
    }
    double pct(double p) const {
        if (v.empty()) return NAN;
        auto w = v;
        std::sort(w.begin(), w.end());
        return w[std::min(w.size() - 1, size_t(p * (w.size() - 1) + 0.5))];
    }
    double max() const { return v.empty() ? NAN : *std::max_element(v.begin(), v.end()); }
};

std::string Kind(const std::string& tag) {          // calib:view / look / move / cruise:0.20 / ...
    const auto a = tag.find(':');
    return a == std::string::npos ? tag : tag.substr(a + 1);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
            const std::string v = argv[++i];
            if (k == "--rec") a.rec = v;
            else if (k == "--scene") a.scene = v;
            else if (k == "--out") a.out = v;
            else if (k == "--agent") a.agent = std::stoi(v);
            else if (k == "--mission") a.mission = v;
            else if (k == "--max-calib-views") a.max_views = std::stoi(v);
            else throw std::invalid_argument("unknown argument " + k);
        }
        if (a.rec.empty() || a.scene.empty() || a.out.empty()) {
            throw std::invalid_argument("usage: ariadne_sim_calib --rec RECDIR --scene SCENEDIR --out OUTROOT");
        }
        const YAML::Node scene = YAML::LoadFile(a.scene + "/scene.json");
        const YAML::Node mission = YAML::LoadFile(a.scene + "/mission_" + a.mission + ".json");
        const YAML::Node route = YAML::LoadFile(a.scene + "/routes/agent_" + std::to_string(a.agent) + ".json");
        YAML::Node ag;
        for (const auto& x : mission["agents"]) {
            if (x["agent"].as<int>() == a.agent) ag.reset(x);
        }
        if (!ag) throw std::runtime_error("agent not in mission");
        const std::string frame = ag["frame_id"].as<std::string>();
        const auto topics = ag["topics"];
        const std::string t_left = topics["left_image"].as<std::string>(), t_right = topics["right_image"].as<std::string>();
        const std::string t_linfo = topics["left_info"].as<std::string>(), t_rinfo = topics["right_info"].as<std::string>();
        const std::string t_pose = topics["gt_pose"].as<std::string>();

        // --- calibration windows from the script --------------------------------
        std::vector<Window> windows;
        for (const auto& s : route["segments"]) {
            const std::string tag = s["tag"].as<std::string>();
            if (tag.rfind("calib:", 0) != 0) continue;
            const double t0 = s["t0"].as<double>(), d = s["duration"].as<double>();
            windows.push_back({int64_t(std::llround(t0 * 1e9)), int64_t(std::llround((t0 + d) * 1e9)), tag});
        }
        if (windows.empty()) throw std::runtime_error("agent has no calib:* segments");
        auto TagAt = [&](int64_t ns) -> std::string {
            for (const auto& w : windows) {
                if (ns >= w.t0 && ns <= w.t1) return w.tag;
            }
            return {};
        };

        // --- the agent's board ----------------------------------------------------
        YAML::Node bnode;
        for (const auto& b : scene["charuco_boards"]) {
            if (b["agent"].as<int>() == a.agent) bnode.reset(b);
        }
        if (!bnode) throw std::runtime_error("no board for agent");
        const BoardInWorld board = LoadBoard(bnode);
        const std::string repo = std::getenv("ARIADNE_SOURCE_DIR") ? std::getenv("ARIADNE_SOURCE_DIR") : ".";
        const YAML::Node assets = YAML::LoadFile(repo + "/simulator/assets/charuco/boards.json");
        double frame_check_px = -1;
        const std::string frame_check = CheckBoardFrame(
            board, repo + "/simulator/assets/charuco/" + bnode["asset"].as<std::string>(),
            assets["px_per_square"].as<int>(), assets["margin_squares"].as<int>() * assets["px_per_square"].as<int>(),
            frame_check_px);
        std::printf("E board frame self-check: %s (worst %.3f px)\n", frame_check.c_str(), frame_check_px);

        // --- canonical: GT pose by stamp, static TF ------------------------------
        std::map<int64_t, Pose> gt;
        std::map<std::string, Pose> T_B_cam;   // "left" / "right" -> body <- optical
        {
            ariadne::runtime::RosbagReader bag;
            bag.Open(fs::path(a.rec) / "canonical" / "canonical_0.mcap");
            bag.Replay([&](const ariadne::runtime::RosMessage& m) {
                if (m.topic == t_pose) {
                    const auto p = cdr::DecodePoseStamped(m.data, m.size);
                    gt[p.header.stamp.ns()] = FromQuat(p.position, p.orientation);
                } else if (m.topic == "/tf_static") {
                    for (const auto& t : cdr::DecodeTfMessage(m.data, m.size)) {
                        if (t.header.frame_id != frame) continue;
                        for (const std::string cam : {"left", "right"}) {
                            if (t.child_frame_id == frame.substr(0, frame.size() - 9) + cam + "_optical") {
                                T_B_cam[cam] = FromQuat(t.translation, t.rotation);
                            }
                        }
                    }
                }
            }, {t_pose, "/tf_static"});
        }
        if (gt.empty() || T_B_cam.size() != 2) throw std::runtime_error("canonical bag lacks GT pose or static TF");

        // --- sensors: camera_info + detections inside the calib windows ----------
        std::map<std::string, CameraInfoK> info;
        std::map<std::string, std::vector<Detection>> dets;   // "left"/"right"
        std::map<std::string, int> decoded;
        cv::Size img_size;
        {
            const cv::aruco::CharucoDetector det(board.board);
            ariadne::runtime::RosbagReader bag;
            fs::path sensors = fs::path(a.rec) / "sensors";
            fs::path mcap;
            for (const auto& e : fs::directory_iterator(sensors)) {
                if (e.path().extension() == ".mcap") mcap = e.path();
            }
            bag.Open(mcap.string());
            bag.Replay([&](const ariadne::runtime::RosMessage& m) {
                const std::string topic(m.topic);
                if (topic == t_linfo || topic == t_rinfo) {
                    const std::string cam = topic == t_linfo ? "left" : "right";
                    if (info.count(cam)) return;
                    const auto ci = cdr::DecodeCameraInfo(m.data, m.size);
                    info[cam] = {int(ci.width), int(ci.height), ci.k[0], ci.k[4], ci.k[2], ci.k[5], ci.d,
                                 ci.distortion_model};
                    return;
                }
                const std::string cam = topic == t_left ? "left" : "right";
                // header stamp sits right after the 4-byte encapsulation: cheap window test
                int32_t sec;
                uint32_t nsec;
                std::memcpy(&sec, m.data + 4, 4);
                std::memcpy(&nsec, m.data + 8, 4);
                const int64_t ns = int64_t(sec) * 1000000000 + nsec;
                const std::string tag = TagAt(ns);
                if (tag.empty()) return;
                const auto im = cdr::DecodeImage(m.data, m.size);
                cdr::ValidateImage(im);
                if (im.encoding != "rgb8") throw std::runtime_error("expected rgb8, got " + im.encoding);
                const cv::Mat rgb(int(im.height), int(im.width), CV_8UC3, const_cast<uint8_t*>(im.data.data()), im.step);
                cv::Mat gray;
                cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
                img_size = gray.size();
                ++decoded[cam];
                Detection d;
                d.ns = ns;
                d.tag = tag;
                det.detectBoard(gray, d.px, d.ids);
                if (d.ids.size() >= 6) dets[cam].push_back(std::move(d));
            }, {t_left, t_right, t_linfo, t_rinfo});
        }
        if (info.size() != 2) throw std::runtime_error("camera_info missing");
        for (const std::string cam : {"left", "right"}) {
            std::printf("%s: %d calib-window frames decoded, %zu with >= 6 ChArUco corners\n", cam.c_str(),
                        decoded[cam], dets[cam].size());
            if (dets[cam].size() < 10) throw std::runtime_error(cam + ": too few board detections to calibrate");
        }

        // --- A: GT reprojection, no calibration ----------------------------------
        std::ofstream csv;
        const fs::path run_dir = fs::path(a.out) / (scene["scene_id"].as<std::string>() + "-agent" +
                                                    std::to_string(a.agent) + "-" + UtcNow("%Y%m%dT%H%M%SZ"));
        fs::create_directories(run_dir);
        csv.open(run_dir / "corners.csv");
        csv << "camera,stamp_ns,tag,id,u_det,v_det,u_gt,v_gt,r_norm,du_dt,dv_dt\n";
        struct Agg {
            Stats res, res_x, res_y, res_pc;   // res_pc: under the pixel-centre principal point (cx-0.5, cy-0.5)
        };
        std::map<std::string, std::map<std::string, Agg>> byKind;     // cam -> kind -> residual stats
        std::map<std::string, std::vector<std::array<double, 2>>> radial;   // cam -> (r_norm, radial residual px)
        std::map<std::string, std::array<double, 2>> timing_num_den;        // cam -> sum(res.v), sum(v.v)
        std::map<std::string, double> max_r;
        auto Project = [&](const CameraInfoK& K, const Pose& T_C_W, const Vec3& pw, bool& front) {
            const Vec3 pc = T_C_W * pw;
            front = pc.z() > 0.05;
            return cv::Point2d(K.fx * pc.x() / pc.z() + K.cx, K.fy * pc.y() / pc.z() + K.cy);
        };
        int missing_gt = 0;
        for (const std::string cam : {"left", "right"}) {
            const CameraInfoK& K = info[cam];
            for (const auto& d : dets[cam]) {
                const auto it = gt.find(d.ns);
                if (it == gt.end()) {
                    ++missing_gt;
                    continue;
                }
                const auto it2 = gt.find(d.ns + 5000000);   // next GT sample: image-plane velocity
                const Pose T_C_W = (it->second * T_B_cam[cam]).inverse();
                const Pose T_C_W2 = it2 == gt.end() ? T_C_W : (it2->second * T_B_cam[cam]).inverse();
                for (size_t i = 0; i < d.ids.size(); ++i) {
                    const cv::Point3f o = board.obj[d.ids[i]];
                    const Vec3 pw = board.T_W_board * Vec3(o.x, o.y, o.z);
                    bool f1, f2;
                    const cv::Point2d g = Project(K, T_C_W, pw, f1);
                    const cv::Point2d g2 = Project(K, T_C_W2, pw, f2);
                    if (!f1) continue;
                    const double du = d.px[i].x - g.x, dv = d.px[i].y - g.y;
                    const double vu = (g2.x - g.x) / 0.005, vv = (g2.y - g.y) / 0.005;   // px/s
                    const double xn = (d.px[i].x - K.cx) / K.fx, yn = (d.px[i].y - K.cy) / K.fy;
                    const double r = std::hypot(xn, yn);
                    max_r[cam] = std::max(max_r[cam], r);
                    const std::string kind = Kind(d.tag);
                    auto& ag2 = byKind[cam][kind];
                    ag2.res.add(std::hypot(du, dv));
                    ag2.res_x.add(du);
                    ag2.res_y.add(dv);
                    ag2.res_pc.add(std::hypot(du + 0.5, dv + 0.5));
                    if (r > 1e-6) radial[cam].push_back({r, (du * xn + dv * yn) / r});
                    if (std::hypot(vu, vv) > 20.0) {        // moving corners: residual ~ delta * velocity
                        timing_num_den[cam][0] += du * vu + dv * vv;
                        timing_num_den[cam][1] += vu * vu + vv * vv;
                    }
                    csv << cam << "," << d.ns << "," << d.tag << "," << d.ids[i] << "," << d.px[i].x << ","
                        << d.px[i].y << "," << g.x << "," << g.y << "," << r << "," << vu << "," << vv << "\n";
                }
            }
        }
        csv.close();
        // Radial residual model (pixels): fx * r * (k1 r^2 + k2 r^4) to first order.
        std::map<std::string, std::array<double, 2>> apparent_k;
        for (auto& [cam, v] : radial) {
            Eigen::MatrixXd A(v.size(), 2);
            Eigen::VectorXd b(v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                const double r = v[i][0];
                A(i, 0) = info[cam].fx * r * r * r;
                A(i, 1) = info[cam].fx * r * r * r * r * r;
                b(i) = v[i][1];
            }
            const Eigen::Vector2d k = A.colPivHouseholderQr().solve(b);
            apparent_k[cam] = {k(0), k(1)};
        }

        // --- B: free calibration per camera --------------------------------------
        struct Calib {
            cv::Mat K, D, stdI;
            double rms = 0, rms_pinhole = 0;
            int views = 0, points = 0;
        };
        std::map<std::string, Calib> calib;
        auto Select = [&](const std::vector<Detection>& v) {   // spread over time, prefer many corners
            std::vector<const Detection*> good;
            for (const auto& d : v) {
                if (d.ids.size() >= 10) good.push_back(&d);
            }
            std::vector<const Detection*> out;
            const size_t n = std::min(good.size(), size_t(a.max_views));
            for (size_t i = 0; i < n; ++i) out.push_back(good[i * good.size() / n]);
            return out;
        };
        for (const std::string cam : {"left", "right"}) {
            std::vector<std::vector<cv::Point3f>> objs;
            std::vector<std::vector<cv::Point2f>> imgs;
            for (const Detection* d : Select(dets[cam])) {
                cv::Mat o, im;
                board.board.matchImagePoints(d->px, d->ids, o, im);
                objs.push_back(o.clone());
                imgs.push_back(im.clone());
                calib[cam].points += int(o.total());
            }
            Calib& c = calib[cam];
            c.views = int(objs.size());
            cv::Mat stdE, perView;
            std::vector<cv::Mat> rv, tv;
            c.rms = cv::calibrateCamera(objs, imgs, img_size, c.K, c.D, rv, tv, c.stdI, stdE, perView,
                                        cv::CALIB_FIX_K3);
            cv::Mat K0, D0;
            c.rms_pinhole = cv::calibrateCamera(objs, imgs, img_size, K0, D0, rv, tv,
                                                cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 |
                                                    cv::CALIB_ZERO_TANGENT_DIST);
        }

        // --- C: stereo --------------------------------------------------------------
        std::map<int64_t, const Detection*> rightByNs;
        for (const auto& d : dets["right"]) rightByNs[d.ns] = &d;
        std::vector<std::vector<cv::Point3f>> sobj;
        std::vector<std::vector<cv::Point2f>> sl, sr;
        std::vector<const Detection*> lefts;
        for (const auto& d : dets["left"]) {
            if (d.ids.size() >= 10 && rightByNs.count(d.ns)) lefts.push_back(&d);
        }
        const size_t ns_pairs = std::min(lefts.size(), size_t(a.max_views));
        for (size_t i = 0; i < ns_pairs; ++i) {
            const Detection* L = lefts[i * lefts.size() / ns_pairs];
            const Detection* R = rightByNs[L->ns];
            std::map<int, cv::Point2f> rp;
            for (size_t j = 0; j < R->ids.size(); ++j) rp[R->ids[j]] = R->px[j];
            std::vector<cv::Point3f> o;
            std::vector<cv::Point2f> l, r;
            for (size_t j = 0; j < L->ids.size(); ++j) {
                const auto it = rp.find(L->ids[j]);
                if (it == rp.end()) continue;
                o.push_back(board.obj[L->ids[j]]);
                l.push_back(L->px[j]);
                r.push_back(it->second);
            }
            if (o.size() >= 8) {
                sobj.push_back(o);
                sl.push_back(l);
                sr.push_back(r);
            }
        }
        cv::Mat Rs, Ts, Es, Fs;
        const double stereo_rms = cv::stereoCalibrate(sobj, sl, sr, calib["left"].K, calib["left"].D, calib["right"].K,
                                                      calib["right"].D, img_size, Rs, Ts, Es, Fs, cv::CALIB_FIX_INTRINSIC);
        // The same pairs with the hypothesised TRUE intrinsics (camera_info K moved to the
        // pixel-centre convention, no distortion): separates calibration noise from the extrinsic.
        auto TrueK = [&](const std::string& cam) {
            const CameraInfoK& K = info[cam];
            return cv::Mat(cv::Matx33d(K.fx, 0, K.cx - 0.5, 0, K.fy, K.cy - 0.5, 0, 0, 1));
        };
        cv::Mat Rt, Tt, Et, Ft;
        const cv::Mat Kl = TrueK("left"), Kr = TrueK("right"), Z = cv::Mat::zeros(1, 5, CV_64F);
        const double stereo_rms_true = cv::stereoCalibrate(sobj, sl, sr, Kl, Z, Kr, Z, img_size, Rt, Tt, Et, Ft,
                                                           cv::CALIB_FIX_INTRINSIC);
        const Pose T_r_l = T_B_cam["right"].inverse() * T_B_cam["left"];   // p_right = T_r_l p_left
        Mat3 Rse;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) Rse(i, j) = Rs.at<double>(i, j);
        }
        const Vec3 Tse(Ts.at<double>(0), Ts.at<double>(1), Ts.at<double>(2));
        const double stereo_rot_err_deg = RotAngle(Rse * T_r_l.R.transpose()) * 180.0 / M_PI;
        const double stereo_t_err_m = (Tse - T_r_l.t).norm();
        Mat3 Rte;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) Rte(i, j) = Rt.at<double>(i, j);
        }
        const Vec3 Tte(Tt.at<double>(0), Tt.at<double>(1), Tt.at<double>(2));
        const double stereo_true_rot_err_deg = RotAngle(Rte * T_r_l.R.transpose()) * 180.0 / M_PI;
        const double stereo_true_t_err_m = (Tte - T_r_l.t).norm();

        // --- D: PnP vs GT per frame (left camera, pixel-centre K, no distortion) ----
        Stats pnp_rot_deg, pnp_t_m;
        int flipped = 0;
        for (const auto& d : dets["left"]) {
            const auto it = gt.find(d.ns);
            if (it == gt.end() || d.ids.size() < 10) continue;
            cv::Mat o, im, rvec, tvec;
            board.board.matchImagePoints(d.px, d.ids, o, im);
            const CameraInfoK& K = info["left"];
            const cv::Mat Km(cv::Matx33d(K.fx, 0, K.cx - 0.5, 0, K.fy, K.cy - 0.5, 0, 0, 1));   // pixel-centre K
            if (!cv::solvePnP(o, im, Km, cv::noArray(), rvec, tvec, false, cv::SOLVEPNP_IPPE)) continue;
            cv::Mat Rm;
            cv::Rodrigues(rvec, Rm);
            Pose est;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) est.R(i, j) = Rm.at<double>(i, j);
                est.t(i) = tvec.at<double>(i);
            }
            const Pose truth = (it->second * T_B_cam["left"]).inverse() * board.T_W_board;   // T_C_board
            const Mat3 dR = est.R * truth.R.transpose();
            const double ang = RotAngle(dR);
            pnp_rot_deg.add(ang * 180.0 / M_PI);
            pnp_t_m.add((est.t - truth.t).norm());
            if (std::abs(ang - M_PI) < 0.05) ++flipped;
        }

        // --- results ------------------------------------------------------------
        std::ofstream res(run_dir / "results.yaml");
        res.precision(10);
        res << "# ariadne_sim_calib results (design SIM_1.1-1.3 §6). Generated; do not hand-edit.\n"
            << "# Measurements only: pass thresholds are set from the first measurement and recorded separately.\n"
            << "scene_id: " << scene["scene_id"].as<std::string>() << "\nagent: " << a.agent << "\n"
            << "board: {first_id: " << board.first_id << ", last_id: " << board.last_id
            << ", frame_self_check: \"" << frame_check << "\", frame_self_check_worst_px: " << frame_check_px << "}\n"
            << "rig_sdf: {fx_nominal: " << mission["intrinsics_nominal"]["fx"].as<double>()
            << ", distortion: {k1: " << mission["rig"]["camera"]["distortion"]["k1"].as<double>()
            << ", k2: " << mission["rig"]["camera"]["distortion"]["k2"].as<double>()
            << ", k3: " << mission["rig"]["camera"]["distortion"]["k3"].as<double>()
            << ", p1: " << mission["rig"]["camera"]["distortion"]["p1"].as<double>()
            << ", p2: " << mission["rig"]["camera"]["distortion"]["p2"].as<double>() << "}}\n"
            << "gt_missing_for_detections: " << missing_gt << "\ncameras:\n";
        for (const std::string cam : {"left", "right"}) {
            const CameraInfoK& K = info[cam];
            const Calib& c = calib[cam];
            res << "  " << cam << ":\n    frames_decoded: " << decoded[cam] << "\n    frames_detected: "
                << dets[cam].size() << "\n    camera_info: {width: " << K.width << ", height: " << K.height
                << ", fx: " << K.fx << ", fy: " << K.fy << ", cx: " << K.cx << ", cy: " << K.cy << ", model: \""
                << K.model << "\", d: [";
            for (size_t i = 0; i < K.d.size(); ++i) res << (i ? ", " : "") << K.d[i];
            res << "]}\n    A_gt_reprojection_px:  # detected corner - GT projection (camera_info K, NO distortion)\n";
            for (const auto& [kind, g] : byKind[cam]) {
                res << "      " << kind << ": {n: " << g.res.v.size() << ", rms: " << g.res.rms() << ", p95: "
                    << g.res.pct(0.95) << ", max: " << g.res.max() << ", mean_du: " << g.res_x.mean()
                    << ", mean_dv: " << g.res_y.mean() << ", rms_pixel_centre_k: " << g.res_pc.rms()
                    << ", p95_pixel_centre_k: " << g.res_pc.pct(0.95) << ", max_pixel_centre_k: " << g.res_pc.max()
                    << "}\n";
            }
            const auto& tn = timing_num_den[cam];
            res << "    A_apparent_radial_distortion: {k1: " << apparent_k[cam][0] << ", k2: " << apparent_k[cam][1]
                << ", max_r_norm: " << max_r[cam] << "}  # least squares on radial residual\n"
                << "    A_time_offset_s: " << (tn[1] > 0 ? tn[0] / tn[1] : NAN)
                << "  # residual = delta * image velocity over moving corners (> 20 px/s); + = image shows a later pose\n"
                << "    B_free_calibration: {views: " << c.views << ", points: " << c.points << ", rms_px: " << c.rms
                << ", rms_px_pinhole_only: " << c.rms_pinhole << ", fx: " << c.K.at<double>(0, 0) << ", fy: "
                << c.K.at<double>(1, 1) << ", cx: " << c.K.at<double>(0, 2) << ", cy: " << c.K.at<double>(1, 2)
                << ", k1: " << c.D.at<double>(0) << ", k2: " << c.D.at<double>(1) << ", p1: " << c.D.at<double>(2)
                << ", p2: " << c.D.at<double>(3) << ",\n      std: {fx: " << c.stdI.at<double>(0) << ", fy: "
                << c.stdI.at<double>(1) << ", cx: " << c.stdI.at<double>(2) << ", cy: " << c.stdI.at<double>(3)
                << ", k1: " << c.stdI.at<double>(4) << ", k2: " << c.stdI.at<double>(5) << ", p1: "
                << c.stdI.at<double>(6) << ", p2: " << c.stdI.at<double>(7) << "}}\n";
        }
        res << "C_stereo: {intrinsics: B_free_calibration, pairs: " << sobj.size() << ", rms_px: " << stereo_rms << ", T_est: [" << Tse.x() << ", "
            << Tse.y() << ", " << Tse.z() << "], T_tf_static: [" << T_r_l.t.x() << ", " << T_r_l.t.y() << ", "
            << T_r_l.t.z() << "], t_err_m: " << stereo_t_err_m << ", rot_err_deg: " << stereo_rot_err_deg
            << ",\n  with_pixel_centre_k_no_distortion: {rms_px: " << stereo_rms_true << ", T_est: [" << Tte.x() << ", "
            << Tte.y() << ", " << Tte.z() << "], t_err_m: " << stereo_true_t_err_m << ", rot_err_deg: "
            << stereo_true_rot_err_deg << "}}\n"
            << "D_pnp_vs_gt_left: {k: pixel_centre_no_distortion, frames: " << pnp_rot_deg.v.size() << ", rot_err_deg: {mean: " << pnp_rot_deg.mean()
            << ", p95: " << pnp_rot_deg.pct(0.95) << ", max: " << pnp_rot_deg.max() << "}, t_err_m: {mean: "
            << pnp_t_m.mean() << ", p95: " << pnp_t_m.pct(0.95) << ", max: " << pnp_t_m.max()
            << "}, flipped_180: " << flipped << "}\n";
        res.close();

        ariadne::runtime::RunManifest m;
        m.run_id = run_dir.filename().string();
        m.created_utc = UtcNow("%Y-%m-%dT%H:%M:%SZ");
        m.git_commit = std::getenv("ARIADNE_GIT_COMMIT") ? std::getenv("ARIADNE_GIT_COMMIT") : ariadne::runtime::kUnknownField;
        m.git_dirty = std::getenv("ARIADNE_GIT_DIRTY") && std::string(std::getenv("ARIADNE_GIT_DIRTY")) == "true";
        m.image_digest = std::getenv("ARIADNE_IMAGE_DIGEST") ? std::getenv("ARIADNE_IMAGE_DIGEST")
                                                             : ariadne::runtime::kUnknownField;
        m.scene = scene["scene_id"].as<std::string>();
        m.seed = scene["seed"].as<uint64_t>();
        m.seed_recorded = true;
        m.agent_count = int(mission["agents"].size());
        m.execution_mode = "replay";
        m.camera_config = mission["rig"]["name"].as<std::string>();
        m.degradation_config = "none";
        m.feature_engine = "charuco";
        m.loop_backend = "none";
        m.imu_mode = "none";
        {
            const YAML::Node rec = YAML::LoadFile(a.rec + "/recording.json");
            for (const auto& f : rec["streams"]["sensors"]["files"]) {
                if (f.first.as<std::string>().find(".mcap") != std::string::npos) {
                    m.bag_checksum = "sha256:" + f.second["sha256"].as<std::string>();
                }
            }
            m.extra["canonical_content_sha256"] = rec["streams"]["canonical"]["content_sha256"].as<std::string>();
            m.extra["recording_git_commit"] = rec["git_commit"].as<std::string>();
        }
        ariadne::runtime::RecordBuildIdentity(m);
        ariadne::runtime::RecordOpenCvIdentity(m);
        ariadne::runtime::RecordDeterminismDigests(m);
        std::string cmd;
        for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
        m.extra["command"] = cmd;
        std::ofstream(run_dir / "manifest.yaml") << ariadne::runtime::ToYaml(m);
        std::printf("results written to %s\n", run_dir.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_sim_calib: %s\n", e.what());
        return 1;
    }
}
