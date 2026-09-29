// ---------------------------------------------------------------------------
// ariadne_euroc_frontend: run the 1.4 placeholder front-end over one EuRoC
// sequence and MEASURE it (Invariant 5: datasets before simulator).
//
//   ariadne_euroc_frontend --sequence DIR --config configs/perception/euroc.yaml
//                          --out DIR [--frames N] [--stereo-every K] [--seed S]
//
// Measures, per run:
//   pairing      stereo frames paired by exact timestamp; unmatched counts
//   rectifier    fx, baseline, Z_min, blind-column fraction
//   rect check   median |dy| of mutual ORB matches between the rectified
//                left/right images (should be ~0), with two controls that
//                must NOT pass: the raw unrectified pair, and a rectifier
//                built with a deliberately 1-degree-wrong pitch
//   tracker      tracks per frame, state counts, rejection totals, timing
//   stereo       valid and dense-supported fractions, timing
//   aliasing     D34's promise, re-measured on real texture. EuRoC has no
//                dense depth ground truth, so a WIDE-range SGBM (3x the
//                disparities) is the pseudo-truth: a pixel valid in the
//                narrow run whose wide-run disparity lies beyond the narrow
//                search range is an ALIAS. Reported: alias rate, the share
//                the dense-support gate rejects, and the gate's cost (share
//                of narrow/wide-agreeing depth it rejects).
// Writes manifest.yaml (§28.2) and results.yaml into a fresh run directory.
//
// The dataset is never in CI (licence and size), so this is a tool that
// records a manifest, not a test that could silently skip.
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/stereo/sgbm_stereo.hpp"
#include "perception/stereo/stereo_lk_matcher.hpp"
#include "perception/stereo/stereo_rectifier.hpp"
#include "perception/tracking/klt_tracker.hpp"
#include "runtime/datasets/euroc.hpp"
#include "runtime/manifest/build_identity.hpp"
#include "runtime/manifest/determinism.hpp"
#include "runtime/manifest/opencv_identity.hpp"
#include "runtime/manifest/run_manifest.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/features.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace ariadne;
using namespace ariadne::perception;

namespace {

struct Args {
    std::string sequence, config, out;
    int frames = 0;          // 0 = all
    int stereo_every = 10;   // SGBM + rect check on every K-th frame (keyframe proxy)
    uint64_t seed = 1;
};

Args Parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--sequence") a.sequence = next();
        else if (k == "--config") a.config = next();
        else if (k == "--out") a.out = next();
        else if (k == "--frames") a.frames = std::stoi(next());
        else if (k == "--stereo-every") a.stereo_every = std::stoi(next());
        else if (k == "--seed") a.seed = std::stoull(next());
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); std::exit(2); }
    }
    if (a.sequence.empty() || a.config.empty() || a.out.empty() || a.stereo_every < 1) {
        std::fprintf(stderr, "usage: %s --sequence DIR --config YAML --out DIR [--frames N] "
                             "[--stereo-every K] [--seed S]\n", argv[0]);
        std::exit(2);
    }
    return a;
}

double Ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double Quantile(std::vector<double> v, double q) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * v.size()))];
}

double Mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? -1.0 : s / v.size();
}

/// |dy| of mutual (cross-checked) ORB matches with Hamming distance <= 50
/// (ORB-SLAM's TH_LOW). On a rectified pair every correct match lies on one row.
std::vector<double> RowErrors(const OrbEngine& orb, const cv::Mat& l, const cv::Mat& r, double* dx_pos) {
    const FeatureSet a = orb.DetectAndCompute(l, cv::Mat());
    const FeatureSet b = orb.DetectAndCompute(r, cv::Mat());
    std::vector<double> dy;
    if (a.keypoints.empty() || b.keypoints.empty()) return dy;
    std::vector<cv::DMatch> m;
    cv::BFMatcher(cv::NORM_HAMMING, /*crossCheck=*/true).match(a.descriptors, b.descriptors, m);
    int pos = 0;
    for (const auto& x : m) {
        if (x.distance > 50) continue;
        const cv::Point2f pl = a.keypoints[static_cast<size_t>(x.queryIdx)].pt;
        const cv::Point2f pr = b.keypoints[static_cast<size_t>(x.trainIdx)].pt;
        dy.push_back(std::abs(pl.y - pr.y));
        pos += pl.x - pr.x > 0.0f;
    }
    if (dx_pos) *dx_pos = dy.empty() ? -1.0 : static_cast<double>(pos) / dy.size();
    return dy;
}

/// Sub-pixel row error via the ONE stereo LK implementation
/// (perception/stereo/stereo_lk_matcher), with its row gate disabled: this
/// function MEASURES the row error, so it must not filter on it.
std::vector<double> LkRowErrors(const cv::Mat& l, const cv::Mat& r, const std::vector<cv::Point2f>& pts,
                                StereoMatchParams p) {
    p.max_dy_px = 1e9;
    const StereoMatches m = MatchLeftToRight(l, r, pts, p);
    std::vector<double> dy;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (m.valid[i]) dy.push_back(std::abs(m.right[i].y - pts[i].y));
    }
    return dy;
}

std::string UtcNow(const char* fmt) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, fmt, &tm);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = Parse(argc, argv);
    const fs::path seq(args.sequence);
    const fs::path mav = seq / "mav0";
    const PerceptionConfig cfg = LoadPerception(args.config);

    // --- Calibration, pairing, rectification ------------------------------
    const auto cam0 = runtime::LoadEurocCamera((mav / "cam0/sensor.yaml").string());
    const auto cam1 = runtime::LoadEurocCamera((mav / "cam1/sensor.yaml").string());
    const RawStereoCalibration raw = runtime::EurocStereo(cam0, cam1);
    const auto idx0 = runtime::ParseEurocImageCsv(runtime::ReadTextFile((mav / "cam0/data.csv").string()), "cam0");
    const auto idx1 = runtime::ParseEurocImageCsv(runtime::ReadTextFile((mav / "cam1/data.csv").string()), "cam1");
    const runtime::StereoPairs pairs = runtime::PairByTimestamp(idx0, idx1);

    const StereoRectifier rectifier(raw, 0.0);
    const RectifiedStereoRig& rig = rectifier.Rig();
    RawStereoCalibration wrong = raw;   // control: 1 degree of pitch error
    wrong.T_right_left.R = raw.T_right_left.R * core::SO3Exp(core::Vec3(0.01745, 0.0, 0.0));
    const StereoRectifier wrong_rectifier(wrong, 0.0);

    // Throws ConfigError if this config leaves a blind zone past its own
    // required_min_range_m for THIS rig (§9.4).
    const SgbmStereo sgbm(cfg.stereo, rig);
    StereoParams wide_params = cfg.stereo;
    wide_params.num_disparities = std::min(3 * cfg.stereo.num_disparities, (rig.width / 2) / 16 * 16);
    wide_params.required_min_range_m = 1e9;   // pseudo-truth only; no range requirement
    const SgbmStereo sgbm_wide(wide_params, rig);
    const int narrow_max_raw = MaxSearchedDisparity(cfg.stereo) * cv::StereoMatcher::DISP_SCALE;

    const OrbEngine orb(cfg.features);
    KltTracker tracker(cfg.tracker, orb);

    const int n = args.frames > 0 ? std::min<int>(args.frames, static_cast<int>(pairs.t_ns.size()))
                                  : static_cast<int>(pairs.t_ns.size());
    std::printf("EuRoC %s: %zu pairs (unmatched L %d, R %d); running %d frames\n",
                seq.filename().c_str(), pairs.t_ns.size(), pairs.unmatched_left, pairs.unmatched_right, n);
    std::printf("rectified rig: fx=%.3f cx=%.3f cy=%.3f B=%.5f m; Z_min=%.3f m; blind cols %d/%d\n", rig.fx,
                rig.cx, rig.cy, rig.baseline_m, MinObservableRange(cfg.stereo, rig),
                cfg.stereo.min_disparity + cfg.stereo.num_disparities, rig.width);

    // --- Replay -------------------------------------------------------------
    std::vector<double> t_rect, t_track, t_sgbm, tracks, survivors;
    std::vector<double> dy_rect, dy_raw, dy_wrong, dxpos_rect;
    std::vector<double> lk_dy_rect, lk_dy_raw, lk_dy_wrong;
    std::vector<double> alias_rate_per_kf;
    std::vector<double> valid_frac, supported_frac;
    std::map<TrackingState, int> states;
    TrackStats totals;
    // Pseudo-truth counts only pixels the WIDE run itself densely supports:
    // the wide search has more chances to produce spurious large disparities
    // in weak texture, which would otherwise be booked as gate misses.
    long alias = 0, alias_rejected = 0, narrow_valid = 0, truth_px = 0;
    // Gate cost, split: interior pixels (>= 15 px from every image edge and
    // from the blind-column boundary) vs the rim, where a 31x31 window with a
    // zero border scores low density by construction.
    long agree_int = 0, agree_int_rejected = 0, agree_rim = 0, agree_rim_rejected = 0;
    const int rim = 15;
    const int blind = cfg.stereo.min_disparity + cfg.stereo.num_disparities;

    for (int i = 0; i < n; ++i) {
        const cv::Mat l_raw = cv::imread((mav / "cam0/data" / pairs.left[static_cast<size_t>(i)]).string(),
                                         cv::IMREAD_GRAYSCALE);
        const cv::Mat r_raw = cv::imread((mav / "cam1/data" / pairs.right[static_cast<size_t>(i)]).string(),
                                         cv::IMREAD_GRAYSCALE);
        if (l_raw.empty() || r_raw.empty()) {
            std::fprintf(stderr, "failed to read frame %d\n", i);
            return 1;
        }
        cv::Mat l, r;
        auto t0 = std::chrono::steady_clock::now();
        rectifier.Rectify(l_raw, r_raw, l, r);
        t_rect.push_back(Ms(t0));

        t0 = std::chrono::steady_clock::now();
        const TrackedFrame tf = tracker.Process(l, static_cast<uint64_t>(i), args.seed);
        t_track.push_back(Ms(t0));
        ++states[tf.state];
        tracks.push_back(static_cast<double>(tf.ids.size()));
        survivors.push_back(static_cast<double>(tf.ids.size()) - tf.stats.replenished);
        totals.attempted += tf.stats.attempted;
        totals.lk_lost += tf.stats.lk_lost;
        totals.fb_rejected += tf.stats.fb_rejected;
        totals.border_rejected += tf.stats.border_rejected;
        totals.epipolar_rejected += tf.stats.epipolar_rejected;
        totals.crowded_out += tf.stats.crowded_out;
        totals.replenished += tf.stats.replenished;

        if (i % args.stereo_every != 0) continue;

        // Rectification check + controls.
        double dxpos = 0.0;
        const auto e = RowErrors(orb, l, r, &dxpos);
        dy_rect.insert(dy_rect.end(), e.begin(), e.end());
        dxpos_rect.push_back(dxpos);
        const auto e_raw = RowErrors(orb, l_raw, r_raw, nullptr);
        dy_raw.insert(dy_raw.end(), e_raw.begin(), e_raw.end());
        cv::Mat lw, rw;
        wrong_rectifier.Rectify(l_raw, r_raw, lw, rw);
        const auto e_wrong = RowErrors(orb, lw, rw, nullptr);
        dy_wrong.insert(dy_wrong.end(), e_wrong.begin(), e_wrong.end());
        auto lk = LkRowErrors(l, r, tf.points, cfg.stereo_match);
        lk_dy_rect.insert(lk_dy_rect.end(), lk.begin(), lk.end());
        lk = LkRowErrors(l_raw, r_raw, tf.points, cfg.stereo_match);   // raw-image control (same seeds)
        lk_dy_raw.insert(lk_dy_raw.end(), lk.begin(), lk.end());
        lk = LkRowErrors(lw, rw, tf.points, cfg.stereo_match);         // 1-degree pitch control
        lk_dy_wrong.insert(lk_dy_wrong.end(), lk.begin(), lk.end());

        // Stereo + dual-range alias measurement.
        t0 = std::chrono::steady_clock::now();
        const StereoDepth d = sgbm.Compute(l, r);
        t_sgbm.push_back(Ms(t0));
        const cv::Mat supported = DenseSupportMask(d, cfg.stereo);
        valid_frac.push_back(d.valid_fraction);
        supported_frac.push_back(static_cast<double>(cv::countNonZero(supported)) / (rig.width * rig.height));
        const StereoDepth w = sgbm_wide.Compute(l, r);
        const cv::Mat w_supported = DenseSupportMask(w, wide_params);
        long kf_alias = 0, kf_truth = 0;
        for (int y = 0; y < rig.height; ++y) {
            const uchar* nv = d.valid.ptr<uchar>(y);
            const uchar* ws = w_supported.ptr<uchar>(y);
            const uchar* sp = supported.ptr<uchar>(y);
            const int16_t* nd = d.disparity.ptr<int16_t>(y);
            const int16_t* wd = w.disparity.ptr<int16_t>(y);
            const bool row_interior = y >= rim && y < rig.height - rim;
            for (int x = 0; x < rig.width; ++x) {
                if (!nv[x]) continue;
                ++narrow_valid;
                if (!ws[x]) continue;           // no trustworthy pseudo-truth here
                ++kf_truth;
                if (wd[x] > narrow_max_raw) {   // truly nearer than Z_min
                    ++kf_alias;
                    alias_rejected += sp[x] == 0;
                } else if (std::abs(wd[x] - nd[x]) <= cv::StereoMatcher::DISP_SCALE) {   // within 1 px
                    const bool interior = row_interior && x >= blind + rim && x < rig.width - rim;
                    (interior ? agree_int : agree_rim) += 1;
                    (interior ? agree_int_rejected : agree_rim_rejected) += sp[x] == 0;
                }
            }
        }
        alias += kf_alias;
        truth_px += kf_truth;
        alias_rate_per_kf.push_back(kf_truth ? static_cast<double>(kf_alias) / kf_truth : 0.0);
        if (i % (args.stereo_every * 50) == 0) {
            std::printf("  frame %d/%d: tracks %zu, state %d, valid %.3f\n", i, n, tf.ids.size(),
                        static_cast<int>(tf.state), d.valid_fraction);
        }
    }

    // --- Outputs ------------------------------------------------------------
    const fs::path run_dir = fs::path(args.out) / (seq.filename().string() + "-" + UtcNow("%Y%m%dT%H%M%SZ"));
    fs::create_directories(run_dir);

    runtime::RunManifest m;
    m.run_id = run_dir.filename().string();
    m.created_utc = UtcNow("%Y-%m-%dT%H:%M:%SZ");
    m.git_commit = std::getenv("ARIADNE_GIT_COMMIT") ? std::getenv("ARIADNE_GIT_COMMIT") : runtime::kUnknownField;
    m.git_dirty = std::getenv("ARIADNE_GIT_DIRTY") && std::string(std::getenv("ARIADNE_GIT_DIRTY")) == "true";
    m.image_digest = std::getenv("ARIADNE_IMAGE_DIGEST") ? std::getenv("ARIADNE_IMAGE_DIGEST") : runtime::kUnknownField;
    m.scene = "EuRoC/" + seq.filename().string();
    m.seed = args.seed;
    m.seed_recorded = true;
    m.execution_mode = "replay";
    m.perception_config = cfg.name;
    m.degradation_config = "none";   // raw dataset pixels; D is not applied in this run
    m.feature_engine = orb.Name();
    m.agent_count = 1;
    // This run uses no inertial data at all. The manifest's default names the
    // §10.7 fallback mode, which would claim an inertial estimator ran.
    m.imu_mode = "none";
    m.loop_backend = "none";   // front-end only: no loop closure in this run
    const fs::path prov = seq.parent_path() / (seq.filename().string() + ".provenance.json");
    if (fs::exists(prov)) m.bag_checksum = "sha256:" + YAML::LoadFile(prov.string())["sha256"].as<std::string>();
    runtime::RecordBuildIdentity(m);
    runtime::RecordOpenCvIdentity(m);
    runtime::RecordDeterminismDigests(m);
    m.extra["opencv_threads"] = std::to_string(cv::getNumThreads());
    m.extra["rectifier_alpha"] = "0";
    m.extra["stereo_every"] = std::to_string(args.stereo_every);
    m.extra["frames"] = std::to_string(n);
    std::string cmd;   // §36: the exact command that produced this directory
    for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
    m.extra["command"] = cmd;
    std::ofstream(run_dir / "manifest.yaml") << runtime::ToYaml(m);

    auto ratio = [](long a, long b) { return b ? static_cast<double>(a) / b : -1.0; };
    long kf_over_1pct = 0;
    for (double r : alias_rate_per_kf) kf_over_1pct += r > 0.01;
    std::ofstream res(run_dir / "results.yaml");
    res << "# ariadne_euroc_frontend results. Generated; do not hand-edit.\n"
        << "sequence: " << seq.filename().string() << "\n"
        << "pairs_total: " << pairs.t_ns.size() << "\nunmatched_left: " << pairs.unmatched_left
        << "\nunmatched_right: " << pairs.unmatched_right << "\nframes_run: " << n << "\n"
        << "rig: {fx: " << rig.fx << ", fy: " << rig.fy << ", cx: " << rig.cx << ", cy: " << rig.cy
        << ", baseline_m: " << rig.baseline_m << "}\n"
        << "stereo_min_range_m: " << MinObservableRange(cfg.stereo, rig) << "\n"
        << "# ORB matches: integer FAST positions, so |dy| has ~1 px resolution.\n"
        << "rect_check_orb: {matches: " << dy_rect.size() << ", median_abs_dy_px: " << Quantile(dy_rect, 0.5)
        << ", p90_abs_dy_px: " << Quantile(dy_rect, 0.9) << ", positive_dx_fraction: " << Mean(dxpos_rect)
        << ", control_raw_median: " << Quantile(dy_raw, 0.5)
        << ", control_1deg_pitch_median: " << Quantile(dy_wrong, 0.5) << "}\n"
        << "# LK left->right with reverse check: sub-pixel.\n"
        << "rect_check_lk: {points: " << lk_dy_rect.size() << ", median_abs_dy_px: " << Quantile(lk_dy_rect, 0.5)
        << ", p90_abs_dy_px: " << Quantile(lk_dy_rect, 0.9) << ", control_raw_points: " << lk_dy_raw.size()
        << ", control_raw_median: " << Quantile(lk_dy_raw, 0.5)
        << ", control_1deg_pitch_points: " << lk_dy_wrong.size()
        << ", control_1deg_pitch_median: " << Quantile(lk_dy_wrong, 0.5) << "}\n"
        << "tracker: {tracks_mean: " << Mean(tracks) << ", tracks_p10: " << Quantile(tracks, 0.1)
        << ", survivors_mean: " << Mean(survivors) << ", initializing: " << states[TrackingState::kInitializing]
        << ", tracking: " << states[TrackingState::kTracking]
        << ", degraded: " << states[TrackingState::kDegraded] << ", attempted: " << totals.attempted
        << ", lk_lost: " << totals.lk_lost << ", fb_rejected: " << totals.fb_rejected
        << ", border_rejected: " << totals.border_rejected << ", epipolar_rejected: " << totals.epipolar_rejected
        << ", crowded_out: " << totals.crowded_out << ", replenished: " << totals.replenished << "}\n"
        << "stereo: {keyframes: " << valid_frac.size() << ", valid_fraction_mean: " << Mean(valid_frac)
        << ", supported_fraction_mean: " << Mean(supported_frac) << "}\n"
        << "# Pseudo-truth = wide-range SGBM, restricted to pixels its own dense-support gate keeps.\n"
        << "aliasing: {wide_num_disparities: " << wide_params.num_disparities
        << ", narrow_valid_px: " << narrow_valid << ", pseudo_truth_px: " << truth_px
        << ", alias_px: " << alias << ", alias_rate: " << ratio(alias, truth_px)
        << ", alias_rate_kf_max: " << Quantile(alias_rate_per_kf, 1.0)
        << ", keyframes_over_1pct: " << kf_over_1pct
        << ", gate_rejects_alias_fraction: " << ratio(alias_rejected, alias)
        << ", residual_alias_rate: " << ratio(alias - alias_rejected, truth_px)
        << ", gate_cost_interior: " << ratio(agree_int_rejected, agree_int)
        << ", gate_cost_rim: " << ratio(agree_rim_rejected, agree_rim)
        << ", agree_interior_px: " << agree_int << ", agree_rim_px: " << agree_rim << "}\n"
        << "timing_ms: {rectify_mean: " << Mean(t_rect) << ", track_mean: " << Mean(t_track)
        << ", track_p95: " << Quantile(t_track, 0.95) << ", sgbm_mean: " << Mean(t_sgbm)
        << ", opencv_threads: " << cv::getNumThreads() << "}\n";
    res.close();
    const auto missing = runtime::MissingRequiredFields(m);
    std::printf("run written to %s\nmanifest missing required fields: %zu", run_dir.c_str(), missing.size());
    for (const auto& f : missing) std::printf(" [%s]", f.c_str());
    std::printf("\n");
    std::fputs(runtime::ReadTextFile((run_dir / "results.yaml").string()).c_str(), stdout);
    return 0;
}
