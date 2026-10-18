// Placeholder stereo VO (plan 1.5), end to end on a synthetic stereo rig: no
// dataset needed, so it runs in CI and keeps the nightly thin run honest.
// Also: Ariadne's motion-only LM against OpenCV's solvePnPRefineLM (oracle).
#include "core/random/distributions.hpp"
#include "core/random/philox.hpp"
#include "runtime/agent/keyframe_records.hpp"
#include "slam/bundle_adjustment/reprojection_lm.hpp"
#include "slam/inertial/attitude.hpp"
#include "slam/odometry/stereo_vo.hpp"

#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/geometry.hpp>

#include <cstring>

using namespace ariadne;
using namespace ariadne::slam;
using core::SE3;
using core::Vec3;

namespace {

const core::RectifiedStereo kRig{{436.2, 436.2, 364.4, 257.0}, 0.11};
constexpr int kW = 752, kH = 480;

/// The SHIPPED sim config (the one the nightly thin run uses), not struct
/// defaults: tests must exercise what runs.
SlamConfig Config() {
    static const SlamConfig c = LoadSlam(std::string(ARIADNE_SOURCE_DIR) + "/configs/slam/placeholder.yaml");
    return c;
}

/// Points on the walls, floor and ceiling of a 10 x 3 x 16 m room.
std::vector<Vec3> Room(uint64_t seed) {
    core::PhiloxStream s(seed, 0, 2);
    auto u = [&s] { return (s.NextBits32() >> 8) * (1.0 / 16777216.0); };
    std::vector<Vec3> pts;
    for (int i = 0; i < 6000; ++i) {
        const double a = u(), b = u();
        switch (i % 5) {
            case 0: pts.emplace_back(-5.0, -1.5 + 3.0 * a, -2.0 + 16.0 * b); break;   // left wall
            case 1: pts.emplace_back(5.0, -1.5 + 3.0 * a, -2.0 + 16.0 * b); break;    // right wall
            case 2: pts.emplace_back(-5.0 + 10.0 * a, 1.5, -2.0 + 16.0 * b); break;   // floor (y down)
            case 3: pts.emplace_back(-5.0 + 10.0 * a, -1.5, -2.0 + 16.0 * b); break;  // ceiling
            default: pts.emplace_back(-5.0 + 10.0 * a, -1.5 + 3.0 * b, 14.0); break;  // far wall
        }
    }
    return pts;
}

/// Ground-truth T_W_C along a gentle S-curve down the room.
SE3 TruthPose(int k) {
    const double t = k * 0.05;
    SE3 T;
    T.R = core::SO3Exp(Vec3(0.02 * std::sin(t), 0.15 * std::sin(0.8 * t), 0.01 * t));
    T.t = Vec3(0.6 * std::sin(0.7 * t), 0.05 * std::sin(t), 0.35 * t);
    return T;
}

struct VoRun {
    std::vector<int> k;                  // frame index of each record (frames not fed are absent)
    std::vector<PoseSource> source;
    std::vector<uint32_t> segment;
    std::vector<SE3> est, truth;
    std::vector<VoStatus> status;
    std::vector<char> was_keyframe;
    int lost = 0, keyframes = 0;
    size_t edges = 0;
    std::vector<double> node_cov_trace;   // absolute covariance, per keyframe
    core::Mat6 last_node_cov = core::Mat6::Zero();
    SE3 last_node_est, last_node_truth;
    std::vector<core::Mat6> edge_information;
};

/// Feed the VO an ideal tracker's output: track id = point id, 0.3 px noise,
/// and `outlier_every`-th track corrupted by a 20 px jump. Frame
/// `blackout_frame` delivers no tracks at all (lamp failure, lens occlusion), and so do the
/// `blackout_frames - 1` frames after it. Inside the blackout only every `blackout_stride`-th frame is FED at all
/// (the others are dropped input: their timestamps never reach the VO).
/// A dark stretch (LOST_FRAMES.md): from `frame` for `frames` frames the 2-D tracks continue but carry ids no landmark
/// has (the map's tracks died) and no stereo (too dark for the right image); afterwards the tracks are a FRESH id set
/// with stereo, as after a real loss with no re-association.
struct DarkStretch {
    int frame = -1, frames = 0;
};

VoRun RunVo(int frames, int outlier_every, int blackout_frame = -1, int blackout_frames = 1, int blackout_stride = 1,
            const SlamConfig& cfg = Config(), DarkStretch dark = {}) {
    const std::vector<Vec3> pts = Room(1);
    StereoVo vo(cfg, kRig, 42);
    VoRun run;
    for (int k = 0; k < frames; ++k) {
        const bool in_blackout = blackout_frame >= 0 && k >= blackout_frame && k < blackout_frame + blackout_frames;
        if (in_blackout && (k - blackout_frame) % blackout_stride != 0) continue;   // dropped input frame
        // One stream per frame, keyed (seed, frame_id, site) as §5.6 keys all
        // randomness: a single stream for the whole sequence exhausts its
        // 2^18 words (the D18 guard throws, as it should).
        core::PhiloxStream noise(7, static_cast<uint64_t>(k), 0);
        const SE3 T_W_C = TruthPose(k);
        const SE3 T_C_W = T_W_C.inverse();
        VoFrame f;
        f.frame_id = static_cast<uint64_t>(k);
        f.t_ns = k * 50000000LL;
        VoStereo st;
        const bool in_dark = dark.frame >= 0 && k >= dark.frame && k < dark.frame + dark.frames;
        const uint64_t id_base = dark.frame < 0 || k < dark.frame ? 0 : in_dark ? 1000000 : 2000000;
        for (size_t j = 0; j < pts.size(); ++j) {
            if (in_blackout) break;
            const Vec3 Xc = T_C_W * pts[j];
            if (Xc.z() < 0.6 || Xc.z() > 6.0) continue;
            core::Vec2 l = core::Project(kRig.cam, Xc);
            core::Vec2 r = core::Project(kRig.cam, Xc - Vec3(kRig.baseline, 0, 0));
            if (l.x() < 1 || l.x() > kW - 2 || l.y() < 1 || l.y() > kH - 2) continue;
            const core::Vec2 n(core::SampleNormal(noise, 0, 0.3), core::SampleNormal(noise, 0, 0.3));
            l += n;
            r += core::Vec2(core::SampleNormal(noise, 0, 0.3), n.y());   // rectified: same row noise
            if (outlier_every > 0 && j % static_cast<size_t>(outlier_every) == 0 && k > 0) l += core::Vec2(20, -15);
            f.track_ids.push_back(id_base + j);
            f.points.emplace_back(float(l.x()), float(l.y()));
            st.right.emplace_back(float(r.x()), float(r.y()));
            st.valid.push_back(r.x() >= 0 && !in_dark ? 1 : 0);
            st.sgbm_z.push_back(std::numeric_limits<float>::quiet_NaN());
            st.sgbm_sigma.push_back(std::numeric_limits<float>::quiet_NaN());
        }
        const VoTrackResult tr = vo.Track(f);
        run.lost += tr.status == VoStatus::kLost;
        run.status.push_back(tr.status);
        run.source.push_back(tr.source);
        run.segment.push_back(tr.segment);
        const bool make_keyframe = tr.need_keyframe || (tr.reanchor_probe && vo.ShouldReanchor(st));
        run.was_keyframe.push_back(make_keyframe);
        SE3 est = tr.T_W_C;
        // The VO's world is the first camera: express truth the same way.
        const SE3 truth = TruthPose(0).inverse() * T_W_C;
        if (make_keyframe) {
            vo.AddKeyframe(st);
            ++run.keyframes;
            const PoseNode& node = vo.graph().nodes().rbegin()->second;
            est = node.T_W_C;   // BA-refined
            run.node_cov_trace.push_back(node.covariance.trace());
            run.last_node_cov = node.covariance;
            run.last_node_est = node.T_W_C;
            run.last_node_truth = truth;
        }
        run.k.push_back(k);
        run.est.push_back(est);
        run.truth.push_back(truth);
    }
    run.edges = vo.graph().edges().size();
    for (const auto& e : vo.graph().edges()) run.edge_information.push_back(e.information);
    return run;
}

double TranslationRmse(const VoRun& r) {
    double s = 0.0;
    for (size_t k = 0; k < r.est.size(); ++k) s += (r.est[k].t - r.truth[k].t).squaredNorm();
    return std::sqrt(s / r.est.size());
}

}  // namespace

TEST(StereoVo, TracksASyntheticSequenceEndToEnd) {
    const VoRun r = RunVo(200, 0);   // 10 s, ~3.5 m of travel
    EXPECT_EQ(r.lost, 0);
    EXPECT_GT(r.keyframes, 3);
    EXPECT_EQ(r.edges, static_cast<size_t>(r.keyframes - 1));
    const double rmse = TranslationRmse(r);
    std::printf("synthetic VO: %d keyframes, translation RMSE %.4f m over %.2f m\n", r.keyframes, rmse,
                r.truth.back().t.norm());
    EXPECT_LT(rmse, 0.05);

    // Node covariance is ABSOLUTE: composed along the chain, it never shrinks
    // (Invariant 7: no quiet overconfidence far from the gauge).
    for (size_t i = 1; i < r.node_cov_trace.size(); ++i) {
        EXPECT_GE(r.node_cov_trace[i], r.node_cov_trace[i - 1] - 1e-15) << "keyframe " << i;
    }
    // ... and the final keyframe's actual error is plausible under it:
    // e^T Sigma^-1 e below the chi^2_6 99.9% point (22.46).
    const core::Vec6 e = core::SE3Log(r.last_node_est * r.last_node_truth.inverse());
    const double nees = e.dot(r.last_node_cov.ldlt().solve(e));
    std::printf("final keyframe NEES under the composed covariance: %.2f\n", nees);
    EXPECT_LT(nees, 22.46);
}

// The lost path (never exercised by a healthy sequence): one frame with no
// tracks at all. The VO must say LOST, hold the constant-velocity pose, force a
// keyframe whose edge is uninformative, and resume tracking within a few frames.
TEST(StereoVo, BlackoutFrameIsFlaggedAndRecovers) {
    const int blackout = 60;
    const VoRun r = RunVo(100, 0, blackout);
    ASSERT_EQ(r.status[blackout], VoStatus::kLost);
    EXPECT_TRUE(r.was_keyframe[blackout]);
    const SE3 predicted = r.est[blackout - 1] * (r.est[blackout - 2].inverse() * r.est[blackout - 1]);
    EXPECT_LT(core::SE3Log(r.est[blackout] * predicted.inverse()).norm(), 1e-9);
    bool resumed = false;
    for (int k = blackout + 1; k < blackout + 4; ++k) resumed |= r.status[static_cast<size_t>(k)] == VoStatus::kTracking;
    EXPECT_TRUE(resumed) << "tracking did not resume within 3 frames";
    // Healthy edges carry large information; the blackout edge almost none.
    double min_info = 1e300, max_info = 0.0;
    for (const auto& I : r.edge_information) {
        min_info = std::min(min_info, I.trace());
        max_info = std::max(max_info, I.trace());
    }
    EXPECT_LT(min_info, 100.0);
    EXPECT_GT(max_info, 1e4);
    EXPECT_LT(TranslationRmse(r), 0.2);   // continuity held through the gap
}

// 10% of tracks jump 20 px each frame: USAC must isolate them, and the
// Huber-weighted refinement must not be dragged.
TEST(StereoVo, SurvivesCorruptedTracks) {
    const VoRun r = RunVo(120, 10);
    EXPECT_EQ(r.lost, 0);
    EXPECT_LT(TranslationRmse(r), 0.08);
}

// Replay determinism (the Phase 1 thin-run requirement): identical output,
// bit for bit, across OpenCV thread counts and global RNG state.
TEST(StereoVo, DeterministicAcrossThreadsAndGlobalRng) {
    const int def = cv::getNumThreads();
    std::vector<VoRun> runs;
    for (int threads : {1, 2, def}) {
        cv::setNumThreads(threads);
        for (int i = 0; i < 13; ++i) cv::theRNG().next();
        runs.push_back(RunVo(80, 10));
    }
    cv::setNumThreads(def);
    for (size_t r = 1; r < runs.size(); ++r) {
        ASSERT_EQ(runs[r].est.size(), runs[0].est.size());
        for (size_t k = 0; k < runs[0].est.size(); ++k) {
            ASSERT_EQ(0, std::memcmp(runs[r].est[k].matrix().data(), runs[0].est[k].matrix().data(), 16 * sizeof(double)))
                << "pose " << k << " differs in run " << r;
        }
    }
}

// Oracle (§11.6 "reference-check"): with Huber off, Ariadne's motion-only LM
// and OpenCV's solvePnPRefineLM minimise the same sum of squared reprojection
// errors from the same start, so they must land on the same pose.
TEST(StereoVo, MotionOnlyLmAgreesWithSolvePnPRefineLM) {
    const std::vector<Vec3> pts = Room(3);
    SE3 truth;
    truth.R = core::SO3Exp(Vec3(0.05, -0.1, 0.02));
    truth.t = Vec3(0.3, -0.1, 1.0);   // T_W_C
    const SE3 T_C_W = truth.inverse();
    core::PhiloxStream noise(11, 0, 0);
    std::vector<cv::Point3d> obj;
    std::vector<cv::Point2d> img;
    BaProblem p;
    p.cam = kRig.cam;
    for (const Vec3& X : pts) {
        const Vec3 Xc = T_C_W * X;
        if (Xc.z() < 0.6 || Xc.z() > 8.0) continue;
        core::Vec2 uv = core::Project(kRig.cam, Xc);
        if (uv.x() < 0 || uv.x() >= kW || uv.y() < 0 || uv.y() >= kH) continue;
        uv += core::Vec2(core::SampleNormal(noise, 0, 1.0), core::SampleNormal(noise, 0, 1.0));
        obj.emplace_back(X.x(), X.y(), X.z());
        img.emplace_back(uv.x(), uv.y());
        p.landmarks.push_back(X);
        p.obs.push_back({0, static_cast<int>(p.landmarks.size()) - 1, uv, Vec3::Zero(), 1.0});
    }
    ASSERT_GT(obj.size(), 100u);
    const SE3 start = core::SE3Exp((core::Vec6() << 0.02, -0.01, 0.03, 0.01, 0.005, -0.01).finished()) * T_C_W;
    p.poses = {start};
    p.pose_fixed = {0};
    p.landmark_fixed.assign(p.landmarks.size(), 1);
    BaOptions o;
    o.huber_delta = 1e9;
    o.max_iterations = 50;
    SolveReprojection(p, o);

    cv::Mat R(3, 3, CV_64F), rvec, tvec(3, 1, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R.at<double>(r, c) = start.R(r, c);
        tvec.at<double>(r) = start.t(r);
    }
    cv::Rodrigues(R, rvec);
    const cv::Mat K = (cv::Mat_<double>(3, 3) << kRig.cam.fx, 0, kRig.cam.cx, 0, kRig.cam.fy, kRig.cam.cy, 0, 0, 1);
    cv::solvePnPRefineLM(obj, img, K, cv::noArray(), rvec, tvec,
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 200, 1e-12));
    cv::Rodrigues(rvec, R);
    SE3 cvT;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) cvT.R(r, c) = R.at<double>(r, c);
        cvT.t(r) = tvec.at<double>(r);
    }
    EXPECT_LT(core::SE3Log(p.poses[0] * cvT.inverse()).norm(), 1e-6);
}

// --- §18.2 KeyframeBundle ------------------------------------------------------

TEST(KeyframeBundle, StampsTheGraphVersionAndCarriesTheNodePose) {
    PoseGraph g;
    PoseNode n0, n1;
    n0.keyframe = 0;
    n1.keyframe = 1;
    n1.T_W_C.t = Vec3(1, 2, 3);
    g.AddNode(n0);
    g.AddNode(n1);
    perception::StereoDepth d;
    d.Z = cv::Mat::zeros(48, 64, CV_16F);
    d.sigma_Z = cv::Mat::zeros(48, 64, CV_16F);
    const cv::Mat img(48, 64, CV_8U, cv::Scalar(7)), mask(48, 64, CV_8U, cv::Scalar(255));
    const auto b = runtime::BuildKeyframeBundle(g, 1, kRig.cam, img, d, mask);
    EXPECT_EQ(b.frame_id_of_pose, g.version());
    EXPECT_EQ(b.T_WiC.t, Vec3(1, 2, 3));
    EXPECT_FALSE(b.exposure_state.has_value());   // unset until §6.5, never faked
    // A later graph change makes the exported version stale: the service's cue to re-anchor.
    g.UpdateNode(1, SE3::Identity(), core::Mat6::Identity());
    EXPECT_LT(b.frame_id_of_pose, g.version());
    EXPECT_THROW(runtime::BuildKeyframeBundle(g, 9, kRig.cam, img, d, mask), std::invalid_argument);
}

// A long blackout coasts on the last MEASURED velocity. Re-deriving the velocity from the model's own extrapolations
// amplified the rounding error in R's orthonormality ~2.4x per frame: EuRoC MH_04 (48 lost frames) went non-finite.
TEST(StereoVo, LongBlackoutCoastsOnTheLastMeasuredVelocity) {
    const int blackout = 40, length = 300;
    const VoRun r = RunVo(blackout + length, 0, blackout, length);
    const SE3 v = r.est[blackout - 2].inverse() * r.est[blackout - 1];
    for (int k = blackout; k < blackout + length; ++k) {
        const SE3& T = r.est[static_cast<size_t>(k)];
        ASSERT_EQ(r.status[static_cast<size_t>(k)], VoStatus::kLost) << k;
        ASSERT_TRUE(T.R.allFinite() && T.t.allFinite()) << k;
        EXPECT_LT((T.R.transpose() * T.R - core::Mat3::Identity()).cwiseAbs().maxCoeff(), 1e-12) << k;
        const SE3 step = r.est[static_cast<size_t>(k) - 1].inverse() * T;
        EXPECT_LT(core::SE3Log(step * v.inverse()).norm(), 1e-9) << k;
    }
}

// Dropped input frames while coasting: the prediction follows elapsed TIME. Feeding every 3rd blackout frame, each
// coasted step must be the measured per-frame motion applied for 3 frame periods: Log(step) = 3 Log(v).
TEST(StereoVo, CoastingPredictsByElapsedTimeAcrossDroppedFrames) {
    const int blackout = 40, length = 60, stride = 3;
    const VoRun r = RunVo(blackout + length, 0, blackout, length, stride);
    ASSERT_EQ(r.k[size_t(blackout - 1)], blackout - 1);   // everything before the blackout was fed
    const core::Vec6 v = core::SE3Log(r.est[size_t(blackout - 2)].inverse() * r.est[size_t(blackout - 1)]);
    int checked = 0;
    for (size_t i = size_t(blackout); i < r.k.size(); ++i) {
        ASSERT_EQ(r.status[i], VoStatus::kLost) << r.k[i];
        const int frames_elapsed = r.k[i] - r.k[i - 1];
        const core::Vec6 step = core::SE3Log(r.est[i - 1].inverse() * r.est[i]);
        EXPECT_LT((step - double(frames_elapsed) * v).norm(), 1e-9) << "frame " << r.k[i] << ", " << frames_elapsed;
        ++checked;
    }
    EXPECT_EQ(checked, length / stride);
}

// ---------------------------------------------------------------------------------------------------------------------
// Lost-frame handling (LOST_FRAMES.md, D56).

namespace {

SlamConfig LostConfig(const std::string& motion, const std::string& reanchor) {
    SlamConfig c = Config();
    c.lost.motion = motion;
    c.lost.reanchor = reanchor;
    return c;
}

double RotationErrorDeg(const SE3& a, const SE3& b) { return core::SO3Log(a.R.transpose() * b.R).norm() * 180.0 / M_PI; }

size_t kTwoViewsCap = 300;
/// Two views of Room(3) from camera poses T0, T1 (T_W_C), 0.3 px noise: matched pixel lists.
void TwoViews(const SE3& T0, const SE3& T1, std::vector<cv::Point2f>& p0, std::vector<cv::Point2f>& p1, uint64_t seed) {
    const std::vector<Vec3> pts = Room(3);
    core::PhiloxStream noise(seed, 0, 1);
    for (const Vec3& X : pts) {
        const Vec3 a = T0.inverse() * X, b = T1.inverse() * X;
        if (a.z() < 0.6 || b.z() < 0.6) continue;
        const core::Vec2 u = core::Project(kRig.cam, a), v = core::Project(kRig.cam, b);
        if (u.x() < 0 || u.x() >= kW || u.y() < 0 || u.y() >= kH || v.x() < 0 || v.x() >= kW || v.y() < 0 || v.y() >= kH)
            continue;
        p0.emplace_back(float(u.x() + core::SampleNormal(noise, 0, 0.3)), float(u.y() + core::SampleNormal(noise, 0, 0.3)));
        p1.emplace_back(float(v.x() + core::SampleNormal(noise, 0, 0.3)), float(v.y() + core::SampleNormal(noise, 0, 0.3)));
        if (p0.size() == kTwoViewsCap) break;
    }
}

}  // namespace

// Pure rotation (a hover): every point moves by the same homography, the essential matrix is degenerate (t = 0), and
// the bridge must take H and recover the rotation.
TEST(RotationBridge, PureRotationSelectsTheHomography) {
    const SE3 T0 = TruthPose(20);
    SE3 T1 = T0;
    T1.R = T0.R * core::SO3Exp(Vec3(0.01, -0.03, 0.02));
    std::vector<cv::Point2f> p0, p1;
    TwoViews(T0, T1, p0, p1, 5);
    ASSERT_GE(p0.size(), 100u);
    const FrameRotation r = EstimateFrameRotation(p0, p1, kRig.cam, RotationBridgeParams{}, 42, 7);
    ASSERT_TRUE(r.ok) << r.why;
    EXPECT_EQ(r.model, 'H');
    EXPECT_LT(RotationErrorDeg(SE3{T0.R.transpose() * T1.R, Vec3::Zero()}, SE3{r.R_prev_cur, Vec3::Zero()}), 0.1);
}

// Real parallax (0.3 m sideways in a room 1-6 m deep): H no longer explains the near points, E does, and its rotation
// must be the one taken.
TEST(RotationBridge, ParallaxSelectsTheEssentialMatrix) {
    const SE3 T0 = TruthPose(20);
    SE3 T1 = T0;
    T1.R = T0.R * core::SO3Exp(Vec3(0.0, 0.05, 0.0));
    T1.t = T0.t + T0.R * Vec3(0.3, 0.0, 0.1);
    std::vector<cv::Point2f> p0, p1;
    TwoViews(T0, T1, p0, p1, 6);
    const FrameRotation r = EstimateFrameRotation(p0, p1, kRig.cam, RotationBridgeParams{}, 42, 8);
    ASSERT_TRUE(r.ok) << r.why;
    EXPECT_EQ(r.model, 'E');
    EXPECT_GT(r.score_f, r.score_h);
    EXPECT_LT(RotationErrorDeg(SE3{T0.R.transpose() * T1.R, Vec3::Zero()}, SE3{r.R_prev_cur, Vec3::Zero()}), 0.2);
}

TEST(RotationBridge, RefusesTooFewTracksAndIsAPureFunction) {
    const SE3 T0 = TruthPose(20);
    SE3 T1 = T0;
    T1.R = T0.R * core::SO3Exp(Vec3(0.0, 0.02, 0.0));
    std::vector<cv::Point2f> p0, p1;
    TwoViews(T0, T1, p0, p1, 9);
    RotationBridgeParams few;
    few.min_tracks = int(p0.size()) + 1;
    EXPECT_FALSE(EstimateFrameRotation(p0, p1, kRig.cam, few, 42, 9).ok);
    const FrameRotation a = EstimateFrameRotation(p0, p1, kRig.cam, RotationBridgeParams{}, 42, 9);
    cv::setNumThreads(4);
    const FrameRotation b = EstimateFrameRotation(p0, p1, kRig.cam, RotationBridgeParams{}, 42, 9);
    ASSERT_TRUE(a.ok && b.ok);
    EXPECT_EQ(std::memcmp(a.R_prev_cur.data(), b.R_prev_cur.data(), sizeof(double) * 9), 0);
    EXPECT_EQ(a.inliers_h, b.inliers_h);
    EXPECT_EQ(a.inliers_e, b.inliers_e);
}

// The default (no `lost:` section) is the 1.5 behaviour, byte for byte: the blackout test above already pins the
// coasted pose; here the config defaults themselves.
TEST(LostFrames, DefaultsAreThePhase1Behaviour) {
    const SlamConfig c = Config();
    EXPECT_EQ(c.lost.motion, "coast");
    EXPECT_EQ(c.lost.reanchor, "every_frame");
}

// A dark stretch: 2-D tracks survive, landmarks and stereo do not. bridge/gated must (1) report every dark frame as
// lost and BRIDGED in one new segment, (2) make no keyframe there except at the time cap, (3) keep the rotation
// measured, (4) re-anchor on the first frame whose stereo can seed PnP, then track.
TEST(LostFrames, DarkStretchIsBridgedWithoutAKeyframeFloodAndReanchors) {
    const int dark = 60, len = 30;   // 1.5 s
    const VoRun r = RunVo(dark + len + 20, 0, -1, 1, 1, LostConfig("bridge", "gated"), {dark, len});
    int kf_in_dark = 0;
    for (int k = dark; k < dark + len; ++k) {
        ASSERT_EQ(r.status[size_t(k)], VoStatus::kLost) << k;
        EXPECT_EQ(r.source[size_t(k)], PoseSource::kBridged) << k;
        EXPECT_EQ(r.segment[size_t(k)], 1u) << k;
        kf_in_dark += r.was_keyframe[size_t(k)];
    }
    EXPECT_EQ(r.segment[size_t(dark - 1)], 0u);
    EXPECT_LE(kf_in_dark, int(std::ceil(len * 0.05 / Config().keyframe.max_interval_s)));
    // Rotation over the stretch, measured from the tracks (the truth is a smooth S-curve, so the bar is tight).
    const SE3 est_rel = r.est[size_t(dark - 1)].inverse() * r.est[size_t(dark + len - 1)];
    const SE3 gt_rel = r.truth[size_t(dark - 1)].inverse() * r.truth[size_t(dark + len - 1)];
    EXPECT_LT(RotationErrorDeg(est_rel, gt_rel), 0.5);
    // Re-anchor at the first frame after the stretch (a fresh, stereo-rich track set), tracking right after.
    EXPECT_TRUE(r.was_keyframe[size_t(dark + len)]);
    for (int k = dark + len + 1; k < dark + len + 20; ++k) EXPECT_EQ(r.status[size_t(k)], VoStatus::kTracking) << k;
}

// The 1.5 behaviour on the same stretch, for contrast: a keyframe on every dark frame.
TEST(LostFrames, EveryFrameReanchorFloodsKeyframes) {
    const int dark = 60, len = 30;
    const VoRun r = RunVo(dark + len + 5, 0, -1, 1, 1, LostConfig("coast", "every_frame"), {dark, len});
    for (int k = dark; k < dark + len; ++k) {
        EXPECT_EQ(r.source[size_t(k)], PoseSource::kCoasted) << k;
        EXPECT_TRUE(r.was_keyframe[size_t(k)]) << k;
    }
}

// The bridge's translation decays: after many tau without measurements the pose stops moving (a hover is not
// extrapolated away), unlike constant-velocity coasting.
TEST(LostFrames, BridgedTranslationDecaysToRest) {
    const int dark = 60, len = 120;   // 6 s = 12 tau
    const VoRun r = RunVo(dark + len, 0, -1, 1, 1, LostConfig("bridge", "gated"), {dark, len});
    const double last_step = (r.est[size_t(dark + len - 1)].t - r.est[size_t(dark + len - 2)].t).norm();
    const double first_step = (r.est[size_t(dark + 1)].t - r.est[size_t(dark)].t).norm();
    EXPECT_GT(first_step, 0.0);
    EXPECT_LT(last_step, 1e-3 * first_step);
}

// The SHIPPED v2 path (sim_v2.yaml: window backend, bridge, every_frame) through a dark stretch: every dark frame
// bridged in one new segment, the rotation measured, every dark frame a keyframe (every_frame), and tracking back on
// the fresh track set right after.
TEST(LostFrames, ShippedV2DarkStretchIsBridgedAndRecovers) {
    const SlamConfig v2 = LoadSlam(std::string(ARIADNE_SOURCE_DIR) + "/configs/slam/sim_v2.yaml");
    ASSERT_EQ(v2.ba.backend, "window");
    ASSERT_EQ(v2.lost.motion, "bridge");
    ASSERT_EQ(v2.lost.reanchor, "every_frame");
    const int dark = 60, len = 30;
    const VoRun r = RunVo(dark + len + 20, 0, -1, 1, 1, v2, {dark, len});
    for (int k = dark; k < dark + len; ++k) {
        ASSERT_EQ(r.status[size_t(k)], VoStatus::kLost) << k;
        EXPECT_EQ(r.source[size_t(k)], PoseSource::kBridged) << k;
        EXPECT_EQ(r.segment[size_t(k)], 1u) << k;
        EXPECT_TRUE(r.was_keyframe[size_t(k)]) << k;
    }
    const SE3 est_rel = r.est[size_t(dark - 1)].inverse() * r.est[size_t(dark + len - 1)];
    const SE3 gt_rel = r.truth[size_t(dark - 1)].inverse() * r.truth[size_t(dark + len - 1)];
    EXPECT_LT(RotationErrorDeg(est_rel, gt_rel), 0.5);
    for (int k = dark + len + 2; k < dark + len + 20; ++k) EXPECT_EQ(r.status[size_t(k)], VoStatus::kTracking) << k;
}

// Determinism of the shipped v2 path with the bridge active: OpenCV thread count and global RNG state must not reach
// the poses (the bridge's USAC is seeded per (seed, frame, site) and single-threaded).
TEST(LostFrames, ShippedV2BridgeIsDeterministicAcrossThreadsAndGlobalRng) {
    const SlamConfig v2 = LoadSlam(std::string(ARIADNE_SOURCE_DIR) + "/configs/slam/sim_v2.yaml");
    const int def = cv::getNumThreads();
    std::vector<VoRun> runs;
    for (int threads : {1, 2, def}) {
        cv::setNumThreads(threads);
        for (int i = 0; i < 13; ++i) cv::theRNG().next();
        runs.push_back(RunVo(90, 10, -1, 1, 1, v2, {40, 25}));
    }
    cv::setNumThreads(def);
    for (size_t r = 1; r < runs.size(); ++r) {
        ASSERT_EQ(runs[r].est.size(), runs[0].est.size());
        for (size_t k = 0; k < runs[0].est.size(); ++k) {
            ASSERT_EQ(0, std::memcmp(runs[r].est[k].matrix().data(), runs[0].est[k].matrix().data(), 16 * sizeof(double)))
                << "pose " << k << " differs in run " << r;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// imu_mode attitude_only (2.4).

// The VO-aided bias: every tracked pair contributes (Log(R_gyro,raw) - Log(R_vo)) / dt; the median over the window
// recovers a planted bias despite 10% wild VO rotations.
TEST(Attitude, GyroBiasFromVoRotationsIsRobust) {
    const Vec3 b(0.02, -0.013, 0.008);
    slam::GyroBiasEstimator est(200);
    core::PhiloxStream rng(3, 0, 0);
    for (int k = 0; k < 300; ++k) {
        const Vec3 w(0.3 * std::sin(0.1 * k), 0.2, -0.1 * std::cos(0.07 * k));
        const double dt = 0.05;
        const core::Mat3 R_true = core::SO3Exp(w * dt);
        core::Mat3 R_vo = R_true * core::SO3Exp(Vec3(core::SampleNormal(rng, 0, 1e-4), core::SampleNormal(rng, 0, 1e-4),
                                                     core::SampleNormal(rng, 0, 1e-4)));
        if (k % 10 == 0) R_vo = core::SO3Exp(Vec3(0.2, -0.1, 0.3));   // a wild VO step
        est.Add(core::SO3Exp((w + b) * dt), R_vo, dt);
    }
    EXPECT_LT((est.Bias() - b).norm(), 5e-3);
}

TEST(Attitude, StationaryStartIsDetectedOnlyAtRest) {
    core::ImuNoise n;
    n.gyro_noise_density = 1.7e-4;
    n.rate_hz = 200;
    const Vec3 b(0.01, -0.02, 0.005);
    core::PhiloxStream rng(5, 0, 0);
    std::vector<core::ImuMeasurement> still, moving;
    const double sd = n.gyro_noise_density * std::sqrt(n.rate_hz);
    for (int k = 0; k < 600; ++k) {
        const Vec3 noise(core::SampleNormal(rng, 0, sd), core::SampleNormal(rng, 0, sd), core::SampleNormal(rng, 0, sd));
        still.push_back({k * 5000000LL, b + noise, Vec3(0, 0, 9.81)});
        moving.push_back({k * 5000000LL, b + noise + Vec3(0.5 * std::sin(0.05 * k), 0, 0), Vec3(0, 0, 9.81)});
    }
    Vec3 mean;
    ASSERT_TRUE(slam::StationaryGyroMean(still, n, 2.0, mean));
    EXPECT_LT((mean - b).norm(), 1e-3);
    EXPECT_FALSE(slam::StationaryGyroMean(moving, n, 2.0, mean));
}

// Mahony's filter (Theorem 5.1) at rest, started 20 deg off in roll: roll/pitch converge to the true gravity direction,
// and the gyro bias on the observable axes is learned.
TEST(Attitude, MahonyConvergesToGravity) {
    const core::Mat3 R_true = core::SO3Exp(Vec3(0.3, -0.2, 0.0));   // body tilted
    const Vec3 accel = R_true.transpose() * Vec3(0, 0, 9.81);         // specific force at rest
    const Vec3 b(0.01, -0.015, 0.0);
    slam::MahonyFilter f(1.0, 0.3);
    f.Init(core::SO3Exp(Vec3(0.35, 0, 0)).transpose() * accel, Vec3::Zero());   // wrong start
    for (int k = 0; k < 200 * 60; ++k) f.Update(b, accel, 0.005);                // 60 s, gyro reads only its bias
    const Vec3 g_true = R_true.transpose() * Vec3(0, 0, -1);
    EXPECT_LT(std::acos(std::min(1.0, f.GravityInBody().dot(g_true))) * 180 / M_PI, 0.05);
}
