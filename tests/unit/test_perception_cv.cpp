// Placeholder perception front-end (plan 1.4): ORB + grid selection, pyramidal
// LK tracking, epipolar rejection, SGBM depth. Synthetic scenes with a known
// answer; texture comes from core Philox, never cv::theRNG(), so the inputs are
// identical on every machine.
#include "core/random/philox.hpp"
#include "perception/features/grid_select.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/opencv/image_adapter.hpp"
#include "perception/stereo/sgbm_stereo.hpp"
#include "perception/stereo/stereo_rectifier.hpp"
#include "perception/tracking/klt_tracker.hpp"

#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <set>
#include <string>

using namespace ariadne;
using namespace ariadne::perception;

namespace {

constexpr int kW = 1280;
constexpr int kH = 720;

/// Band-limited random texture: white noise, blurred so its spectrum suits a
/// 21 px LK window and SGBM's 5 px blocks, then stretched to full range.
cv::Mat Texture(uint64_t seed, int w = kW, int h = kH, double blur_sigma = 1.5) {
    cv::Mat noise(h, w, CV_8U);
    for (int y = 0; y < h; ++y) {
        core::PhiloxStream s(seed, 0, static_cast<uint64_t>(y));
        uchar* row = noise.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) row[x] = static_cast<uchar>(s.NextBits32() & 0xFF);
    }
    cv::Mat blurred;
    cv::GaussianBlur(noise, blurred, cv::Size(0, 0), blur_sigma);
    cv::normalize(blurred, blurred, 0, 255, cv::NORM_MINMAX);
    return blurred;
}

/// dst(x, y) = src(x + dx, y + dy): the scene moved by (-dx, -dy).
cv::Mat Shift(const cv::Mat& src, double dx, double dy) {
    const cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::Mat dst;
    cv::warpAffine(src, dst, m, src.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REFLECT_101);
    return dst;
}

RectifiedStereoRig Rig() {
    RectifiedStereoRig r;
    r.width = kW;
    r.height = kH;
    r.fx = r.fy = 640.0;
    r.cx = 639.5;
    r.cy = 359.5;
    r.baseline_m = 0.10;
    return r;
}

/// The SHIPPED config, not struct defaults: tests must exercise what runs.
const PerceptionConfig& Placeholder() {
    static const PerceptionConfig c =
        LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/placeholder.yaml");
    return c;
}

FeatureParams Features() { return Placeholder().features; }

TrackerParams TrackerNoEpipolar() {
    TrackerParams t = Placeholder().tracker;
    t.f_min_points = INT_MAX;   // shift scenes are planar: F is degenerate on them
    return t;
}

double Median(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

}  // namespace

// --- Grayscale --------------------------------------------------------------

// Channel order must be explicit: pure red is luma 76 (0.299 * 255) read as
// RGB, and 29 (0.114 * 255) if the same bytes are misread as BGR.
TEST(PerceptionCv, GrayscaleHonoursChannelOrder) {
    core::ImageU8 red(4, 4, 3);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) red.At(x, y, 0) = 255;
    const cv::Mat view = WrapU8(red);
    EXPECT_EQ(ToGray(view, PixelFormat::kRgb8).at<uchar>(0, 0), 76);
    EXPECT_EQ(ToGray(view, PixelFormat::kBgr8).at<uchar>(0, 0), 29);
}

// --- Grid selection (§7.3) ----------------------------------------------------

// One very strong patch and a weak background: a global top-N would put every
// keypoint in the patch; grid selection must give every cell its share.
TEST(PerceptionCv, GridSelectSpreadsTheBudget) {
    std::vector<cv::KeyPoint> kps;
    for (int i = 0; i < 2000; ++i)   // strong, all in cell (0, 0)
        kps.emplace_back(cv::Point2f(10.0f + (i % 50), 10.0f + (i / 50) % 50), 7.f, -1.f, 100.f + i);
    for (int cy = 0; cy < 6; ++cy)
        for (int cx = 0; cx < 8; ++cx)
            for (int j = 0; j < 30; ++j)   // weak, everywhere
                kps.emplace_back(cv::Point2f(cx * 160.f + 5.f + j * 5.f, cy * 120.f + 60.f), 7.f,
                                 -1.f, 1.f + j);
    const auto sel = GridSelect(kps, cv::Size(kW, kH), 8, 6, 480);
    ASSERT_EQ(sel.size(), 480u);
    std::vector<int> per_cell(48, 0);
    for (const auto& k : sel) ++per_cell[GridCell(k.pt, cv::Size(kW, kH), 8, 6)];
    for (int c = 0; c < 48; ++c) EXPECT_GE(per_cell[c], 10) << "cell " << c;
}

// The result depends on the keypoint SET only: input order and response ties
// cannot change it.
TEST(PerceptionCv, GridSelectIsOrderIndependent) {
    std::vector<cv::KeyPoint> kps;
    for (int i = 0; i < 500; ++i)
        kps.emplace_back(cv::Point2f(float((i * 37) % kW), float((i * 91) % kH)), 7.f, -1.f,
                         float(i % 7));   // heavy ties
    auto rev = kps;
    std::reverse(rev.begin(), rev.end());
    const auto a = GridSelect(kps, cv::Size(kW, kH), 8, 6, 100);
    const auto b = GridSelect(rev, cv::Size(kW, kH), 8, 6, 100);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].pt, b[i].pt);
}

// --- ORB engine ----------------------------------------------------------------

TEST(PerceptionCv, OrbDescriptorsAlignWithKeypoints) {
    const OrbEngine orb(Features());
    const FeatureSet fs = orb.DetectAndCompute(Texture(1), cv::Mat());
    ASSERT_FALSE(fs.keypoints.empty());
    EXPECT_LE(static_cast<int>(fs.keypoints.size()), Features().max_keypoints);
    EXPECT_EQ(fs.descriptors.rows, static_cast<int>(fs.keypoints.size()));
    EXPECT_EQ(fs.descriptors.type(), CV_8U);
    EXPECT_EQ(fs.descriptors.cols, 32);   // 256-bit rBRIEF
}

TEST(PerceptionCv, OrbOnABlankImageReturnsNothing) {
    const OrbEngine orb(Features());
    EXPECT_TRUE(orb.DetectAndCompute(cv::Mat(kH, kW, CV_8U, cv::Scalar(128)), cv::Mat())
                    .keypoints.empty());
}

// ORB-SLAM's two-threshold idea: a dim, low-contrast half of the image yields
// few corners at the normal FAST threshold. The low-threshold pass must put
// keypoints there.
TEST(PerceptionCv, OrbLowThresholdRescuesDarkCells) {
    cv::Mat img = Texture(2);
    cv::Mat dark = img(cv::Rect(kW / 2, 0, kW / 2, kH));
    // Contrast / 4. At / 12 (tried first) local differences fall below even
    // the low FAST threshold of 7 and NO corner detector threshold helps:
    // that regime needs the radiometric chain of §6.2, not a lower threshold.
    dark.convertTo(dark, -1, 0.25, 10);
    auto count_right = [&](const FeatureParams& p) {
        int n = 0;
        for (const auto& k : OrbEngine(p).Detect(img, cv::Mat(), 0)) n += k.pt.x >= kW / 2;
        return n;
    };
    FeatureParams single = Features();
    single.orb_fast_threshold_min = single.orb_fast_threshold;   // rescue disabled
    const int without = count_right(single);
    const int with = count_right(Features());
    EXPECT_GT(with, without);
    EXPECT_GE(with, Features().max_keypoints / 4);   // the dark half gets a real share
}

// --- LK tracker ----------------------------------------------------------------

// A known sub-pixel scene motion must be recovered to well under a pixel, and
// track identities must persist.
TEST(PerceptionCv, TrackerRecoversKnownShift) {
    const OrbEngine orb(Features());
    KltTracker tracker(TrackerNoEpipolar(), orb);
    const cv::Mat f0 = Texture(3);
    const cv::Mat f1 = Shift(f0, 3.3, -1.7);   // scene moves by (-3.3, +1.7)

    const TrackedFrame t0 = tracker.Process(f0, 0, 42);
    EXPECT_EQ(t0.state, TrackingState::kInitializing);
    ASSERT_GT(t0.ids.size(), 500u);
    const TrackedFrame t1 = tracker.Process(f1, 1, 42);
    EXPECT_EQ(t1.state, TrackingState::kTracking);

    std::set<uint64_t> first(t0.ids.begin(), t0.ids.end());
    std::vector<double> err;
    for (size_t i = 0; i < t1.ids.size(); ++i) {
        if (t1.ages[i] == 0) continue;
        EXPECT_TRUE(first.count(t1.ids[i]));
        const cv::Point2f d = t1.points[i] - t1.prev_points[i];
        err.push_back(std::hypot(d.x + 3.3, d.y - 1.7));
    }
    ASSERT_GT(err.size(), t0.ids.size() * 9 / 10);   // >90% survive a clean shift
    EXPECT_LT(Median(err), 0.05);
}

// §8.2: when support collapses the tracker says so.
TEST(PerceptionCv, TrackerReportsDegradedOnABlankFrame) {
    const OrbEngine orb(Features());
    KltTracker tracker(TrackerNoEpipolar(), orb);
    tracker.Process(Texture(4), 0, 42);
    const TrackedFrame t = tracker.Process(cv::Mat(kH, kW, CV_8U, cv::Scalar(128)), 1, 42);
    EXPECT_EQ(t.state, TrackingState::kDegraded);
    EXPECT_EQ(t.stats.replenished, 0);
}

// Tracks keep the configured spacing: older tracks win, newcomers respect it.
TEST(PerceptionCv, TrackerKeepsMinimumSpacing) {
    const OrbEngine orb(Features());
    const TrackerParams params = TrackerNoEpipolar();
    KltTracker tracker(params, orb);
    tracker.Process(Texture(5), 0, 42);
    const TrackedFrame t = tracker.Process(Shift(Texture(5), 1.0, 0.5), 1, 42);
    // The mask is drawn with integer radius at rounded centres: allow 1.5 px.
    const double min_d = params.min_track_distance_px - 1.5;
    for (size_t i = 0; i < t.points.size(); ++i)
        for (size_t j = i + 1; j < t.points.size(); ++j)
            ASSERT_GE(cv::norm(t.points[i] - t.points[j]), min_d);
}

// --- Epipolar rejection (non-planar geometry) ----------------------------------

// Points spread through a 3D volume, seen from two poses; 20% of the second
// view's points displaced by 15 px. The robust F fit must separate them.
TEST(PerceptionCv, EpipolarRejectsOutliersOnNonPlanarScene) {
    const double f = 640.0, cx = 639.5, cy = 359.5;
    const double yaw = 0.05, tx = 0.3, tz = 0.05;
    std::vector<cv::Point2f> p0, p1;
    std::vector<bool> is_outlier;
    core::PhiloxStream s(7, 0, 0);
    auto u = [&s] { return (s.NextBits32() >> 8) * (1.0 / 16777216.0); };
    while (p0.size() < 400) {
        const double X = -2 + 4 * u(), Y = -1.2 + 2.4 * u(), Z = 2 + 4 * u();
        // camera 2: rotate about y by yaw, then translate
        const double X2 = std::cos(yaw) * X + std::sin(yaw) * Z - tx;
        const double Z2 = -std::sin(yaw) * X + std::cos(yaw) * Z - tz;
        cv::Point2f a(float(f * X / Z + cx), float(f * Y / Z + cy));
        cv::Point2f b(float(f * X2 / Z2 + cx), float(f * Y / Z2 + cy));
        const bool out = p0.size() % 5 == 0;
        if (out) b += cv::Point2f(15.f * float(std::cos(p0.size())), 15.f * float(std::sin(p0.size())));
        p0.push_back(a);
        p1.push_back(b);
        is_outlier.push_back(out);
    }
    const TrackerParams params = Placeholder().tracker;
    const EpipolarResult r = EpipolarInliers(p0, p1, params, 42, 1);
    ASSERT_TRUE(r.ran);
    int tp = 0, fn = 0, fp = 0;
    for (size_t i = 0; i < p0.size(); ++i) {
        if (is_outlier[i] && !r.inlier[i]) ++tp;
        if (is_outlier[i] && r.inlier[i]) ++fn;
        if (!is_outlier[i] && !r.inlier[i]) ++fp;
    }
    EXPECT_GE(tp, 76);   // >= 95% of the 80 outliers caught
    EXPECT_LE(fp, 16);   // <= 5% of the 320 inliers lost
    // Same (seed, frame) -> same decision, whatever cv::theRNG() did meanwhile.
    cv::theRNG().next();
    cv::theRNG().next();
    EXPECT_EQ(EpipolarInliers(p0, p1, params, 42, 1).inlier, r.inlier);
    (void)fn;
}

// --- SGBM depth (§9.3-9.6) ---------------------------------------------------

namespace {

/// A fronto-parallel plane at depth z: constant disparity f B / z.
StereoDepth PlaneAt(double z, uint64_t seed) {
    const RectifiedStereoRig rig = Rig();
    const cv::Mat left = Texture(seed);
    const double d = rig.fx * rig.baseline_m / z;
    const cv::Mat right = Shift(left, d, 0.0);   // right(x) = left(x + d)
    return SgbmStereo(Placeholder().stereo, rig).Compute(left, right);
}

}  // namespace

TEST(PerceptionCv, SgbmRecoversAPlaneWithinSigma) {
    const double z_true = 1.5;   // d = 42.67 px, sub-pixel
    const StereoDepth depth = PlaneAt(z_true, 8);
    EXPECT_EQ(depth.disparity.type(), CV_16S);
    EXPECT_EQ(depth.Z.type(), CV_16F);
    EXPECT_TRUE(depth.confidence_is_placeholder);
    EXPECT_EQ(depth.blind_columns, 256);

    cv::Mat z32, s32;
    depth.Z.convertTo(z32, CV_32F);
    depth.sigma_Z.convertTo(s32, CV_32F);
    int within = 0, valid = 0;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            if (!depth.valid.at<uchar>(y, x)) continue;
            ++valid;
            within += std::abs(z32.at<float>(y, x) - z_true) <= 3.0 * s32.at<float>(y, x);
        }
    const int usable = (kW - depth.blind_columns) * kH;
    EXPECT_GT(valid, usable * 8 / 10);
    EXPECT_GT(within, valid * 95 / 100);
    // A real surface is densely supported (only a rim near the image and
    // blind-column edges falls under the gate).
    EXPECT_GT(cv::countNonZero(DenseSupportMask(depth, Placeholder().stereo)), valid * 9 / 10);
    EXPECT_NEAR(SgbmStereo::DepthSigma(1.5, 640, 0.1, 0.5), 1.5 * 1.5 / 64.0 * 0.5, 1e-12);
}

// D18 spirit: evaluate the ends of the domain. Just inside the search range
// (d ~246 of max 255) the plane must be measured.
//
// Beyond it (d = 320) every "valid" pixel is an alias, and the probe behind
// D34 found they report depths spread from 0.3 m to past 3 m: phantom FAR
// depth, which a free-space carving consumer (1.8) would turn into phantom
// free space through a real obstacle. SGBM alone cannot eliminate this; WLS
// confidence and a near-field check are Phase 2. This test bounds the measured
// rate so it cannot silently get worse (0.078% of pixels at uniqueness 15 and
// speckle window 200), and requires the dense-support gate to reject ALL of
// the aliases, since that gate is what 1.8 relies on before carving free space.
TEST(PerceptionCv, SgbmNearRangeBoundary) {
    const StereoDepth inside = PlaneAt(0.26, 9);
    EXPECT_GT(inside.valid_fraction, 0.5);

    const StereoDepth beyond = PlaneAt(0.20, 10);
    EXPECT_LT(beyond.valid_fraction, 0.002)
        << "near-range aliasing rose above the D34 bound: " << beyond.valid_fraction;
    EXPECT_EQ(cv::countNonZero(DenseSupportMask(beyond, Placeholder().stereo)), 0)
        << "an aliased near-range depth passed the dense-support gate";
}

// --- Rectification (§9.2) ---------------------------------------------------

namespace {

/// A distorted, slightly toed-in rig resembling EuRoC's (752x480, radtan,
/// B = 0.11 m), with the right camera rotated by `pitch` and 0.6 deg in yaw.
RawStereoCalibration SyntheticRawRig(double pitch = 0.004) {
    RawStereoCalibration c;
    c.left = {752, 480, 458.0, 457.0, 367.0, 248.0, {-0.28, 0.074, 0.0002, 0.00002}};
    c.right = {752, 480, 457.5, 456.0, 380.0, 255.0, {-0.283, 0.0745, -0.0001, -0.00003}};
    c.T_right_left.R = core::SO3Exp(core::Vec3(pitch, 0.01, 0.002));
    c.T_right_left.t = core::Vec3(-0.110, 0.0002, -0.0009);
    return c;
}

cv::Mat CvMat3(const core::Mat3& m) {
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out.at<double>(r, c) = m(r, c);
    return out;
}

}  // namespace

// The whole rectification chain, checked geometrically without any image:
// project 3D points through the RAW distorted cameras, map the raw pixels into
// rectified coordinates with the rectifier's R_i and P_i, and require (a) the
// same row in both images and (b) disparity = f B / Z.
TEST(PerceptionCv, RectificationPutsCorrespondencesOnOneRowWithMetricDisparity) {
    const RawStereoCalibration raw = SyntheticRawRig();
    const StereoRectifier rect(raw);
    const RectifiedStereoRig& rig = rect.Rig();
    EXPECT_NEAR(rig.baseline_m, raw.T_right_left.t.norm(), 1e-9);
    EXPECT_NEAR(rect.P2().at<double>(1, 3), 0.0, 1e-9);

    auto K = [](const PinholeRadtan& c) -> cv::Mat {
        return (cv::Mat_<double>(3, 3) << c.fu, 0, c.cu, 0, c.fv, c.cv, 0, 0, 1);
    };
    auto D = [](const PinholeRadtan& c) -> cv::Mat {
        return (cv::Mat_<double>(1, 4) << c.dist[0], c.dist[1], c.dist[2], c.dist[3]);
    };
    std::vector<cv::Point3d> pts;
    core::PhiloxStream s(21, 0, 0);
    auto u = [&s] { return (s.NextBits32() >> 8) * (1.0 / 16777216.0); };
    for (int i = 0; i < 300; ++i) {
        const double z = 1.0 + 7.0 * u();
        pts.emplace_back((u() - 0.5) * 0.8 * z, (u() - 0.5) * 0.5 * z, z);   // inside both views
    }
    std::vector<cv::Point2d> raw_l, raw_r;
    cv::Mat rvec;
    cv::Rodrigues(CvMat3(raw.T_right_left.R), rvec);
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << raw.T_right_left.t.x(), raw.T_right_left.t.y(),
                          raw.T_right_left.t.z());
    const cv::Mat zero3 = cv::Mat::zeros(3, 1, CV_64F);
    cv::projectPoints(pts, zero3, zero3, K(raw.left),
                      D(raw.left), raw_l);
    cv::projectPoints(pts, rvec, tvec, K(raw.right), D(raw.right), raw_r);

    std::vector<cv::Point2d> rl, rr;
    cv::undistortPoints(raw_l, rl, K(raw.left), D(raw.left), rect.R1(), rect.P1());
    cv::undistortPoints(raw_r, rr, K(raw.right), D(raw.right), rect.R2(), rect.P2());
    double worst_dy = 0.0, worst_depth_rel = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        worst_dy = std::max(worst_dy, std::abs(rl[i].y - rr[i].y));
        // Depth along the RECTIFIED left optical axis: rotate the point by R1.
        const cv::Mat p = rect.R1() * (cv::Mat_<double>(3, 1) << pts[i].x, pts[i].y, pts[i].z);
        const double z_rect = p.at<double>(2);
        const double z_from_d = rig.fx * rig.baseline_m / (rl[i].x - rr[i].x);
        worst_depth_rel = std::max(worst_depth_rel, std::abs(z_from_d - z_rect) / z_rect);
    }
    // undistortPoints inverts the distortion iteratively; 1e-3 px is well
    // below any pixel-level effect and far below a wrong-transform error.
    EXPECT_LT(worst_dy, 1e-3);
    EXPECT_LT(worst_depth_rel, 1e-4);
}

// Negative control for the check above: the same measurement against a rig
// whose rotation is wrong by 1 deg of pitch must fail it. Otherwise the check
// could not have caught anything.
TEST(PerceptionCv, RectificationCheckDetectsAWrongCalibration) {
    const RawStereoCalibration truth = SyntheticRawRig();
    RawStereoCalibration wrong = truth;
    wrong.T_right_left.R = core::SO3Exp(core::Vec3(0.004 + 0.01745, 0.01, 0.002));
    const StereoRectifier rect(wrong);
    auto K = [](const PinholeRadtan& c) -> cv::Mat {
        return (cv::Mat_<double>(3, 3) << c.fu, 0, c.cu, 0, c.fv, c.cv, 0, 0, 1);
    };
    auto D = [](const PinholeRadtan& c) -> cv::Mat {
        return (cv::Mat_<double>(1, 4) << c.dist[0], c.dist[1], c.dist[2], c.dist[3]);
    };
    std::vector<cv::Point3d> pts{{0.2, 0.1, 3.0}, {-0.5, -0.2, 2.0}, {0.1, 0.3, 5.0}};
    std::vector<cv::Point2d> raw_l, raw_r, rl, rr;
    cv::Mat rvec;
    cv::Rodrigues(CvMat3(truth.T_right_left.R), rvec);   // the TRUE rig produced the pixels
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << -0.110, 0.0002, -0.0009);
    const cv::Mat zero3 = cv::Mat::zeros(3, 1, CV_64F);
    cv::projectPoints(pts, zero3, zero3, K(truth.left),
                      D(truth.left), raw_l);
    cv::projectPoints(pts, rvec, tvec, K(truth.right), D(truth.right), raw_r);
    cv::undistortPoints(raw_l, rl, K(truth.left), D(truth.left), rect.R1(), rect.P1());
    cv::undistortPoints(raw_r, rr, K(truth.right), D(truth.right), rect.R2(), rect.P2());
    for (size_t i = 0; i < pts.size(); ++i) EXPECT_GT(std::abs(rl[i].y - rr[i].y), 5.0);
}

// A transform in the wrong direction (right->left instead of left->right)
// yields a negative baseline; the rectifier refuses it.
TEST(PerceptionCv, RectifierRejectsAnInvertedExtrinsic) {
    RawStereoCalibration c = SyntheticRawRig();
    c.T_right_left = c.T_right_left.inverse();
    EXPECT_THROW(StereoRectifier{c}, std::invalid_argument);
}

// --- Determinism ----------------------------------------------------------------

// Same inputs, same outputs, bit for bit: across repeated runs, across
// cv::theRNG() state, and across OpenCV thread counts.
TEST(PerceptionCv, OutputsIndependentOfThreadCountAndGlobalRng) {
    const cv::Mat f0 = Texture(11), f1 = Shift(f0, 2.2, 0.9), right = Shift(f0, 30.25, 0.0);
    struct Run {
        std::vector<uint64_t> ids;
        std::vector<cv::Point2f> pts;
        cv::Mat disparity;
        cv::Mat desc;
        cv::Mat rectified;
    };
    const StereoRectifier rectifier(SyntheticRawRig());
    const cv::Mat raw752 = Texture(12, 752, 480);
    auto run = [&](int threads) {
        cv::setNumThreads(threads);
        for (int i = 0; i < 17; ++i) cv::theRNG().next();
        const OrbEngine orb(Features());
        // Shipped tracker params, epipolar check ON: it must be deterministic too.
        KltTracker tracker(Placeholder().tracker, orb);
        tracker.Process(f0, 0, 99);
        const TrackedFrame t = tracker.Process(f1, 1, 99);
        cv::Mat rl, rr;
        rectifier.Rectify(raw752, raw752, rl, rr);
        return Run{t.ids, t.points, SgbmStereo(Placeholder().stereo, Rig()).Compute(f0, right).disparity,
                   orb.DetectAndCompute(f0, cv::Mat()).descriptors, rl};
    };
    const int default_threads = cv::getNumThreads();
    const Run ref = run(1);
    for (int threads : {1, 2, default_threads}) {
        const Run r = run(threads);
        EXPECT_EQ(r.ids, ref.ids) << "threads=" << threads;
        ASSERT_EQ(r.pts.size(), ref.pts.size()) << "threads=" << threads;
        EXPECT_EQ(0, std::memcmp(r.pts.data(), ref.pts.data(), r.pts.size() * sizeof(cv::Point2f)))
            << "tracks differ at threads=" << threads;
        EXPECT_EQ(cv::norm(r.disparity, ref.disparity, cv::NORM_INF), 0.0)
            << "SGBM differs at threads=" << threads;
        EXPECT_EQ(cv::norm(r.desc, ref.desc, cv::NORM_INF), 0.0) << "ORB differs at threads=" << threads;
        EXPECT_EQ(cv::norm(r.rectified, ref.rectified, cv::NORM_INF), 0.0)
            << "remap differs at threads=" << threads;
    }
    cv::setNumThreads(default_threads);
}
