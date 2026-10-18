// Placeholder perception front-end (plan 1.4): ORB + grid selection, pyramidal
// LK tracking, epipolar rejection, SGBM depth. Synthetic scenes with a known
// answer; texture comes from core Philox, never cv::theRNG(), so the inputs are
// identical on every machine.
#include "core/random/philox.hpp"
#include "perception/features/grid_select.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/opencv/image_adapter.hpp"
#include "perception/stereo/sgbm_stereo.hpp"
#include "perception/stereo/stereo_lk_matcher.hpp"
#include "perception/stereo/stereo_rectifier.hpp"
#include "perception/tracking/klt_tracker.hpp"

#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <set>
#include <cstdlib>
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

// --- Sparse stereo matcher ------------------------------------------------------

// A known disparity must be recovered to sub-pixel; a pure VERTICAL offset (a
// broken rectification) must be rejected by the row gate, not matched.
TEST(PerceptionCv, StereoMatcherRecoversDisparityAndRejectsRowOffsets) {
    const cv::Mat left = Texture(31, 752, 480);
    std::vector<cv::Point2f> pts;
    for (int y = 40; y < 440; y += 40)
        for (int x = 120; x < 700; x += 40) pts.emplace_back(float(x), float(y));
    const StereoMatchParams p = Placeholder().stereo_match;

    const cv::Mat right = Shift(left, 17.25, 0.0);   // right(x) = left(x + 17.25): disparity 17.25
    const StereoMatches m = MatchLeftToRight(left, right, pts, p);
    int ok = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!m.valid[i]) continue;
        ++ok;
        EXPECT_NEAR(pts[i].x - m.right[i].x, 17.25, 0.05);
        EXPECT_NEAR(m.right[i].y, pts[i].y, 0.05);
    }
    EXPECT_GT(ok, static_cast<int>(pts.size()) * 9 / 10);

    const cv::Mat shifted_rows = Shift(left, 17.25, 3.0);   // 3 px vertical error
    const StereoMatches bad = MatchLeftToRight(left, shifted_rows, pts, p);
    int accepted = 0;
    for (uchar v : bad.valid) accepted += v;
    EXPECT_EQ(accepted, 0);
    EXPECT_GT(bad.row_rejected, static_cast<int>(pts.size()) * 9 / 10);
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

// SMOKE §4q: OpenCV 5.0.0's USAC findFundamentalMat throws CV_Assert(!model.empty()) on some inputs. Replays the captured
// EuRoC V1_01_easy frame-1009 input (.cache/fixtures/usac/, EuRoC-derived, so not in the repo; skipped without it):
//  1. the defect itself, at the draw the tracker made -- if an OpenCV upgrade fixes it, this fails and §4q is revisited;
//  2. EpipolarInliers absorbs it with a redraw from the same Philox stream and still runs the check.
TEST(KltTracker, UsacEmptyModelDefectIsAbsorbedByARedraw) {
    const char* path = std::getenv("ARIADNE_USAC_FIXTURE");
    if (!path) GTEST_SKIP() << "set ARIADNE_USAC_FIXTURE (.cache/fixtures/usac/V1_01_easy_frame1009.yml)";
    cv::FileStorage fs(path, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened()) << path;
    std::vector<cv::Point2f> p0, p1;
    int rng = 0, seed = 0, frame = 0;
    fs["p0"] >> p0;
    fs["p1"] >> p1;
    fs["rng"] >> rng;
    fs["seed"] >> seed;
    fs["frame"] >> frame;
    const TrackerParams params =
        LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/euroc.yaml").tracker;
    cv::UsacParams u;
    u.threshold = params.f_ransac_threshold_px;
    u.confidence = params.f_confidence;
    u.maxIterations = params.f_max_iterations;
    u.isParallel = false;
    u.randomGeneratorState = rng;
    std::vector<uchar> mask;
    try {
        cv::findFundamentalMat(p0, p1, mask, u);
        ADD_FAILURE() << "the OpenCV USAC defect no longer reproduces: revisit SMOKE §4q and the redraw";
    } catch (const cv::Exception& e) {
        EXPECT_EQ(e.code, cv::Error::StsAssert);
        EXPECT_NE(e.err.find("!model.empty()"), std::string::npos) << e.err;
    }
    const EpipolarResult r = EpipolarInliers(p0, p1, params, uint64_t(seed), uint64_t(frame));
    EXPECT_GE(r.usac_defect_retries, 1);
    EXPECT_FALSE(r.usac_defect_skipped);
    EXPECT_TRUE(r.ran);
    EXPECT_EQ(r.inlier.size(), p0.size());
}

// --- Online row refinement (AUDIT_2.3.md, D54) -------------------------------------------------------------------

namespace {

/// A rectified pair with a known residual misalignment: a left point (x_r + d, y_r - dy(x_r, y_r)) appears in the
/// right image at (x_r, y_r), with dy = a + b x_r + c y_r in RIGHT pixel coordinates (EstimateRowCorrection's model).
cv::Mat RightWithRowError(const cv::Mat& left, double d, double a, double b, double c) {
    cv::Mat mx(left.size(), CV_32F), my(left.size(), CV_32F);
    for (int y = 0; y < left.rows; ++y)
        for (int x = 0; x < left.cols; ++x) {
            mx.at<float>(y, x) = float(x + d);
            my.at<float>(y, x) = float(y - (a + b * x + c * y));
        }
    cv::Mat right;
    cv::remap(left, right, mx, my, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return right;
}

std::vector<cv::Point2f> Grid(int w, int h) {
    std::vector<cv::Point2f> pts;
    for (int y = 30; y < h - 30; y += 20)
        for (int x = 60; x < w - 30; x += 20) pts.emplace_back(float(x), float(y));
    return pts;
}

}  // namespace

// The estimator recovers an injected offset + slope with the documented sign (dy = y_right - y_left, right pixel
// coordinates), and reads ~0 on an aligned pair: the "correct rig" control.
TEST(StereoRowRefinement, EstimatorRecoversAKnownResidualAndZeroOnAnAlignedPair) {
    const cv::Mat left = Texture(41, 752, 480);
    StereoMatchParams p = LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/euroc.yaml").stereo_match;
    const std::vector<cv::Point2f> pts = Grid(752, 480);
    const double a = -1.30, b = 0.23 / 752.0, c = -1.10 / 480.0;   // the V2_01 shape (AUDIT_2.3.md §1)
    const RowCorrection est = EstimateRowCorrection(left, RightWithRowError(left, 12.5, a, b, c), pts, p);
    ASSERT_TRUE(est.ok) << est.why;
    EXPECT_TRUE(est.slopes_fitted);
    EXPECT_NEAR(est.a, a, 0.02);
    EXPECT_NEAR(est.b * 752.0, b * 752.0, 0.02);   // compared as the change across the image, in px
    EXPECT_NEAR(est.c * 480.0, c * 480.0, 0.02);
    // The strict matcher rejects most of these rows before refinement...
    const StereoMatches before = MatchLeftToRight(left, RightWithRowError(left, 12.5, a, b, c), pts, p);
    EXPECT_GT(before.row_rejected, int(pts.size()) / 2);

    const RowCorrection zero = EstimateRowCorrection(left, RightWithRowError(left, 12.5, 0, 0, 0), pts, p);
    ASSERT_TRUE(zero.ok) << zero.why;
    EXPECT_LT(std::abs(zero.a) + std::abs(zero.b * 752.0) + std::abs(zero.c * 480.0), 0.02);
}

// Refusals are loud: too few matches, or a model that would exceed the search bound somewhere in the image.
TEST(StereoRowRefinement, EstimatorRefusesWhatItCannotTrust) {
    const cv::Mat left = Texture(42, 752, 480);
    StereoMatchParams p = LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/euroc.yaml").stereo_match;
    const RowCorrection few = EstimateRowCorrection(left, RightWithRowError(left, 10.0, 0.5, 0, 0),
                                                    {cv::Point2f(100, 100), cv::Point2f(200, 200)}, p);
    EXPECT_FALSE(few.ok);
    EXPECT_NE(few.why.find("row_min_matches"), std::string::npos) << few.why;
    // dy runs from 0 at the top to 6 px at the bottom: inside the 4 px search bound only in the upper part, and the
    // fitted model then predicts > 4 px at the bottom corners -> refused.
    const RowCorrection steep = EstimateRowCorrection(left, RightWithRowError(left, 10.0, 0.0, 0.0, 6.0 / 480.0),
                                                      Grid(752, 480), p);
    EXPECT_FALSE(steep.ok);
    EXPECT_FALSE(steep.why.empty());
}

// RefineRightRows moves right-image content by exactly -dy: a right pixel (x, y) lands at (x, y - (a + b x + c y)).
// Measured with LK between the right image rectified before and after (the same raw image), so the sign of the fold
// into the camera matrix is tested on real maps, not on algebra.
TEST(StereoRowRefinement, RefineRightRowsMovesTheRightImageByMinusDy) {
    const RawStereoCalibration raw = SyntheticRawRig();
    StereoRectifier rect(raw);
    const cv::Mat raw_left = Texture(43, 752, 480, 2.0), raw_right = Texture(44, 752, 480, 2.0);
    cv::Mat l0, r0, l1, r1;
    rect.Rectify(raw_left, raw_right, l0, r0);
    const double a = 1.4, b = -0.3 / 752.0, c = 1.2 / 480.0;
    rect.RefineRightRows(a, b, c);
    EXPECT_EQ(rect.row_a(), a);
    EXPECT_NEAR(rect.P2().at<double>(1, 3), 0.0, 1e-9);   // the ideal rig model is untouched
    rect.Rectify(raw_left, raw_right, l1, r1);
    EXPECT_EQ(cv::norm(l0, l1, cv::NORM_INF), 0.0);       // left map untouched
    const std::vector<cv::Point2f> pts = Grid(752, 480);
    std::vector<cv::Point2f> moved = pts;
    std::vector<uchar> st;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(r0, r1, pts, moved, st, err, cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001),
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    int checked = 0;
    double worst = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!st[i]) continue;
        const double dy = a + b * pts[i].x + c * pts[i].y;
        worst = std::max(worst, std::abs((moved[i].y - pts[i].y) - (-dy)) + std::abs(moved[i].x - pts[i].x));
        ++checked;
    }
    EXPECT_GT(checked, int(pts.size()) * 8 / 10);
    EXPECT_LT(worst, 0.1);
}

// Exposure: a right image at 0.6x gain and +20 bias is brought back to the left's statistics.
TEST(StereoRowRefinement, MatchExposureRemovesAGlobalGainAndBias) {
    const cv::Mat left = Texture(45, 752, 480);
    cv::Mat right;
    left.convertTo(right, CV_8U, 0.6, 20.0);
    const cv::Mat fixed = MatchExposure(left, right);
    cv::Scalar ml, sl, mf, sf;
    cv::meanStdDev(left, ml, sl);
    cv::meanStdDev(fixed, mf, sf);
    EXPECT_NEAR(mf[0], ml[0], 1.0);
    EXPECT_NEAR(sf[0], sl[0], 1.0);
}

// Pooling: two sparse keyframes, each below row_min_matches alone, fit together (the residual is a rig property).
TEST(StereoRowRefinement, SamplesPoolAcrossKeyframes) {
    const cv::Mat left = Texture(46, 752, 480);
    const StereoMatchParams p =
        LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/euroc.yaml").stereo_match;
    const cv::Mat right = RightWithRowError(left, 9.0, -1.5, 0.0, -1.0 / 480.0);
    // Each keyframe's points spread over the WHOLE image (so the pooled fit determines the slopes too), but each
    // keyframe alone has only 2/3 of row_min_matches.
    const std::vector<cv::Point2f> all = Grid(752, 480);
    const size_t per = size_t(p.row_min_matches * 2 / 3), stride = all.size() / (2 * per);
    std::vector<cv::Point2f> a, b;
    for (size_t i = 0; i < per; ++i) {
        a.push_back(all[2 * i * stride]);
        b.push_back(all[(2 * i + 1) * stride]);
    }
    EXPECT_FALSE(EstimateRowCorrection(left, right, a, p).ok);
    RowSamples pool;
    CollectRowSamples(left, right, a, p, pool);
    EXPECT_FALSE(FitRowCorrection(pool, p).ok);
    CollectRowSamples(left, right, b, p, pool);
    const RowCorrection est = FitRowCorrection(pool, p);
    ASSERT_TRUE(est.ok) << est.why;
    EXPECT_TRUE(est.slopes_fitted);
    EXPECT_NEAR(est.a, -1.5, 0.05);
    EXPECT_NEAR(est.c * 480.0, -1.0, 0.05);
}

// 2.4 seam (§10.5): a motion beyond LK's pyramid reach is lost unseeded, and recovered when the search starts at the
// predicted position (here the exact prev->current homography, a pure image translation).
TEST(PerceptionCv, TrackerSeededByAPredictedHomographyFollowsALargeMotion) {
    const OrbEngine orb(Features());
    const cv::Mat f0 = Texture(5);
    const cv::Mat f1 = Shift(f0, 110.0, -70.0);   // scene moves by (-110, +70)
    const cv::Matx33d H(1, 0, -110.0, 0, 1, 70.0, 0, 0, 1);
    auto survivors = [&](const cv::Matx33d* pred, double& median_err) {
        KltTracker tracker(TrackerNoEpipolar(), orb);
        tracker.Process(f0, 0, 42);
        const TrackedFrame t1 = tracker.Process(f1, 1, 42, pred);
        std::vector<double> err;
        for (size_t i = 0; i < t1.ids.size(); ++i) {
            if (t1.ages[i] == 0) continue;
            const cv::Point2f d = t1.points[i] - t1.prev_points[i];
            err.push_back(std::hypot(d.x + 110.0, d.y - 70.0));
        }
        median_err = err.empty() ? 1e9 : Median(err);
        return err.size();
    };
    double e_none = 0, e_seed = 0;
    const size_t n_none = survivors(nullptr, e_none), n_seed = survivors(&H, e_seed);
    EXPECT_GT(n_seed, 300u);
    EXPECT_LT(e_seed, 0.05);
    EXPECT_GT(n_seed, 3 * n_none + 50) << "unseeded kept " << n_none;
}
