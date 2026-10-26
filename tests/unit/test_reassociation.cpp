// 2.1 re-association (perception.md §4.3): the appearance matcher, the geometric verifier, the tracker rename,
// the ORB Describe() contract, and the config section.
#include "core/random/philox.hpp"
#include "perception/association/landmark_reassociation.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/perception_config.hpp"
#include "perception/tracking/klt_tracker.hpp"

#include <gtest/gtest.h>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include <climits>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>

using namespace ariadne;
using namespace ariadne::perception;

namespace {

constexpr double kFx = 435.0, kFy = 435.0, kCx = 376.0, kCy = 240.0;   // EuRoC-like rectified left camera
constexpr int kW = 752, kH = 480;

const PerceptionConfig& Euroc() {
    static const PerceptionConfig c =
        LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/euroc.yaml");
    return c;
}

ReassociationParams Params() {
    ReassociationParams p;
    p.enabled = true;
    return p;
}

/// 32-byte descriptors, independent uniform bits: two random ones differ in ~128 of 256 bits.
cv::Mat RandomBinary(int rows, uint64_t seed) {
    cv::Mat d(rows, 32, CV_8U);
    for (int r = 0; r < rows; ++r) {
        core::PhiloxStream s(seed, 7, static_cast<uint64_t>(r));
        for (int c = 0; c < 32; ++c) d.at<uchar>(r, c) = static_cast<uchar>(s.NextBits32() & 0xFF);
    }
    return d;
}

/// Flip `bits` distinct bits of row r (appearance change between keyframe and now).
void FlipBits(cv::Mat& d, int r, int bits, uint64_t seed) {
    core::PhiloxStream s(seed, 9, static_cast<uint64_t>(r));
    std::set<int> done;
    while (static_cast<int>(done.size()) < bits) {
        const int b = static_cast<int>(s.NextBits32() % 256);
        if (!done.insert(b).second) continue;
        d.at<uchar>(r, b / 8) ^= static_cast<uchar>(1u << (b % 8));
    }
}

cv::Mat Texture(uint64_t seed) {
    cv::Mat noise(kH, kW, CV_8U);
    for (int y = 0; y < kH; ++y) {
        core::PhiloxStream s(seed, 0, static_cast<uint64_t>(y));
        for (int x = 0; x < kW; ++x) noise.at<uchar>(y, x) = static_cast<uchar>(s.NextBits32() & 0xFF);
    }
    cv::Mat b;
    cv::GaussianBlur(noise, b, cv::Size(0, 0), 1.5);
    cv::normalize(b, b, 0, 255, cv::NORM_MINMAX);
    return b;
}

/// A world of N landmarks 2-6 m in front of a camera at T_C_W = (R(rvec), t); their pixels in that camera.
struct Scene {
    std::vector<cv::Point3d> X_w;
    std::vector<cv::Point2f> px;
};
Scene MakeScene(int n, uint64_t seed, const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    Scene s;
    core::PhiloxStream rng(seed, 3, 0);
    const auto u = [&]() { return (rng.NextBits32() & 0xFFFFFF) / double(0xFFFFFF); };
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    while (static_cast<int>(s.X_w.size()) < n) {
        // Sample in the camera, then express in the world: X_w = R^T (X_c - t).
        const double z = 2.0 + 4.0 * u();
        const cv::Vec3d Xc((u() * kW - kCx) * z / kFx, (u() * kH - kCy) * z / kFy, z);
        const cv::Vec3d Xw = R.t() * (Xc - tvec);
        s.X_w.emplace_back(Xw[0], Xw[1], Xw[2]);
        s.px.emplace_back(static_cast<float>(kFx * Xc[0] / z + kCx), static_cast<float>(kFy * Xc[1] / z + kCy));
    }
    return s;
}

}  // namespace

// --- Appearance matching ------------------------------------------------------

TEST(Reassociation, MutualRatioPairsIdenticalDescriptors) {
    const cv::Mat train = RandomBinary(200, 1);
    cv::Mat query(50, 32, CV_8U);
    for (int r = 0; r < 50; ++r) train.row(3 * r).copyTo(query.row(r));   // query r <- candidate 3r
    for (int r = 0; r < 50; ++r) FlipBits(query, r, 10, 2);               // 10 of 256 bits: well under TH_LOW 50
    const auto m = MutualRatioMatches(query, train, Params());
    ASSERT_EQ(m.size(), 50u);
    for (const auto& x : m) EXPECT_EQ(x.trainIdx, 3 * x.queryIdx);
}

// Two candidates with the same appearance (repeated texture): the ratio test must refuse rather than guess.
TEST(Reassociation, RatioTestRefusesAmbiguousCandidates) {
    cv::Mat train = RandomBinary(20, 3);
    train.row(5).copyTo(train.row(6));
    cv::Mat query;
    train.row(5).copyTo(query);
    EXPECT_TRUE(MutualRatioMatches(query, train, Params()).empty());
}

// Beyond the absolute cap (ORB-SLAM TH_LOW = 50 bits) a nearest neighbour is not a match, however unique.
TEST(Reassociation, HammingCapRejectsDistantNeighbours) {
    const cv::Mat train = RandomBinary(1, 4);
    cv::Mat query = train.clone();
    FlipBits(query, 0, 60, 5);
    EXPECT_TRUE(MutualRatioMatches(query, train, Params()).empty());
    ReassociationParams loose = Params();
    loose.max_hamming = 70;
    EXPECT_EQ(MutualRatioMatches(query, train, loose).size(), 1u);
}

// Mutual: two queries whose nearest is the same candidate -> only the candidate's own nearest survives.
TEST(Reassociation, MutualCheckKeepsOneQueryPerCandidate) {
    const cv::Mat train = RandomBinary(30, 6);
    cv::Mat query(2, 32, CV_8U);
    train.row(9).copyTo(query.row(0));
    train.row(9).copyTo(query.row(1));
    FlipBits(query, 0, 4, 7);
    FlipBits(query, 1, 12, 7);
    const auto m = MutualRatioMatches(query, train, Params());
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].queryIdx, 0);
    EXPECT_EQ(m[0].trainIdx, 9);
}

// --- Geometric verification ---------------------------------------------------

// The relocalisation case: 8 healthy correspondences survive (below PnP's 15), 300 orphan tracks, 40 of which are
// old landmarks. Appearance finds them plus 10 descriptor-identical decoys at WRONG pixels (aliasing); geometry
// must keep the 40 and drop the 10.
TEST(Reassociation, PnpKeepsTrueMatchesAndDropsAliases) {
    const cv::Vec3d rvec(0.05, -0.2, 0.02), tvec(0.3, -0.1, 0.4);
    const Scene s = MakeScene(400, 11, rvec, tvec);
    AnchorCorrespondences anchors;
    for (int i = 0; i < 8; ++i) {
        anchors.X_w.push_back(s.X_w[static_cast<size_t>(i)]);
        anchors.points.push_back(s.px[static_cast<size_t>(i)]);
    }
    // Candidates 0..99 = landmarks 8..107.
    CandidateLandmarks cand;
    cand.descriptors = RandomBinary(100, 12);
    for (int c = 0; c < 100; ++c) {
        cand.landmark_ids.push_back(1000 + static_cast<uint64_t>(c));
        cand.X_w.push_back(s.X_w[static_cast<size_t>(8 + c)]);
    }
    OrphanTracks orph;
    orph.descriptors = RandomBinary(300, 13);   // unrelated appearance by default
    for (int k = 0; k < 300; ++k) {
        orph.track_ids.push_back(5000 + static_cast<uint64_t>(k));
        orph.points.push_back(s.px[static_cast<size_t>(100 + k % 300)]);
    }
    std::map<uint64_t, uint64_t> truth;
    for (int k = 0; k < 40; ++k) {   // orphan k sits on candidate k's landmark and looks like it
        orph.points[static_cast<size_t>(k)] = s.px[static_cast<size_t>(8 + k)];
        cand.descriptors.row(k).copyTo(orph.descriptors.row(k));
        FlipBits(orph.descriptors, k, 8, 14);
        truth[orph.track_ids[static_cast<size_t>(k)]] = cand.landmark_ids[static_cast<size_t>(k)];
    }
    for (int k = 40; k < 50; ++k) {   // aliases: look like candidates 40..49, sit at some other landmark's pixel
        orph.points[static_cast<size_t>(k)] = s.px[static_cast<size_t>(300 + k)];
        cand.descriptors.row(k).copyTo(orph.descriptors.row(k));
    }
    const ReassociationResult r =
        ReassociateLost(orph, cand, anchors, kFx, kFy, kCx, kCy, Params(), /*seed=*/1, /*frame_id=*/77);
    EXPECT_EQ(r.matched, 50);
    ASSERT_TRUE(r.accepted) << r.why;
    EXPECT_EQ(r.from.size(), 40u);
    for (size_t k = 0; k < r.from.size(); ++k) EXPECT_EQ(truth.at(r.from[k]), r.to[k]);
    EXPECT_GE(r.inliers, 48);   // 40 matches + the 8 anchors

    // Determinism: the same inputs and seed give the same renames.
    const ReassociationResult again = ReassociateLost(orph, cand, anchors, kFx, kFy, kCx, kCy, Params(), 1, 77);
    EXPECT_EQ(again.from, r.from);
    EXPECT_EQ(again.to, r.to);
}

// Appearance matches with no consistent geometry (all aliases) must not be applied.
TEST(Reassociation, RefusesMatchesWithoutGeometricSupport) {
    const Scene s = MakeScene(200, 21, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0));
    CandidateLandmarks cand;
    cand.descriptors = RandomBinary(30, 22);
    OrphanTracks orph;
    orph.descriptors = cand.descriptors.clone();
    core::PhiloxStream rng(23, 0, 0);
    for (int c = 0; c < 30; ++c) {
        cand.landmark_ids.push_back(static_cast<uint64_t>(c));
        cand.X_w.push_back(s.X_w[static_cast<size_t>(c)]);
        orph.track_ids.push_back(100 + static_cast<uint64_t>(c));
        // Each lookalike sits at a random pixel: no single pose explains them.
        orph.points.emplace_back(static_cast<float>(rng.NextBits32() % kW), static_cast<float>(rng.NextBits32() % kH));
    }
    const ReassociationResult r = ReassociateLost(orph, cand, {}, kFx, kFy, kCx, kCy, Params(), 1, 3);
    EXPECT_EQ(r.matched, 30);
    EXPECT_FALSE(r.accepted);
    EXPECT_TRUE(r.from.empty());
}

// --- Tracker rename -----------------------------------------------------------

TEST(Reassociation, TrackerRelabelCarriesTheNewIdForward) {
    TrackerParams tp = Euroc().tracker;
    tp.f_min_points = INT_MAX;   // a pure image shift is planar: F would be degenerate
    const OrbEngine orb(Euroc().features);
    KltTracker tracker(tp, orb);
    const cv::Mat f0 = Texture(31);
    const TrackedFrame t0 = tracker.Process(f0, 0, 1);
    ASSERT_GT(t0.ids.size(), 10u);
    const uint64_t a = t0.ids[0], b = t0.ids[1];
    tracker.Relabel({a, b}, {900000, 900001});
    cv::Mat f1;
    const cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, 1.5, 0, 1, -0.5);
    cv::warpAffine(f0, f1, m, f0.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REFLECT_101);
    const TrackedFrame t1 = tracker.Process(f1, 1, 1);
    std::map<uint64_t, cv::Point2f> at;
    for (size_t i = 0; i < t1.ids.size(); ++i) at[t1.ids[i]] = t1.points[i];
    EXPECT_FALSE(at.count(a));
    EXPECT_FALSE(at.count(b));
    ASSERT_TRUE(at.count(900000));
    // Same physical point: the renamed track moved with the image (scene moved by (-1.5, +0.5)).
    EXPECT_NEAR(at[900000].x, t0.points[0].x - 1.5f, 0.1f);
    EXPECT_NEAR(at[900000].y, t0.points[0].y + 0.5f, 0.1f);
}

TEST(Reassociation, TrackerRelabelRejectsCollisions) {
    const OrbEngine orb(Euroc().features);
    KltTracker tracker(Euroc().tracker, orb);
    const TrackedFrame t0 = tracker.Process(Texture(32), 0, 1);
    ASSERT_GT(t0.ids.size(), 3u);
    EXPECT_THROW(tracker.Relabel({123456789}, {900000}), std::invalid_argument);        // not live
    EXPECT_THROW(tracker.Relabel({t0.ids[0]}, {t0.ids[1]}), std::invalid_argument);     // target live
    EXPECT_THROW(tracker.Relabel({t0.ids[0], t0.ids[1]}, {7, 7}), std::invalid_argument);   // repeated
}

// --- 2.1 tracker cadence (§4.2) -----------------------------------------------------

namespace {
/// Frame 0: texture on the left half only (right half flat); frame 1: the full texture. So frame 1 has new texture to
/// start tracks on, and only the cadence decides whether they start.
std::pair<cv::Mat, cv::Mat> HalfThenFull() {
    const cv::Mat full = Texture(41);
    cv::Mat half = full.clone();
    half(cv::Rect(full.cols / 2, 0, full.cols - full.cols / 2, full.rows)).setTo(128);
    return {half, full};
}
TrackerParams CadenceParams(const std::string& mode, int min_tracks) {
    TrackerParams tp = Euroc().tracker;
    tp.f_min_points = INT_MAX;   // no epipolar test on a static pair
    tp.target_tracks = 1000;     // more than either frame supplies, so a shortfall always exists
    tp.replenish = mode;
    tp.replenish_min_tracks = min_tracks;
    return tp;
}
}  // namespace

TEST(TrackerCadence, KeyframeModeStartsTracksOnlyWhenAsked) {
    const OrbEngine orb(Euroc().features);
    KltTracker tracker(CadenceParams("keyframe", 0), orb);
    const auto [f0, f1] = HalfThenFull();
    const TrackedFrame t0 = tracker.Process(f0, 0, 1);
    ASSERT_GT(t0.stats.replenished, 10);   // the first frame always seeds
    const TrackedFrame t1 = tracker.Process(f1, 1, 1);
    EXPECT_EQ(t1.stats.replenished, 0);    // not a keyframe, tracks not short: no detection
    const int added = tracker.Replenish(f1);
    EXPECT_GT(added, 10);                  // the right half's texture
    const TrackedFrame t2 = tracker.Process(f1, 2, 1);
    std::set<uint64_t> live(t2.ids.begin(), t2.ids.end());
    int carried = 0;
    for (uint64_t id = t0.ids.back() + 1; id <= t0.ids.back() + static_cast<uint64_t>(added); ++id) carried += live.count(id);
    EXPECT_EQ(carried, added);             // Replenish()'s tracks are tracked by the next Process()
    int right = 0;
    for (size_t i = 0; i < t2.ids.size(); ++i) right += t2.ids[i] > t0.ids.back() && t2.points[i].x > f1.cols / 2;
    EXPECT_GT(right, added * 8 / 10);      // and they are mostly where the new texture is
}

TEST(TrackerCadence, OffCadenceRecoveryReplenishesBelowTheThreshold) {
    const OrbEngine orb(Euroc().features);
    KltTracker tracker(CadenceParams("keyframe", 1000), orb);   // threshold above any supply: always short
    const auto [f0, f1] = HalfThenFull();
    tracker.Process(f0, 0, 1);
    EXPECT_GT(tracker.Process(f1, 1, 1).stats.replenished, 10);
}

TEST(TrackerCadence, EveryFrameIsThePhaseOneBehaviourAndReplenishIsANoOp) {
    const OrbEngine orb(Euroc().features);
    KltTracker every(CadenceParams("every_frame", 0), orb);
    const auto [f0, f1] = HalfThenFull();
    every.Process(f0, 0, 1);
    EXPECT_GT(every.Process(f1, 1, 1).stats.replenished, 10);
    EXPECT_EQ(every.Replenish(f1), 0);
    KltTracker kf(CadenceParams("keyframe", 0), orb);
    EXPECT_THROW(kf.Replenish(f0), std::logic_error);           // before any Process()
    kf.Process(f0, 0, 1);
    EXPECT_THROW(kf.Replenish(f0(cv::Rect(0, 0, 320, 240)).clone()), std::logic_error);   // not that image
}

// 5.2 (D146): Replenish can detect on another image of the same pixels (the albedo proxy) inside its mask, while LK
// keeps the tracked image.
TEST(TrackerCadence, ReplenishDetectsOnTheDetectionImageInsideItsMask) {
    const OrbEngine orb(Euroc().features);
    const auto [f0, f1] = HalfThenFull();
    KltTracker blank(CadenceParams("keyframe", 0), orb);
    blank.Process(f0, 0, 1);
    blank.Process(f1, 1, 1);
    // A flat detection image: nothing to detect, although the tracked image has new texture on the right.
    EXPECT_EQ(blank.Replenish(f1, cv::Mat(f1.size(), CV_8U, cv::Scalar(128)), cv::Mat()), 0);

    KltTracker tracker(CadenceParams("keyframe", 0), orb);
    const TrackedFrame t0 = tracker.Process(f0, 0, 1);
    tracker.Process(f1, 1, 1);
    cv::Mat top(f1.size(), CV_8U, cv::Scalar(0));
    top.rowRange(0, f1.rows / 2).setTo(255);
    const int added = tracker.Replenish(f1, f1, top);
    EXPECT_GT(added, 5);
    const TrackedFrame t2 = tracker.Process(f1, 2, 1);
    int outside = 0;
    for (size_t i = 0; i < t2.ids.size(); ++i) outside += t2.ids[i] > t0.ids.back() && t2.points[i].y >= f1.rows / 2 + 1;
    EXPECT_EQ(outside, 0);   // every new track was started inside the mask (and LK held it on the static pair)

    EXPECT_THROW(tracker.Replenish(f1, f1(cv::Rect(0, 0, 320, 240)).clone(), cv::Mat()), std::invalid_argument);
    EXPECT_THROW(tracker.Replenish(f1, f1, top(cv::Rect(0, 0, 320, 240)).clone()), std::invalid_argument);
}

// 5.3 (D147): known-gain LK. The next frame is the scene shifted by a sub-pixel motion AND sqrt(2) brighter (half a
// stop: the controller's largest step). With the previous pyramid rescaled by the known ratio, LK tracks the motion as
// if nothing had changed; ratio 1 is a no-op.
TEST(TrackerCadence, RescalePreviousAppliesTheKnownGain) {
    const OrbEngine orb(Euroc().features);
    cv::Mat f0 = Texture(41) * 0.5;   // headroom: x sqrt(2) stays below 255
    cv::Mat f1;
    const double dx = 2.3, dy = 1.6, r = std::sqrt(2.0);
    cv::warpAffine(f0, f1, cv::Matx23d(1, 0, dx, 0, 1, dy), f0.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    f1 = f1 * r;
    const auto run = [&](bool rescale, double* median_err) {
        KltTracker tracker(CadenceParams("every_frame", 0), orb);
        const TrackedFrame t0 = tracker.Process(f0, 0, 1);
        tracker.RescalePrevious(1.0);   // no-op
        if (rescale) tracker.RescalePrevious(r);
        const TrackedFrame t1 = tracker.Process(f1, 1, 1);
        std::map<uint64_t, cv::Point2f> p0;
        for (size_t i = 0; i < t0.ids.size(); ++i) p0[t0.ids[i]] = t0.points[i];
        std::vector<double> err;
        for (size_t i = 0; i < t1.ids.size(); ++i) {
            const auto it = p0.find(t1.ids[i]);
            if (it == p0.end()) continue;
            err.push_back(std::hypot(t1.points[i].x - it->second.x - dx, t1.points[i].y - it->second.y - dy));
        }
        std::sort(err.begin(), err.end());
        *median_err = err.empty() ? 1e9 : err[err.size() / 2];
        return err.size();
    };
    double e_known = 0, e_plain = 0;
    const size_t n_known = run(true, &e_known), n_plain = run(false, &e_plain);
    std::printf("known-gain LK: %zu tracks, median error %.4f px; plain LK: %zu tracks, %.4f px\n", n_known, e_known,
                n_plain, e_plain);
    EXPECT_GT(n_known, 50u);
    EXPECT_LT(e_known, 0.05);
    EXPECT_GE(n_known, n_plain);
    EXPECT_LE(e_known, e_plain);
    KltTracker fresh(CadenceParams("every_frame", 0), orb);
    EXPECT_NO_THROW(fresh.RescalePrevious(2.0));   // no previous frame yet: nothing to rescale
    EXPECT_THROW(fresh.RescalePrevious(0.0), std::invalid_argument);
}

TEST(TrackerCadence, QualityGateKeepsNewTracksOffWeakGradients) {
    // Left half: full-contrast texture. Right half: the same texture at 12% contrast, still detectable by ORB's low
    // FAST threshold, but with a structure-tensor eigenvalue ~0.12^2 = 1.4% of the left's: below a 0.1 gate.
    const cv::Mat full = Texture(43);
    cv::Mat img = full.clone();
    cv::Mat right = img(cv::Rect(kW / 2, 0, kW - kW / 2, kH));
    full(cv::Rect(kW / 2, 0, kW - kW / 2, kH)).convertTo(right, CV_8U, 0.12, 128 * (1 - 0.12));
    const OrbEngine orb(Euroc().features);
    const auto right_tracks = [&](double q) {
        TrackerParams tp = CadenceParams("every_frame", 0);
        tp.replenish_min_eig_quality = q;
        KltTracker t(tp, orb);
        const TrackedFrame f = t.Process(img, 0, 1);
        int n = 0;
        for (const auto& p : f.points) n += p.x > kW / 2 + 20;
        return std::pair<int, int>(n, static_cast<int>(f.points.size()));
    };
    const auto [off_right, off_all] = right_tracks(0.0);
    const auto [on_right, on_all] = right_tracks(0.1);
    ASSERT_GT(off_right, 20) << "the weak half must be detectable, or the test proves nothing";
    EXPECT_EQ(on_right, 0);
    EXPECT_GT(on_all, 50);   // the strong half still seeds
}

// --- ORB Describe() -------------------------------------------------------------

// The same physical points in a shifted image describe identically, and match back one to one: what lost mode
// relies on. Points on a distinct integer grid (detections from different octaves can coincide, and the ratio test
// then rightly refuses them as ambiguous). Border points are reported as not described.
TEST(Reassociation, OrbDescribeIsStableUnderShiftAndFlagsBorders) {
    const OrbEngine orb(Euroc().features);
    const cv::Mat f0 = Texture(41);
    std::vector<cv::Point2f> p0, p1;
    for (int y = 40; y < kH - 40; y += 25)
        for (int x = 40; x < kW - 40; x += 25) {
            p0.emplace_back(static_cast<float>(x), static_cast<float>(y));
            p1.emplace_back(static_cast<float>(x - 4), static_cast<float>(y + 3));   // scene moved by (-4, +3)
        }
    cv::Mat f1;
    const cv::Mat m = (cv::Mat_<double>(2, 3) << 1, 0, 4.0, 0, 1, -3.0);
    cv::warpAffine(f0, f1, m, f0.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REFLECT_101);
    std::vector<uchar> ok0, ok1;
    const cv::Mat d0 = orb.Describe(f0, p0, ok0), d1 = orb.Describe(f1, p1, ok1);
    ASSERT_EQ(d0.rows, static_cast<int>(p0.size()));
    cv::Mat q, t;
    int identical = 0;
    for (size_t i = 0; i < p0.size(); ++i) {
        if (!ok0[i] || !ok1[i]) continue;
        q.push_back(d1.row(static_cast<int>(i)));
        t.push_back(d0.row(static_cast<int>(i)));
        identical += cv::norm(d0.row(static_cast<int>(i)), d1.row(static_cast<int>(i)), cv::NORM_HAMMING) == 0;
    }
    ASSERT_GT(q.rows, 300);
    EXPECT_EQ(identical, q.rows);   // an integer shift copies pixels: the descriptor is a function of the patch
    const auto mm = MutualRatioMatches(q, t, Params());
    int right = 0;
    for (const auto& x : mm) right += x.queryIdx == x.trainIdx;
    EXPECT_GE(right, q.rows * 95 / 100);
    EXPECT_EQ(right, static_cast<int>(mm.size()));   // no wrong pairs

    std::vector<uchar> okb;
    orb.Describe(f0, {cv::Point2f(2, 2), cv::Point2f(kW / 2.f, kH / 2.f)}, okb);
    EXPECT_EQ(okb[0], 0);
    EXPECT_EQ(okb[1], 1);
}

// --- Config -------------------------------------------------------------------

TEST(Reassociation, AbsentSectionMeansOff) {
    EXPECT_FALSE(Euroc().reassociation.enabled);
}

TEST(Reassociation, ShippedV2ConfigEnablesIt) {
    const PerceptionConfig c =
        LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/euroc_v2.yaml");
    EXPECT_TRUE(c.reassociation.enabled);
    EXPECT_DOUBLE_EQ(c.reassociation.ratio, 0.75);
    EXPECT_EQ(c.reassociation.max_hamming, 50);
}
