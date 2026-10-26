#include "perception/association/landmark_reassociation.hpp"

#include "core/random/philox.hpp"

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>

#include <stdexcept>

namespace ariadne::perception {

namespace {
// Philox sites are allocated downward from kMaxSite by each USAC user (tracker -1, PnP -2, bridge -3..-6). This one
// sits well below them so a site added next does not collide.
constexpr uint64_t kSiteReassociationUsac = core::PhiloxStream::kMaxSite - 16;
}  // namespace

std::vector<cv::DMatch> MutualRatioMatches(const cv::Mat& query, const cv::Mat& train,
                                           const ReassociationParams& params) {
    std::vector<cv::DMatch> out;
    if (query.empty() || train.empty()) return out;
    if (query.type() != train.type() || query.cols != train.cols) {
        throw std::invalid_argument("MutualRatioMatches: descriptor type/width mismatch");
    }
    const bool binary = query.depth() == CV_8U;
    const int norm = binary ? cv::NORM_HAMMING : cv::NORM_L2;
    const double cap = binary ? static_cast<double>(params.max_hamming) : params.max_l2;
    // Brute force: exact, and deterministic (no randomised trees). The candidate set is one window's landmarks.
    cv::BFMatcher bf(norm, /*crossCheck=*/false);
    std::vector<std::vector<cv::DMatch>> fwd;
    bf.knnMatch(query, train, fwd, 2);
    std::vector<cv::DMatch> back;
    bf.match(train, query, back);   // each candidate's nearest orphan
    for (const auto& m : fwd) {
        if (m.empty()) continue;
        const cv::DMatch& best = m[0];
        if (best.distance > cap) continue;
        // Lowe's ratio: the best must beat the second best clearly, or the descriptor is ambiguous (repeated
        // texture). With one candidate there is no second: accepted on the cap alone.
        if (m.size() > 1 && !(best.distance < params.ratio * m[1].distance)) continue;
        if (back[static_cast<size_t>(best.trainIdx)].trainIdx != best.queryIdx) continue;   // mutual
        out.push_back(best);
    }
    return out;
}

ReassociationResult ReassociateLost(const OrphanTracks& orphans, const CandidateLandmarks& candidates,
                                    const AnchorCorrespondences& anchors, double fx, double fy, double cx,
                                    double cy, const ReassociationParams& params, uint64_t seed,
                                    uint64_t frame_id) {
    if (orphans.track_ids.size() != orphans.points.size() ||
        static_cast<int>(orphans.track_ids.size()) != orphans.descriptors.rows) {
        throw std::invalid_argument("ReassociateLost: orphan arrays");
    }
    if (candidates.landmark_ids.size() != candidates.X_w.size() ||
        static_cast<int>(candidates.landmark_ids.size()) != candidates.descriptors.rows) {
        throw std::invalid_argument("ReassociateLost: candidate arrays");
    }
    if (anchors.X_w.size() != anchors.points.size()) throw std::invalid_argument("ReassociateLost: anchor arrays");

    ReassociationResult res;
    res.orphans = static_cast<int>(orphans.track_ids.size());
    res.candidates = static_cast<int>(candidates.landmark_ids.size());
    res.anchors = static_cast<int>(anchors.X_w.size());
    const std::vector<cv::DMatch> matches = MutualRatioMatches(orphans.descriptors, candidates.descriptors, params);
    res.matched = static_cast<int>(matches.size());
    if (matches.empty()) {
        res.why = "no appearance match";
        return res;
    }
    // Anchors first, then matches: an inlier index >= anchors marks a match.
    std::vector<cv::Point3d> obj(anchors.X_w);
    std::vector<cv::Point2d> img;
    img.reserve(anchors.points.size() + matches.size());
    for (const cv::Point2f& p : anchors.points) img.emplace_back(p.x, p.y);
    for (const cv::DMatch& m : matches) {
        obj.push_back(candidates.X_w[static_cast<size_t>(m.trainIdx)]);
        const cv::Point2f& p = orphans.points[static_cast<size_t>(m.queryIdx)];
        img.emplace_back(p.x, p.y);
    }
    if (static_cast<int>(obj.size()) < params.min_inliers) {
        res.why = "fewer anchors + matches than min_inliers";
        return res;
    }
    const cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::UsacParams usac;
    usac.threshold = params.ransac_threshold_px;
    usac.confidence = 0.999;
    usac.maxIterations = 1000;
    usac.isParallel = false;
    usac.randomGeneratorState = static_cast<int>(
        core::PhiloxStream(seed, frame_id, kSiteReassociationUsac).NextBits32() & 0x7FFFFFFFu);
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    res.ran = true;
    bool solved = false;
    try {
        solved = cv::solvePnPRansac(obj, img, K, cv::noArray(), rvec, tvec, inliers, usac);
    } catch (const cv::Exception&) {
        // OpenCV 5.0.0 USAC can throw on an empty model (SMOKE §4q); for a verifier, no model = no evidence.
        solved = false;
    }
    if (!solved) {
        res.why = "PnP found no model";
        return res;
    }
    res.inliers = static_cast<int>(inliers.size());
    std::vector<uint64_t> from, to;
    for (int k : inliers) {
        if (k < res.anchors) continue;
        const cv::DMatch& m = matches[static_cast<size_t>(k - res.anchors)];
        from.push_back(orphans.track_ids[static_cast<size_t>(m.queryIdx)]);
        to.push_back(candidates.landmark_ids[static_cast<size_t>(m.trainIdx)]);
    }
    res.match_inliers = static_cast<int>(from.size());
    if (res.inliers < params.min_inliers) {
        res.why = "too few PnP inliers";
        return res;
    }
    if (from.empty()) {
        res.why = "no match is an inlier";
        return res;
    }
    res.from = std::move(from);
    res.to = std::move(to);
    res.accepted = true;
    return res;
}

}  // namespace ariadne::perception
