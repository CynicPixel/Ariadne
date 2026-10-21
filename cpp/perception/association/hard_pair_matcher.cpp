#include "perception/association/hard_pair_matcher.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace ariadne::perception {

namespace {

cv::Mat Pixels(const std::vector<cv::KeyPoint>& kp) {
    cv::Mat m(static_cast<int>(kp.size()), 2, CV_32F);
    for (size_t i = 0; i < kp.size(); ++i) {
        m.at<float>(static_cast<int>(i), 0) = kp[i].pt.x;
        m.at<float>(static_cast<int>(i), 1) = kp[i].pt.y;
    }
    return m;
}

}  // namespace

HardPairMatcher::HardPairMatcher(const HardPairParams& p) : p_(p) {
    if (p_.normalization != "upstream" && p_.normalization != "opencv") {
        throw std::invalid_argument("HardPairMatcher: normalization upstream|opencv");
    }
    if (p_.width <= 0 || p_.height <= 0) throw std::invalid_argument("HardPairMatcher: size");
    cv::ALIKED::Params ap;
    ap.inputSize = cv::Size(p_.width, p_.height);   // = the image size, so blobFromImage does not resize
    ap.engine = cv::dnn::ENGINE_NEW;                // ENGINE_CLASSIC cannot import either graph (SMOKE §4s)
    aliked_ = cv::ALIKED::create(p_.aliked_onnx, ap);
    lightglue_ = cv::LightGlueMatcher::create(p_.lightglue_onnx, p_.min_score);
    if (aliked_.empty() || aliked_->empty() || lightglue_.empty()) {
        throw std::runtime_error("HardPairMatcher: cannot load " + p_.aliked_onnx + " / " + p_.lightglue_onnx);
    }
}

HardPairFeatures HardPairMatcher::Extract(const cv::Mat& gray) const {
    if (gray.empty() || gray.type() != CV_8UC1) throw std::invalid_argument("HardPairMatcher: expected CV_8UC1");
    if (gray.cols != p_.width || gray.rows != p_.height) {
        throw std::invalid_argument("HardPairMatcher: image " + std::to_string(gray.cols) + "x" +
                                    std::to_string(gray.rows) + " but the export is " + std::to_string(p_.width) +
                                    "x" + std::to_string(p_.height));
    }
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);   // gray replicated: the model is RGB-trained
    HardPairFeatures f;
    f.size = gray.size();
    aliked_->detectAndCompute(bgr, cv::noArray(), f.keypoints, f.descriptors);
    return f;
}

cv::Mat HardPairMatcher::NormalizeUpstream(const std::vector<cv::KeyPoint>& kp, cv::Size size) {
    const float sx = 0.5f * static_cast<float>(size.width), sy = 0.5f * static_cast<float>(size.height);
    const float scale = 0.5f * static_cast<float>(std::max(size.width, size.height));
    cv::Mat m = Pixels(kp);
    for (int i = 0; i < m.rows; ++i) {
        m.at<float>(i, 0) = (m.at<float>(i, 0) - sx) / scale;
        m.at<float>(i, 1) = (m.at<float>(i, 1) - sy) / scale;
    }
    return m;
}

std::vector<cv::DMatch> HardPairMatcher::Match(const HardPairFeatures& a, const HardPairFeatures& b) const {
    std::vector<cv::DMatch> out;
    if (a.keypoints.empty() || b.keypoints.empty()) return out;
    if (p_.normalization == "upstream") {
        // Empty sizes: LightGlueMatcher passes the keypoints through unchanged.
        lightglue_->setPairInfo(NormalizeUpstream(a.keypoints, a.size), NormalizeUpstream(b.keypoints, b.size),
                                cv::Size(), cv::Size());
    } else {
        lightglue_->setPairInfo(Pixels(a.keypoints), Pixels(b.keypoints), a.size, b.size);
    }
    lightglue_->match(a.descriptors, b.descriptors, out);   // setPairInfo is consumed by this call
    return out;
}

std::string HardPairMatcher::Name() const {
    return std::filesystem::path(p_.aliked_onnx).stem().string() + "+" +
           std::filesystem::path(p_.lightglue_onnx).stem().string() + "/cv_dnn_new/norm_" + p_.normalization;
}

}  // namespace ariadne::perception
