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

HardPairParams Checked(const HardPairParams& p) {
    if (p.normalization != "upstream" && p.normalization != "opencv") {
        throw std::invalid_argument("HardPairMatcher: normalization upstream|opencv");
    }
    if (p.width <= 0 || p.height <= 0) throw std::invalid_argument("HardPairMatcher: size");
    return p;
}

}  // namespace

HardPairMatcher::HardPairMatcher(const HardPairParams& p)
    : p_(Checked(p)), nets_([q = p_] {
          cv::ALIKED::Params ap;
          ap.inputSize = cv::Size(q.width, q.height);   // = the image size, so blobFromImage does not resize
          ap.engine = cv::dnn::ENGINE_NEW;              // ENGINE_CLASSIC cannot import either graph (SMOKE §4s)
          auto n = std::make_unique<Nets>();
          n->aliked = cv::ALIKED::create(q.aliked_onnx, ap);
          n->lightglue = cv::LightGlueMatcher::create(q.lightglue_onnx, q.min_score);
          if (n->aliked.empty() || n->aliked->empty() || n->lightglue.empty()) {
              throw std::runtime_error("HardPairMatcher: cannot load " + q.aliked_onnx + " / " + q.lightglue_onnx);
          }
          return n;
      }) {
    nets_.Local();   // load on the constructing thread: a bad file fails here
}

void HardPairMatcher::CheckSize(const cv::Mat& im) const {
    if (im.cols != p_.width || im.rows != p_.height) {
        throw std::invalid_argument("HardPairMatcher: image " + std::to_string(im.cols) + "x" +
                                    std::to_string(im.rows) + " but the export is " + std::to_string(p_.width) +
                                    "x" + std::to_string(p_.height));
    }
}

HardPairFeatures HardPairMatcher::ExtractRgb(const cv::Mat& rgb) const {
    if (rgb.empty() || rgb.type() != CV_8UC3) throw std::invalid_argument("HardPairMatcher: expected CV_8UC3 (RGB)");
    CheckSize(rgb);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    HardPairFeatures f;
    f.size = rgb.size();
    nets_.Local().aliked->detectAndCompute(bgr, cv::noArray(), f.keypoints, f.descriptors);
    return f;
}

HardPairFeatures HardPairMatcher::Extract(const cv::Mat& gray) const {
    if (gray.empty() || gray.type() != CV_8UC1) throw std::invalid_argument("HardPairMatcher: expected CV_8UC1");
    CheckSize(gray);
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);   // gray replicated: the model is RGB-trained
    HardPairFeatures f;
    f.size = gray.size();
    nets_.Local().aliked->detectAndCompute(bgr, cv::noArray(), f.keypoints, f.descriptors);
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
    cv::LightGlueMatcher& lightglue = *nets_.Local().lightglue;
    if (p_.normalization == "upstream") {
        // Empty sizes: LightGlueMatcher passes the keypoints through unchanged.
        lightglue.setPairInfo(NormalizeUpstream(a.keypoints, a.size), NormalizeUpstream(b.keypoints, b.size),
                                cv::Size(), cv::Size());
    } else {
        lightglue.setPairInfo(Pixels(a.keypoints), Pixels(b.keypoints), a.size, b.size);
    }
    lightglue.match(a.descriptors, b.descriptors, out);   // setPairInfo is consumed by this call
    return out;
}

std::vector<dnn::ModelRecord> HardPairMatcher::Records() const {
    return {{std::filesystem::path(p_.aliked_onnx).stem().string(), p_.aliked_onnx, "cv_dnn_new"},
            {std::filesystem::path(p_.lightglue_onnx).stem().string(), p_.lightglue_onnx, "cv_dnn_new"}};
}

std::string HardPairMatcher::Name() const {
    return std::filesystem::path(p_.aliked_onnx).stem().string() + "+" +
           std::filesystem::path(p_.lightglue_onnx).stem().string() + "/cv_dnn_new/norm_" + p_.normalization;
}

}  // namespace ariadne::perception
