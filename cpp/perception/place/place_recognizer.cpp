#include "perception/place/place_recognizer.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace ariadne::perception {

namespace {
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};   // ImageNet, upstream's test transform
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
}  // namespace

PlaceRecognizer::PlaceRecognizer(const PlaceRecognizerParams& p) : p_(p) {
    if (p_.engine != "classic" && p_.engine != "new") throw std::invalid_argument("PlaceRecognizer: engine classic|new");
    if (p_.width <= 0 || p_.height <= 0) throw std::invalid_argument("PlaceRecognizer: size");
    net_ = cv::dnn::readNetFromONNX(p_.onnx, p_.engine == "new" ? cv::dnn::ENGINE_NEW : cv::dnn::ENGINE_CLASSIC);
    if (net_.empty()) throw std::runtime_error("PlaceRecognizer: cannot load " + p_.onnx);
}

cv::Mat PlaceRecognizer::Preprocess(const cv::Mat& gray, int width, int height) {
    if (gray.empty() || gray.type() != CV_8UC1) throw std::invalid_argument("PlaceRecognizer: expected CV_8UC1");
    cv::Mat f;
    gray.convertTo(f, CV_32F, 1.0 / 255.0);   // ToTensor
    if (f.cols != width || f.rows != height) {
        cv::Mat r;
        cv::resize(f, r, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        f = r;
    }
    const int sz[4] = {1, 3, height, width};
    cv::Mat blob(4, sz, CV_32F);
    for (int c = 0; c < 3; ++c) {
        cv::Mat plane(height, width, CV_32F, blob.ptr<float>(0, c));
        f.convertTo(plane, CV_32F, 1.0 / kStd[c], -kMean[c] / kStd[c]);   // (x - mean) / std, gray in all three
    }
    return blob;
}

std::vector<float> PlaceRecognizer::Describe(const cv::Mat& gray) {
    net_.setInput(Preprocess(gray, p_.width, p_.height));
    const cv::Mat out = net_.forward().reshape(1, 1);
    if (out.total() != 256) throw std::runtime_error("PlaceRecognizer: expected a 256-D output");
    std::vector<float> d(out.begin<float>(), out.end<float>());
    // The graph ends in L2Norm already; renormalising guards the fp error only (|d| = 1.000000 in SMOKE §4r).
    double n = 0;
    for (float v : d) n += double(v) * v;
    n = std::sqrt(n);
    if (n > 0) for (float& v : d) v = float(v / n);
    return d;
}

std::string PlaceRecognizer::Name() const {
    return "eigenplaces_r50_256@" + std::to_string(p_.width) + "x" + std::to_string(p_.height) + "/cv_dnn_" + p_.engine;
}

}  // namespace ariadne::perception
