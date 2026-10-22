#include "perception/place/place_recognizer.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace ariadne::perception {

namespace {
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};   // ImageNet, upstream's test transform
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
}  // namespace

namespace {
PlaceRecognizerParams Checked(const PlaceRecognizerParams& p) {
    if (p.engine != "classic" && p.engine != "new") throw std::invalid_argument("PlaceRecognizer: engine classic|new");
    if (p.width <= 0 || p.height <= 0) throw std::invalid_argument("PlaceRecognizer: size");
    if (p.enhance != "none" && p.enhance != "clahe") throw std::invalid_argument("PlaceRecognizer: enhance none|clahe");
    return p;
}
}  // namespace

PlaceRecognizer::PlaceRecognizer(const PlaceRecognizerParams& p)
    : p_(Checked(p)), net_([q = p_] {
          auto net = std::make_unique<cv::dnn::Net>(
              cv::dnn::readNetFromONNX(q.onnx, q.engine == "new" ? cv::dnn::ENGINE_NEW : cv::dnn::ENGINE_CLASSIC));
          if (net->empty()) throw std::runtime_error("PlaceRecognizer: cannot load " + q.onnx);
          return net;
      }) {
    net_.Local();   // load on the constructing thread: a bad file fails here, not at the first query
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

cv::Mat PlaceRecognizer::PreprocessRgb(const cv::Mat& rgb, int width, int height) {
    if (rgb.empty() || rgb.type() != CV_8UC3) throw std::invalid_argument("PlaceRecognizer: expected CV_8UC3 (RGB)");
    cv::Mat f;
    rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);
    if (f.cols != width || f.rows != height) {
        cv::Mat r;
        cv::resize(f, r, cv::Size(width, height), 0, 0, cv::INTER_AREA);   // all three channels together, as the exporter
        f = r;
    }
    const int sz[4] = {1, 3, height, width};
    cv::Mat blob(4, sz, CV_32F);
    std::vector<cv::Mat> ch;
    cv::split(f, ch);   // R, G, B
    for (int c = 0; c < 3; ++c) {
        cv::Mat plane(height, width, CV_32F, blob.ptr<float>(0, c));
        ch[size_t(c)].convertTo(plane, CV_32F, 1.0 / kStd[c], -kMean[c] / kStd[c]);
    }
    return blob;
}

cv::Mat PlaceRecognizer::Enhance(const cv::Mat& image, const std::string& enhance, bool rgb) {
    if (enhance == "none") return image;
    if (enhance != "clahe") throw std::invalid_argument("PlaceRecognizer: enhance none|clahe");
    const cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat out;
    if (!rgb) {
        if (image.type() != CV_8UC1) throw std::invalid_argument("PlaceRecognizer: expected CV_8UC1");
        clahe->apply(image, out);
        return out;
    }
    if (image.type() != CV_8UC3) throw std::invalid_argument("PlaceRecognizer: expected CV_8UC3 (RGB)");
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_RGB2Lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);
    clahe->apply(ch[0], ch[0]);   // lightness only: hue and saturation stay
    cv::merge(ch, lab);
    cv::cvtColor(lab, out, cv::COLOR_Lab2RGB);
    return out;
}

std::vector<float> PlaceRecognizer::Describe(const cv::Mat& gray) const {
    return Run(Preprocess(Enhance(gray, p_.enhance, false), p_.width, p_.height));
}

std::vector<float> PlaceRecognizer::DescribeRgb(const cv::Mat& rgb) const {
    return Run(PreprocessRgb(Enhance(rgb, p_.enhance, true), p_.width, p_.height));
}

std::vector<float> PlaceRecognizer::Run(const cv::Mat& blob) const {
    cv::dnn::Net& net = net_.Local();
    net.setInput(blob);
    const cv::Mat out = net.forward().reshape(1, 1);
    if (out.total() != 256) throw std::runtime_error("PlaceRecognizer: expected a 256-D output");
    std::vector<float> d(out.begin<float>(), out.end<float>());
    // The graph ends in L2Norm already; renormalising guards the fp error only (|d| = 1.000000 in SMOKE §4r).
    double n = 0;
    for (float v : d) n += double(v) * v;
    n = std::sqrt(n);
    if (n > 0) for (float& v : d) v = float(v / n);
    return d;
}

dnn::ModelRecord PlaceRecognizer::Record() const {
    return {"eigenplaces_r50_256@" + std::to_string(p_.width) + "x" + std::to_string(p_.height) +
                (p_.enhance == "none" ? "" : "+" + p_.enhance),
            p_.onnx, "cv_dnn_" + p_.engine};
}

std::string PlaceRecognizer::Name() const {
    return "eigenplaces_r50_256@" + std::to_string(p_.width) + "x" + std::to_string(p_.height) +
           (p_.enhance == "none" ? "" : "+" + p_.enhance) + "/cv_dnn_" + p_.engine;
}

}  // namespace ariadne::perception
