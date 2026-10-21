#include "perception/features/xfeat_engine.hpp"

#include "perception/features/grid_select.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace ariadne::perception {

namespace {

xfeat::Engine DnnEngine(const std::string& s) {
    return s == "gemm_reference" ? xfeat::Engine::kGemmReference : xfeat::Engine::kOpenCvDnn;
}

bool SamePixels(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || a.size() != b.size() || a.type() != b.type()) return false;
    const size_t row = size_t(a.cols) * a.elemSize();
    for (int y = 0; y < a.rows; ++y) {
        if (std::memcmp(a.ptr(y), b.ptr(y), row) != 0) return false;
    }
    return true;
}

double Quantile(std::vector<double> v, double q) {
    if (v.empty()) return -1.0;
    const size_t k = std::min(v.size() - 1, static_cast<size_t>(q * double(v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
    return v[k];
}

}  // namespace

XFeatEngine::XFeatEngine(const FeatureParams& params)
    : params_(params), net_(params.xfeat_weights, DnnEngine(params.xfeat_dnn_engine)) {}

cv::Size XFeatEngine::NetSize(cv::Size image) const {
    if (params_.xfeat_input == "pad") return {(image.width + 31) / 32 * 32, (image.height + 31) / 32 * 32};
    return {params_.xfeat_input_width > 0 ? params_.xfeat_input_width : image.width / 32 * 32,
            params_.xfeat_input_height > 0 ? params_.xfeat_input_height : image.height / 32 * 32};
}

const XFeatEngine::Maps& XFeatEngine::Run(const cv::Mat& gray) const {
    if (gray.empty() || gray.type() != CV_8UC1) throw std::invalid_argument("XFeatEngine: expected CV_8UC1");
    if (SamePixels(gray, cached_input_)) {
        ++cache_hits_;
        return cached_;
    }
    const auto t0 = std::chrono::steady_clock::now();
    Maps m;
    m.image = gray.size();
    m.net = NetSize(m.image);
    if (m.net.width < 32 || m.net.height < 32) throw std::invalid_argument("XFeatEngine: image smaller than 32 px");
    cv::Mat in;
    if (m.net == m.image) {
        in = gray;   // already a multiple of 32: the D50-tested 8-bit path, untouched
    } else if (params_.xfeat_input == "pad") {
        cv::copyMakeBorder(gray, in, 0, m.net.height - m.image.height, 0, m.net.width - m.image.width,
                           cv::BORDER_REPLICATE);
    } else {
        // upstream: x.float() then F.interpolate(bilinear, align_corners=False). Resizing the float image keeps the
        // 0..255 values unquantised, as upstream's are.
        cv::Mat f;
        gray.convertTo(f, CV_32F);
        cv::resize(f, in, m.net, 0, 0, cv::INTER_LINEAR);
        m.sx = float(m.image.width) / float(m.net.width);
        m.sy = float(m.image.height) / float(m.net.height);
    }
    m.out = net_.Forward(in);
    cached_ = std::move(m);
    cached_input_ = gray.clone();
    forward_ms_.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return cached_;
}

std::vector<cv::KeyPoint> XFeatEngine::Select(const Maps& m, const cv::Mat& mask, int budget,
                                              std::vector<cv::Point2f>& net_pos) const {
    xfeat::DetectParams dp;
    dp.detection_threshold = params_.xfeat_detection_threshold;
    dp.nms_kernel = params_.xfeat_nms_kernel;
    const std::vector<xfeat::XFeatNet::Candidate> cand = xfeat::XFeatNet::Candidates(m.out, dp);

    std::vector<cv::KeyPoint> kps;
    net_pos.clear();
    for (const auto& c : cand) {
        if (!(c.score > 0)) continue;                                                 // upstream: valid = scores > 0
        if (c.pos.x >= m.image.width || c.pos.y >= m.image.height) continue;          // pad region (pad mode only)
        const cv::Point2f pt(float(c.pos.x) * m.sx, float(c.pos.y) * m.sy);         // upstream: mkpts * [rw, rh]
        if (!mask.empty()) {
            const int x = std::clamp(cvRound(pt.x), 0, mask.cols - 1), y = std::clamp(cvRound(pt.y), 0, mask.rows - 1);
            if (mask.at<uchar>(y, x) == 0) continue;                                  // BEFORE selection (§7.3)
        }
        // size = one 8x8 cell in image pixels (XFeat has no scale); angle -1 = none; class_id -> net_pos.
        kps.emplace_back(pt, 8.0f * m.sx, -1.0f, c.score, 0, static_cast<int>(net_pos.size()));
        net_pos.emplace_back(float(c.pos.x), float(c.pos.y));
    }
    return GridSelect(std::move(kps), m.image, params_.grid_cols, params_.grid_rows, budget);
}

std::vector<cv::KeyPoint> XFeatEngine::Detect(const cv::Mat& gray, const cv::Mat& mask, int max_keypoints) const {
    if (gray.empty()) return {};
    const Maps& m = Run(gray);
    std::vector<cv::Point2f> net_pos;
    return Select(m, mask, max_keypoints > 0 ? max_keypoints : params_.max_keypoints, net_pos);
}

FeatureSet XFeatEngine::DetectAndCompute(const cv::Mat& gray, const cv::Mat& mask) const {
    FeatureSet out;
    if (gray.empty()) return out;
    const Maps& m = Run(gray);
    std::vector<cv::Point2f> net_pos;
    out.keypoints = Select(m, mask, params_.max_keypoints, net_pos);
    // Sample at the NETWORK pixel each keypoint was found at (exactly upstream's sampling), not at pt / scale.
    std::vector<cv::Point2f> at;
    at.reserve(out.keypoints.size());
    for (const auto& k : out.keypoints) at.push_back(net_pos[static_cast<size_t>(k.class_id)]);
    out.descriptors = xfeat::XFeatNet::SampleDescriptors(m.out, at);
    return out;
}

cv::Mat XFeatEngine::Describe(const cv::Mat& gray, const std::vector<cv::Point2f>& points,
                              std::vector<uchar>& described) const {
    described.assign(points.size(), 0);
    cv::Mat out = cv::Mat::zeros(static_cast<int>(points.size()), 64, CV_32F);
    if (points.empty()) return out;
    const Maps& m = Run(gray);
    std::vector<cv::Point2f> at;
    std::vector<size_t> rows;
    for (size_t i = 0; i < points.size(); ++i) {
        const cv::Point2f& p = points[i];
        if (!(p.x >= 0.f && p.y >= 0.f && p.x <= float(m.image.width - 1) && p.y <= float(m.image.height - 1))) continue;
        at.emplace_back(p.x / m.sx, p.y / m.sy);
        rows.push_back(i);
    }
    const cv::Mat d = xfeat::XFeatNet::SampleDescriptors(m.out, at);
    for (size_t k = 0; k < rows.size(); ++k) {
        d.row(static_cast<int>(k)).copyTo(out.row(static_cast<int>(rows[k])));
        described[rows[k]] = 1;
    }
    return out;
}

std::map<std::string, std::string> XFeatEngine::ManifestFields() const {
    std::ostringstream in;
    in << params_.xfeat_input << ':';
    if (cached_.net.area() > 0) in << cached_.net.width << 'x' << cached_.net.height;
    else in << "not_run";
    return {{"xfeat_dnn_engine", xfeat::EngineName(net_.engine())},
            {"xfeat_input", in.str()},
            {"xfeat_weights", params_.xfeat_weights}};
}

std::string XFeatEngine::ResultsYaml() const {
    std::ostringstream o;
    o << "xfeat: {dnn_engine: " << xfeat::EngineName(net_.engine()) << ", input: " << ManifestFields()["xfeat_input"]
      << ", opencv_threads: " << cv::getNumThreads() << ", forwards: " << forwards() << ", cache_hits: " << cache_hits_
      << ", forward_ms_median: " << Quantile(forward_ms_, 0.5) << ", forward_ms_p90: " << Quantile(forward_ms_, 0.9)
      << "}\n";
    return o.str();
}

}  // namespace ariadne::perception
