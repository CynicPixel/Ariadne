// COOL day-1 loader-switch probe (plan 0.3; register INV2).
//
// Built once against the STOCK OpenCV 5.0.0 install, then run twice: with the stock libraries, and with
// LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib first. It answers the one question the "same binary, loader switch only"
// benchmark (arch §26.3) rests on: does the switch actually load COOL's libraries? If the sonames differ, the
// dynamic loader silently keeps the stock ones and every "COOL" number would be fake. So the probe prints:
//   - the OpenCV version and an FNV-1a hash of cv::getBuildInformation() (stock and COOL differ);
//   - every libopencv_* path mapped into the process (/proc/self/maps): the ground truth of what loaded;
//   - per-op output checksums on a fixed synthetic frame (output equivalence: a speedup with different outputs must
//     be disclosed);
//   - per-op median timings over 30 repeats, after 5 warm-ups, at a fixed thread count (first look only; the
//     benchmark proper is plan 7.3 with its own fixed definition).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

static uint64_t Fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}
static uint64_t MatHash(const cv::Mat& m) {
    cv::Mat c = m.isContinuous() ? m : m.clone();
    return Fnv(c.data, c.total() * c.elemSize());
}

static void Time(const char* name, const std::function<cv::Mat()>& op) {
    cv::Mat out;
    for (int i = 0; i < 5; ++i) out = op();
    std::vector<double> ms;
    for (int i = 0; i < 30; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        out = op();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("op %-28s median_ms %9.4f  p90_ms %9.4f  out_hash %016llx\n", name, ms[15], ms[27],
                (unsigned long long)MatHash(out));
}

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 1;
    cv::setNumThreads(threads);
    const std::string info = cv::getBuildInformation();
    std::printf("version %s\nbuild_info_fnv %016llx\nthreads %d\n", cv::getVersionString().c_str(),
                (unsigned long long)Fnv(info.data(), info.size()), cv::getNumThreads());
    std::set<std::string> libs;
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
        const auto p = line.find('/');
        if (p != std::string::npos && line.find("libopencv_") != std::string::npos) libs.insert(line.substr(p));
    }
    for (const auto& l : libs) std::printf("loaded %s\n", l.c_str());

    // A fixed 1280x720 textured frame (the sim rig's size), deterministic.
    cv::Mat bgr(720, 1280, CV_8UC3);
    cv::RNG rng(12345);
    rng.fill(bgr, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(bgr, bgr, cv::Size(0, 0), 2.0);
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mapx(gray.size(), CV_32FC1), mapy(gray.size(), CV_32FC1);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {   // a mild radial warp, like rectification
            const double dx = x - 640.0, dy = y - 360.0, r2 = (dx * dx + dy * dy) / (640.0 * 640.0);
            mapx.at<float>(y, x) = float(640.0 + dx * (1.0 + 0.05 * r2));
            mapy.at<float>(y, x) = float(360.0 + dy * (1.0 + 0.05 * r2));
        }
    cv::Mat shifted;
    cv::warpAffine(gray, shifted, (cv::Mat_<double>(2, 3) << 1, 0, 1.5, 0, 1, -0.75), gray.size());
    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(gray, pts, 500, 0.01, 8);

    Time("cvtColor_BGR2GRAY", [&] { cv::Mat o; cv::cvtColor(bgr, o, cv::COLOR_BGR2GRAY); return o; });
    Time("remap_linear", [&] { cv::Mat o; cv::remap(gray, o, mapx, mapy, cv::INTER_LINEAR); return o; });
    Time("resize_half_area", [&] { cv::Mat o; cv::resize(gray, o, cv::Size(), 0.5, 0.5, cv::INTER_AREA); return o; });
    Time("GaussianBlur_5x5", [&] { cv::Mat o; cv::GaussianBlur(gray, o, cv::Size(5, 5), 0); return o; });
    Time("pyrDown", [&] { cv::Mat o; cv::pyrDown(gray, o); return o; });
    Time("buildOpticalFlowPyramid_L3", [&] {
        std::vector<cv::Mat> pyr;
        cv::buildOpticalFlowPyramid(gray, pyr, cv::Size(21, 21), 3);
        return pyr.back().clone();
    });
    Time("calcOpticalFlowPyrLK_500", [&] {
        std::vector<cv::Point2f> next;
        std::vector<uchar> st;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(gray, shifted, pts, next, st, err, cv::Size(21, 21), 3);
        return cv::Mat(next).clone();
    });
    Time("Sobel_dx", [&] { cv::Mat o; cv::Sobel(gray, o, CV_16S, 1, 0); return o; });
    Time("threshold_otsu", [&] { cv::Mat o; cv::threshold(gray, o, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); return o; });
    return 0;
}
