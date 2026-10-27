#pragma once
// Display-only, latest-wins camera export. JPEG work never runs on the estimator
// thread. Images and tracker observations are committed by one atomic manifest.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

namespace ariadne::runtime::live {
class DemoCamera {
    struct Packet { int64_t t; cv::Mat rgb; std::vector<cv::Point2f> points; };
    std::filesystem::path dir_;
    int agent_;
    int64_t last_ = -100000000;
    std::mutex mu_;
    std::condition_variable cv_;
    std::optional<Packet> pending_;
    bool stop_ = false;
    size_t dropped_ = 0;
    std::thread thread_;
    void Run() {
        for (;;) {
            std::optional<Packet> p;
            size_t dropped;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || pending_.has_value(); });
                if (!pending_ && stop_) return;
                p = std::move(pending_); pending_.reset(); dropped = dropped_;
            }
            try {
                cv::Mat small, bgr;
                const double scale = std::min(1.0, 640.0 / p->rgb.cols);
                cv::resize(p->rgb, small, {}, scale, scale, cv::INTER_AREA);
                if (small.channels() == 3) cv::cvtColor(small, bgr, cv::COLOR_RGB2BGR);
                else bgr = small;
                const std::string stem = "cam_" + std::to_string(agent_);
                const std::string image = stem + "_" + std::to_string(p->t) + ".jpg";
                if (!cv::imwrite((dir_ / image).string(), bgr, {cv::IMWRITE_JPEG_QUALITY, 75})) continue;
                std::ostringstream j;
                j << "{\"agent\":" << agent_ << ",\"t_ns\":" << p->t << ",\"image\":\"" << image
                  << "\",\"w\":" << bgr.cols << ",\"h\":" << bgr.rows << ",\"dropped\":" << dropped << ",\"pts\":[";
                for (size_t i = 0; i < p->points.size(); ++i)
                    j << (i ? "," : "") << '[' << p->points[i].x * scale << ',' << p->points[i].y * scale << ']';
                j << "]}";
                const auto tmp = dir_ / (stem + ".tmp");
                { std::ofstream f(tmp); f << j.str(); }
                std::filesystem::rename(tmp, dir_ / (stem + ".json"));
            } catch (const std::exception&) {
                // Display loss cannot fail estimation. A stale manifest is visible in the UI.
                std::lock_guard<std::mutex> lk(mu_); ++dropped_;
            }
        }
    }
public:
    DemoCamera(std::filesystem::path dir, int agent) : dir_(std::move(dir)), agent_(agent) {
        std::filesystem::create_directories(dir_);
        thread_ = std::thread([this] { Run(); });
    }
    ~DemoCamera() {
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_one(); thread_.join();
    }
    bool Wants(int64_t t) { if (t - last_ < 100000000) return false; last_ = t; return true; }
    void Put(int64_t t, cv::Mat rgb, const std::vector<cv::Point2f>& points) {
        if (rgb.empty()) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (pending_) ++dropped_;
        pending_ = Packet{t, std::move(rgb), points};
        cv_.notify_one();
    }
};
} // namespace ariadne::runtime::live
