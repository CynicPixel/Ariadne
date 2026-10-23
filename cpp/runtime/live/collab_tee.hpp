#pragma once
// ---------------------------------------------------------------------------
// The estimator worker's side of the live collaboration (design LIVE_COLLAB §2, D174): a tee, no DNN. With
// `collab: true` the worker (ariadne_live, D168) connects to its agent's ariadne_collab_worker and calls
//   OnKeyframe(kv, gt)  in on_keyframe: #ckin (id, t, rgb, the bundle's CV_16F depth + mask, pinhole, T_B_C, gt), then
//                       the release tracker (the LoopBackend mirror: cheap, no OpenCV parallel work) and one #crel per
//                       keyframe whose odometry became final at this keyframe;
//   OnFrame(t_ns)       in on_frame: #cprog t (every #crel with t_final <= t is already ahead of it on this socket).
// A send to a dead collab worker returns false; the worker keeps estimating (collaboration is not on VO's path) and
// the tee goes quiet, counted.
// ---------------------------------------------------------------------------
#include "runtime/agent/collab_frontend.hpp"
#include "runtime/live/collab_link.hpp"
#include "runtime/live/link_frame.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <mutex>
#include <optional>
#include <string>

namespace ariadne::runtime::live {

class CollabTee {
public:
    /// fd: connected to the agent's collab worker (Connect(path)); the tee owns it from here.
    CollabTee(int fd, uint32_t agent) : fd_(fd) { Send("#hello", EncodeHello({"estimator", agent})); }
    ~CollabTee() {
        if (fd_ >= 0) ::close(fd_);
    }
    CollabTee(const CollabTee&) = delete;
    CollabTee& operator=(const CollabTee&) = delete;

    void OnKeyframe(const tools::KeyframeView& kv, const std::optional<core::SE3>& gt_T_W_B) {
        const runtime::KeyframeBundle& b = kv.bundle;
        CollabKeyframeIn k;
        k.id = kv.keyframe.id;
        k.t_ns = kv.keyframe.t_ns;
        k.fx = b.K.fx, k.fy = b.K.fy, k.cx = b.K.cx, k.cy = b.K.cy, k.baseline_m = kv.baseline_m;
        k.T_B_C = kv.T_B_C;
        k.gt_T_W_B = gt_T_W_B;
        k.rgb = b.rgb, k.depth16 = b.depth, k.mask = b.mask;
        Send("#ckin", EncodeKeyframeIn(k));
        for (const CollabRelease& r : tracker_.OnKeyframe(k.id, k.t_ns, *kv.graph)) Send("#crel", EncodeRelease(r));
    }

    void OnFrame(int64_t t_ns) { Send("#cprog", EncodeWatermark(t_ns)); }

    uint64_t failed_sends() const { return failed_; }
    std::pair<uint64_t, uint64_t> Unreleased() const { return tracker_.Unreleased(); }

private:
    void Send(const char* topic, const std::vector<uint8_t>& b) {
        std::lock_guard<std::mutex> lk(mu_);
        if (dead_ || !WriteFrame(fd_, topic, b.data(), b.size())) dead_ = true, ++failed_;
    }
    int fd_;
    std::mutex mu_;
    bool dead_ = false;
    uint64_t failed_ = 0;
    CollabReleaseTracker tracker_;
};

}  // namespace ariadne::runtime::live
