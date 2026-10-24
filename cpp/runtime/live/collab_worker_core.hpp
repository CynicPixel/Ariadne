#pragma once
// ---------------------------------------------------------------------------
// The logic of ariadne_collab_worker (design LIVE_COLLAB §1-§2, D174), one per agent, socket-agnostic so it is tested
// in-process. All DNN work of the live collaboration happens here, never in the estimator worker (D152a/D168):
//
//   #ckin  (estimator) -> MaskedDepth32 of the bundle's depth + mask, EigenPlaces now, keep rgb + depth for ALIKED
//   #crel  (estimator) -> assemble the record (no local features) -> #ckf to the parent
//   #cprog t (estimator: it has processed frame t, so every #crel with t_final <= t is already ahead of this message
//                       on the same socket) -> #cwm t to the parent: the lockstep watermark (§3)
//   #freq kf (parent)  -> ALIKED + points3d of a kept keyframe -> #fresp (no features = gone)
//
// Messages are processed one at a time in arrival order on one thread, so the place descriptor and ALIKED never run
// concurrently in this process's OpenCV pool. The two sockets (estimator, parent) are merged by the driver; within
// one socket the order is the sender's.
// ---------------------------------------------------------------------------
#include "runtime/agent/collab_frontend.hpp"
#include "runtime/agent/keyframe_records.hpp"
#include "runtime/live/collab_link.hpp"

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::runtime::live {

struct CollabWorkerStats {
    uint64_t keyframes_in = 0, releases = 0, records_sent = 0, requests = 0, replies_gone = 0;
    int64_t watermark_ns = INT64_MIN;
};

class CollabWorkerCore {
public:
    /// send(topic, payload): to the parent.
    using Send = std::function<void(const std::string&, const std::vector<uint8_t>&)>;

    /// n_features: the local feature count each #ckf advertises before extraction (the model's K: 1024 for
    /// aliked_n16_k1024), which the broker's byte predictor prices fetches with.
    CollabWorkerCore(CollabFeatureExtractor& x, Send send, uint32_t n_features)
        : x_(x), send_(std::move(send)), n_features_(n_features) {}

    void OnKeyframeIn(const CollabKeyframeIn& k) {
        ++stats_.keyframes_in;
        Pending p;
        p.t_create_ns = k.t_ns;
        p.T_B_C = k.T_B_C;
        p.gt_T_W_B = k.gt_T_W_B;
        p.camera = {float(k.fx), float(k.fy), float(k.cx), float(k.cy), float(k.baseline_m)};
        p.global = x_.Describe(k.rgb);
        x_.Keep(k.id, k.rgb, MaskedDepth32(k.depth16, k.mask), CollabPinhole{k.fx, k.fy, k.cx, k.cy});
        pending_[k.id] = std::move(p);
    }

    void OnRelease(const CollabRelease& rel) {
        ++stats_.releases;
        const auto it = pending_.find(rel.id);
        if (it == pending_.end()) throw std::logic_error("collab worker: a release of a keyframe never received");
        CollabStoreRecord r;
        r.id = rel.id;
        r.t_create_ns = it->second.t_create_ns;
        r.t_final_ns = rel.t_final_ns;
        r.T_W_C = rel.T_W_C;
        r.odo_from = rel.odo_from;
        r.odo_Z = rel.odo_Z;
        r.odo_information = rel.odo_information;
        r.global = std::move(it->second.global);
        r.camera = it->second.camera;
        r.has_gt = it->second.gt_T_W_B.has_value();
        if (r.has_gt) r.gt_T_W_B = *it->second.gt_T_W_B;
        r.T_B_C = it->second.T_B_C;
        pending_.erase(it);
        x_.Released(r.id, r.t_final_ns);
        send_("#ckf", EncodeRecord(r, n_features_));
        ++stats_.records_sent;
    }

    void OnProgress(int64_t t_ns) {
        stats_.watermark_ns = t_ns;
        x_.Expire(t_ns);
        send_("#cwm", EncodeWatermark(t_ns));
    }

    void OnFeatureRequest(const CollabFeatureRequest& q) {
        ++stats_.requests;
        CollabFeatureReply f;
        f.kf = q.kf;
        f.features = q.t_req_ns >= 0 ? x_.Features(q.kf, q.t_req_ns) : x_.Features(q.kf);
        if (!f.features) ++stats_.replies_gone;
        send_("#fresp", EncodeFeatureReply(f));
    }

    /// One frame from either socket; returns false for an unknown topic (counted by the driver, never fatal).
    bool Dispatch(const std::string& topic, const std::vector<uint8_t>& payload) {
        if (topic == "#ckin") OnKeyframeIn(DecodeKeyframeIn(payload));
        else if (topic == "#crel") OnRelease(DecodeRelease(payload));
        else if (topic == "#cprog") OnProgress(DecodeWatermark(payload));
        else if (topic == "#freq") OnFeatureRequest(DecodeFeatureRequest(payload));
        else return false;
        return true;
    }

    const CollabWorkerStats& stats() const { return stats_; }
    size_t pending() const { return pending_.size(); }

private:
    struct Pending {
        int64_t t_create_ns = 0;
        std::vector<float> global;
        std::array<float, 5> camera{};
        std::optional<core::SE3> gt_T_W_B;
        core::SE3 T_B_C;
    };
    CollabFeatureExtractor& x_;
    Send send_;
    uint32_t n_features_;
    std::map<uint64_t, Pending> pending_;
    CollabWorkerStats stats_;
};

}  // namespace ariadne::runtime::live
