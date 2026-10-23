#pragma once
// ---------------------------------------------------------------------------
// The parent's side of the live collaboration (design LIVE_COLLAB §2-§5, D174): one CollabFleet over a CommsNode, fed
// by the agents' ariadne_collab_workers, stepped on SIM time at the runtime's barrier.
//
//   Attach(k, fd)   a connection to agent k's collab worker; a reader thread files its #ckf records, #fresp replies
//                   and #cwm watermarks.
//   Step(now)       at every radio tick (the runtime's 0.5 s ladder inside each barrier, D218: one tick per 5 s
//                   barrier starves membership): (lockstep) wait until every attached worker's watermark is >= now; deliver the
//                   network up to now; release every record with t_final <= now (sim time, never arrival time);
//                   tick the fleet. Features answer through CollabNode's deferred path: the first ask sends #freq and
//                   returns "not ready"; the reply is handed over at the first tick >= ask + feature_latency_s, and in
//                   lockstep that tick waits for it, so wall-clock arrival never decides the order (§3).
//   Demo mode (lockstep false): records release when they arrive (at the current tick), replies when they are in.
//   T_O0_O()        per agent k merged with agent 0 (the same adopted frame): T_G_W(0)^-1 T_G_W(k), the runtime's
//                   T_W_O feedback (§4; B's mutex-guarded T_W_O[k]). Linked(): the comms state for A1's linked[k].
//   Telemetry(now)  one JSON line for the console's comms panel (§5).
// The fleet's nodes ARE the runtime's radio nodes with collab: true (one node per participant, the operator
// included: it has no keyframes but relays and may be elected broker); `customize` adds the D175 coordination
// mailbox to each. The runtime sends patches/bids/consensus through fleet().node(k) and never AddKeyframe()s itself:
// keyframes enter only through Release (final odometry, #ckf), so SLAM and coordination share one radio and ledger.
// GROUND TRUTH IS EVALUATION ONLY: the records' gt feeds edges.csv and injections, never a hook's decision.
// ---------------------------------------------------------------------------
#include "runtime/agent/collab_fleet.hpp"
#include "runtime/live/collab_link.hpp"
#include "runtime/live/link_frame.hpp"

#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ariadne::runtime::live {

struct CollabLiveParams {
    bool lockstep = true;              // gate runs: replays equal live (§3); demo: false
    double feature_latency_s = 3.0;    // [ASSUMPTION] sim-time model of one ALIKED extraction in the collab worker
                                       // (~2.3 s/kf measured on the laptop, LIVE_COLLAB §2); recorded in the manifest
    double wait_timeout_s = 60.0;      // lockstep: a worker silent this long (wall) is a hard error, not a hang
};

class CollabLive {
public:
    CollabLive(const CollabConfig& cfg, const std::vector<ac::AgentId>& fleet, ac::CommsNode& comms,
               const perception::HardPairMatcher& lightglue, uint64_t seed, int64_t t_first_ns, CollabLiveParams p,
               std::ostream* edges_csv, std::function<void(ac::AgentId, ac::NodeHooks&)> customize = {})
        : p_(p), fleet_ids_(fleet), enc_(ac::ParseLocalEncoding(cfg.local_encoding)), comms_(comms),
          fleet_(cfg, fleet, comms, lightglue, seed, t_first_ns,
                 [this](ac::AgentId k, uint64_t kf) { return Features(k, kf); },
                 [this](const ac::KeyframeRef& r) { return GtCamera(r); }, edges_csv, std::move(customize)) {}

    ~CollabLive() { Shutdown(); }
    CollabLive(const CollabLive&) = delete;
    CollabLive& operator=(const CollabLive&) = delete;

    /// fd connected to agent k's collab worker (Connect(path)); this object owns it from here.
    void Attach(ac::AgentId k, int fd) {
        std::lock_guard<std::mutex> lk(mu_);
        Peer& peer = peers_[k];
        peer.fd = fd;
        Send(k, "#hello", EncodeHello({"parent", k}));
        peer.reader = std::thread([this, k, fd] { Read(k, fd); });
    }

    void Step(core::Timestamp now) {
        now_ = now;
        if (p_.lockstep) WaitWatermarks(now.ns);
        fleet_.Deliver(now);
        std::vector<std::pair<ac::AgentId, CollabStoreRecord>> due;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& [k, peer] : peers_)
                while (!peer.records.empty() && (!p_.lockstep || peer.records.front().first.t_final_ns <= now.ns)) {
                    due.emplace_back(k, std::move(peer.records.front().first));
                    n_features_[{k, due.back().second.id}] = peer.records.front().second;
                    peer.records.pop_front();
                }
        }
        // Lockstep: in (t_final, agent, id) order, as pass B's release list; demo: arrival order per agent.
        if (p_.lockstep)
            std::stable_sort(due.begin(), due.end(), [](const auto& x, const auto& y) {
                return std::tie(x.second.t_final_ns, x.first, x.second.id) < std::tie(y.second.t_final_ns, y.first, y.second.id);
            });
        for (auto& [k, r] : due) {
            gt_[{k, r.id}] = r.has_gt ? std::optional<core::SE3>(r.gt_T_W_B * r.T_B_C) : std::nullopt;
            fleet_.Release(k, r, now, n_features_.at({k, r.id}));
            released_[k].push_back(std::move(r));
            ++stats_.released;
        }
        fleet_.Tick(now);
    }

    /// Agent k's odometry frame in agent 0's (T_O0_O), for agents that adopted the same frame as agent 0.
    std::map<ac::AgentId, core::SE3> T_O0_O() const {
        std::map<ac::AgentId, core::SE3> out;
        const auto& n0 = fleet_.node(fleet_ids_.front());
        const auto T0 = n0.T_G_W();
        if (!T0) return out;
        for (ac::AgentId k : fleet_ids_) {
            const auto& n = fleet_.node(k);
            if (const auto T = n.T_G_W(); T && n.T_G_W_anchor() == n0.T_G_W_anchor()) out[k] = T0->inverse() * *T;
        }
        return out;
    }

    /// k is linked when agent 0's node currently has it in its connected component (a partitioned agent is not).
    std::map<ac::AgentId, bool> Linked() const {
        std::map<ac::AgentId, bool> out;
        const auto members = fleet_.node(fleet_ids_.front()).Component().members;
        for (ac::AgentId k : fleet_ids_) out[k] = std::find(members.begin(), members.end(), k) != members.end();
        return out;
    }

    /// One JSON line (no newline) for the console's comms panel.
    std::string Telemetry(core::Timestamp now) const {
        std::ostringstream o;
        o << "{\"t_ns\":" << now.ns << ",\"released\":" << stats_.released << ",\"freq\":" << stats_.requests
          << ",\"fresp\":" << stats_.replies << ",\"gone\":" << stats_.gone << ",\"agents\":[";
        bool first = true;
        for (ac::AgentId k : fleet_ids_) {
            const auto& n = fleet_.node(k);
            const auto& s = n.stats();
            o << (first ? "" : ",") << "{\"agent\":" << k << ",\"broker\":" << n.Component().broker
              << ",\"merged\":" << (n.T_G_W() ? "true" : "false") << ",\"anchor\":" << n.T_G_W_anchor()
              << ",\"edges_verified\":" << s.edges_verified << ",\"solves\":" << s.solves
              << ",\"answers_deferred\":" << s.answers_deferred << "}";
            first = false;
        }
        const auto& vs = fleet_.verify_stats();
        o << "],\"verify\":{\"calls\":" << vs.calls << ",\"gated\":" << vs.gated << "},\"kept_inter_edges\":"
          << KeptCount() << ",\"cut_active\":" << (comms_.cut_active() ? "true" : "false") << ",\"injections\":[";
        for (size_t i = 0; i < fleet_.injections().size(); ++i) {
            const auto& st = fleet_.injections()[i];
            o << (i ? "," : "") << "{\"planted\":" << (st.planted ? "true" : "false") << ",\"kept_solves\":"
              << st.kept_solves << ",\"solves_seen\":" << st.solves_seen << ",\"weight\":"
              << (std::isfinite(st.weight) ? st.weight : -1.0) << "}";
        }
        o << "]}";
        return o.str();
    }

    // --- panel controls (LIVE_COLLAB §5) ---
    /// "Cut comms": the groups as a scripted partition from `now` until RestoreComms (logged in comms.cuts()).
    void CutComms(core::Timestamp now, std::vector<std::vector<ac::AgentId>> groups) { comms_.Cut(now, std::move(groups)); }
    void RestoreComms(core::Timestamp now) { comms_.Restore(now); }
    /// "Inject false closure": the fixed experiment, 4 m and 45 deg, on the pair with the most kept true loops.
    std::optional<size_t> InjectFalse(core::Timestamp now) {
        return fleet_.InjectNow(now, core::Vec3(4.0, 0.0, 0.0), 45.0);
    }

    CollabFleet& fleet() { return fleet_; }
    const std::map<ac::AgentId, std::vector<CollabStoreRecord>>& released() const { return released_; }

    void Shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stopping_) return;
            stopping_ = true;
            for (auto& [k, peer] : peers_)
                if (peer.fd >= 0) ::shutdown(peer.fd, SHUT_RDWR);
        }
        cv_.notify_all();
        for (auto& [k, peer] : peers_) {
            if (peer.reader.joinable()) peer.reader.join();
            if (peer.fd >= 0) ::close(peer.fd), peer.fd = -1;
        }
    }

    struct Stats {
        uint64_t released = 0, requests = 0, replies = 0, gone = 0, bad_frames = 0;
    };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(mu_);
        return stats_;
    }

private:
    struct Peer {
        int fd = -1;
        std::thread reader;
        std::deque<std::pair<CollabStoreRecord, uint32_t>> records;   // #ckf in arrival (= release) order
        std::map<uint64_t, std::optional<CollabLocalFeatures>> replies;
        int64_t watermark_ns = INT64_MIN;
        bool closed = false;
    };
    struct Ask {
        int64_t asked_ns = 0;
    };

    void Send(ac::AgentId k, const char* topic, const std::vector<uint8_t>& b) {   // under mu_
        Peer& peer = peers_.at(k);
        if (!WriteFrame(peer.fd, topic, b.data(), b.size())) peer.closed = true;
    }

    void Read(ac::AgentId k, int fd) {
        LinkFrame f;
        try {
            while (ReadFrame(fd, f)) {
                std::lock_guard<std::mutex> lk(mu_);
                Peer& peer = peers_.at(k);
                if (f.topic == "#ckf") {
                    uint32_t n = 0;
                    CollabStoreRecord r = DecodeRecord(f.payload, &n);
                    peer.records.emplace_back(std::move(r), n);
                } else if (f.topic == "#fresp") {
                    CollabFeatureReply rep = DecodeFeatureReply(f.payload);
                    peer.replies[rep.kf] = std::move(rep.features);
                } else if (f.topic == "#cwm") {
                    peer.watermark_ns = std::max(peer.watermark_ns, DecodeWatermark(f.payload));
                } else {
                    ++stats_.bad_frames;
                }
                cv_.notify_all();
            }
        } catch (const std::exception&) {
            std::lock_guard<std::mutex> lk(mu_);
            ++stats_.bad_frames;
        }
        std::lock_guard<std::mutex> lk(mu_);
        peers_.at(k).closed = true;
        cv_.notify_all();
    }

    void WaitWatermarks(int64_t t_ns) {
        std::unique_lock<std::mutex> lk(mu_);
        const bool ok = cv_.wait_for(lk, std::chrono::duration<double>(p_.wait_timeout_s), [&] {
            if (stopping_) return true;
            for (const auto& [k, peer] : peers_)
                if (!peer.closed && peer.watermark_ns < t_ns) return false;
            return true;
        });
        if (!ok) throw std::runtime_error("collab live: a collab worker's watermark stalled (lockstep)");
    }

    /// The FeatureSource: deferred through CollabNode (nullopt = not ready yet).
    std::optional<core::FeatureResponseValue> Features(ac::AgentId k, uint64_t kf) {
        std::unique_lock<std::mutex> lk(mu_);
        const auto key = std::make_pair(k, kf);
        auto ask = asks_.find(key);
        if (ask == asks_.end()) {
            asks_[key] = Ask{now_.ns};
            if (peers_.count(k)) Send(k, "#freq", EncodeFeatureRequest(kf));
            ++stats_.requests;
            return std::nullopt;
        }
        const int64_t due = ask->second.asked_ns + int64_t(std::llround(p_.feature_latency_s * 1e9));
        if (now_.ns < due) return std::nullopt;
        Peer& peer = peers_.at(k);
        auto rep = peer.replies.find(kf);
        if (rep == peer.replies.end()) {
            if (!p_.lockstep) return std::nullopt;
            const bool ok = cv_.wait_for(lk, std::chrono::duration<double>(p_.wait_timeout_s), [&] {
                return stopping_ || peer.closed || peer.replies.count(kf);
            });
            rep = peer.replies.find(kf);
            if (!ok || rep == peer.replies.end()) throw std::runtime_error("collab live: a feature reply stalled (lockstep)");
        }
        ++stats_.replies;
        std::optional<CollabLocalFeatures> f = std::move(rep->second);
        peer.replies.erase(rep);
        asks_.erase(key);
        if (!f) {   // gone: an empty response, so the broker's verifier rejects it ("too_few_features")
            ++stats_.gone;
            return CollabFleet::Response(kf, {}, {}, {}, CameraOf(k, kf), enc_);
        }
        return CollabFleet::Response(kf, f->keypoints, f->descriptors, f->points3d, CameraOf(k, kf), enc_);
    }

    std::array<float, 5> CameraOf(ac::AgentId k, uint64_t kf) const {   // from the released record
        const auto it = released_.find(k);
        if (it != released_.end())
            for (const auto& r : it->second)
                if (r.id == kf) return r.camera;
        return {};
    }

    std::optional<core::SE3> GtCamera(const ac::KeyframeRef& r) const {
        const auto it = gt_.find({r.agent, r.keyframe});
        return it == gt_.end() ? std::nullopt : it->second;
    }

    size_t KeptCount() const {
        size_t n = 0;
        for (const auto& [k, v] : fleet_.kept_edges()) n += v.last_weight >= 0.5;
        return n;
    }

    CollabLiveParams p_;
    std::vector<ac::AgentId> fleet_ids_;
    ac::LocalEncoding enc_;
    ac::CommsNode& comms_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::map<ac::AgentId, Peer> peers_;
    std::map<std::pair<ac::AgentId, uint64_t>, Ask> asks_;
    std::map<std::pair<ac::AgentId, uint64_t>, uint32_t> n_features_;
    std::map<std::pair<ac::AgentId, uint64_t>, std::optional<core::SE3>> gt_;
    std::map<ac::AgentId, std::vector<CollabStoreRecord>> released_;
    core::Timestamp now_;
    Stats stats_;
    CollabFleet fleet_;   // last: its hooks capture `this`
};

}  // namespace ariadne::runtime::live
