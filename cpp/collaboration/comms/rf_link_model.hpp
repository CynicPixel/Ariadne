#pragma once
// ---------------------------------------------------------------------------
// 3.3 (D92): the RFComms-derived LinkModel with wall attenuation (arch §5.7).
//
// PHYSICAL LAYER, transcribed from gz-sim8 src/systems/rf_comms/RFComms.cc
// (read 2026-10-22), so the broker reproduces what the sim's RFComms plugin
// would decide:
//   range d > max_range                      -> no link
//   PL(d)   = l0 + 10 n log10(d)             [dB]  (RFComms names l0 "received power at 1 m" but uses it as the
//                                                   path loss at 1 m -- QUIRK, reproduced)
//   P_rx    ~ N(tx_power - PL - walls * wall_loss_db, sigma^2)   [dBm], redrawn per packet
//   BER     = erfc(sqrt(P_rx / N0)), powers in W  (QUIRK: no 1/2 factor, power ratio not Eb/N0; reproduced)
//   PER     = 1 - (1 - BER)^bytes                 (QUIRK: exponent in BYTES, not bits; reproduced)
// The wall term is ours (arch §5.7): the number of solid wall blocks the straight path crosses, from a ray-cast
// against the scene's GT wall faces, times a per-wall loss (COST-231 multi-wall form).
//
// LinkState.loss_p is the per-datagram MARGINAL drop probability: E[PER(P_rx)] over the log-normal shadowing.
// That is exactly RFComms' per-packet drop probability (it redraws P_rx independently per packet), and keeps the
// draw itself in Network's keyed Philox stream. PER(P_rx) is nearly a step, so the expectation is integrated on a
// fine fixed grid over +-8 sigma (not Gauss-Hermite), tabulated once against the mean received power.
//
// SEAM LIMIT (stated): LinkModel::Sample has no datagram size, so loss_p is computed for one size,
// `datagram_bytes` (default: the serializer's path MTU, 1500). A 40-byte heartbeat sees a lower PER than a full
// fragment near the edge; COMMS_3.3 §L8 quantifies it.
//
// Other deviations: RFComms enforces `capacity` by dropping over a 1 s window; here it is the link rate
// (Network queues). RFComms has no latency; `latency_s` is ours. Positions are agent BODY positions (radio at the
// body origin).
//
// TIME BASE: Sample(t) reads t.ns as the sim clock (the GT trajectories' t_ns). Outside a trajectory the position
// is clamped to its first / last sample.
// ---------------------------------------------------------------------------
#include "collaboration/protocol/network.hpp"
#include "core/math/lie.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace ariadne::collaboration {

struct RfParams {
    // RFComms RangeConfiguration / RadioConfiguration, gz-sim8 defaults.
    double max_range_m = 50.0;
    double fading_exponent = 2.5;
    double l0_db = 40.0;
    double sigma_db = 10.0;
    double capacity_bps = 54e6;
    double tx_power_dbm = 27.0;
    double noise_floor_dbm = -90.0;
    // Ours.
    double wall_loss_db = 0.0;       // per solid wall block crossed (0 = RFComms as is)
    double latency_s = 0.001;        // [ASSUMPTION] one-hop radio latency; RFComms models none
    double datagram_bytes = 1500.0;  // size loss_p is computed for (see SEAM LIMIT)
};

/// A wall face of the scene's GT geometry: a triangle plus the outward normal of its face (pointing into free
/// space) and the id of the surface (quad) it belongs to.
struct WallTriangle {
    core::Vec3 a, b, c;
    core::Vec3 normal;
    uint64_t surface = 0;
};

/// An agent's body positions over time (sim clock, ns), time-sorted.
struct AgentTrack {
    std::vector<int64_t> t_ns;
    std::vector<core::Vec3> p;

    core::Vec3 At(int64_t t) const {
        if (t_ns.empty()) throw std::logic_error("AgentTrack: empty");
        if (t <= t_ns.front()) return p.front();
        if (t >= t_ns.back()) return p.back();
        const size_t j = size_t(std::upper_bound(t_ns.begin(), t_ns.end(), t) - t_ns.begin());
        const double w = double(t - t_ns[j - 1]) / double(t_ns[j] - t_ns[j - 1]);
        return (1.0 - w) * p[j - 1] + w * p[j];
    }
};

namespace rf {

inline double DbmToW(double dbm) { return 0.001 * std::pow(10.0, dbm / 10.0); }

/// RFComms' per-packet drop probability at a given received power.
inline double Per(double rx_dbm, double noise_dbm, double bytes) {
    const double ber = std::erfc(std::sqrt(DbmToW(rx_dbm) / DbmToW(noise_dbm)));
    if (ber >= 1.0) return 1.0;
    return 1.0 - std::exp(bytes * std::log1p(-ber));
}

/// E[PER] over P_rx ~ N(mean, sigma^2): trapezoid on a 4001-point grid over +-8 sigma.
inline double ExpectedPer(double mean_dbm, double sigma_db, double noise_dbm, double bytes) {
    if (!(sigma_db > 0.0)) return Per(mean_dbm, noise_dbm, bytes);
    constexpr int kN = 4001;
    constexpr double kSpan = 8.0;
    const double h = 2.0 * kSpan / (kN - 1);
    double s = 0.0, wsum = 0.0;
    for (int i = 0; i < kN; ++i) {
        const double z = -kSpan + h * i;
        const double w = std::exp(-0.5 * z * z) * ((i == 0 || i == kN - 1) ? 0.5 : 1.0);
        s += w * Per(mean_dbm + sigma_db * z, noise_dbm, bytes);
        wsum += w;
    }
    return s / wsum;
}

/// Solid wall blocks the segment p -> q crosses: surfaces whose face it ENTERS (outward normal against the
/// direction). A block has a face on each side, so counting entries counts blocks, not faces; a ray through a
/// quad's diagonal hits both triangles but the surface counts once. Moller-Trumbore.
inline int WallsCrossed(const core::Vec3& p, const core::Vec3& q, const std::vector<WallTriangle>& walls) {
    const core::Vec3 d = q - p;
    std::vector<uint64_t> hit;
    for (const WallTriangle& w : walls) {
        if (w.normal.dot(d) >= 0.0) continue;   // leaving a block (or parallel): not an entry
        const core::Vec3 e1 = w.b - w.a, e2 = w.c - w.a;
        const core::Vec3 pv = d.cross(e2);
        const double det = e1.dot(pv);
        if (std::abs(det) < 1e-12) continue;
        const double inv = 1.0 / det;
        const core::Vec3 tv = p - w.a;
        const double u = tv.dot(pv) * inv;
        if (u < 0.0 || u > 1.0) continue;
        const core::Vec3 qv = tv.cross(e1);
        const double v = d.dot(qv) * inv;
        if (v < 0.0 || u + v > 1.0) continue;
        const double s = e2.dot(qv) * inv;
        if (s <= 0.0 || s >= 1.0) continue;     // strictly between the two radios
        hit.push_back(w.surface);
    }
    std::sort(hit.begin(), hit.end());
    return int(std::unique(hit.begin(), hit.end()) - hit.begin());
}

}  // namespace rf

/// What Sample() saw, for the probe and the tests.
struct RfLinkDetail {
    double distance_m = 0.0;
    int walls = 0;
    double rx_mean_dbm = -std::numeric_limits<double>::infinity();
    LinkState state;
};

class RfLinkModel : public LinkModel {
public:
    RfLinkModel(RfParams p, std::map<AgentId, AgentTrack> tracks, std::vector<WallTriangle> walls)
        : p_(p), tracks_(std::move(tracks)), walls_(std::move(walls)) {
        // E[PER] against the mean received power, tabulated at 0.1 dB over noise +- 80 dB; clamped outside.
        for (int i = 0; i < kTable; ++i) {
            table_[i] = rf::ExpectedPer(TableDbm(i), p_.sigma_db, p_.noise_floor_dbm, p_.datagram_bytes);
        }
    }

    LinkState Sample(AgentId from, AgentId to, Timestamp t) const override { return Detail(from, to, t).state; }

    RfLinkDetail Detail(AgentId from, AgentId to, Timestamp t) const {
        RfLinkDetail r;
        const auto a = tracks_.find(from), b = tracks_.find(to);
        if (a == tracks_.end() || b == tracks_.end()) return r;   // an unknown agent has no radio: down
        const core::Vec3 pa = a->second.At(t.ns), pb = b->second.At(t.ns);
        r.distance_m = (pb - pa).norm();
        if (p_.max_range_m > 0.0 && r.distance_m > p_.max_range_m) return r;
        r.walls = rf::WallsCrossed(pa, pb, walls_);
        const double pl = p_.l0_db + 10.0 * p_.fading_exponent * std::log10(std::max(r.distance_m, 1e-3));
        r.rx_mean_dbm = p_.tx_power_dbm - pl - r.walls * p_.wall_loss_db;
        r.state.up = true;
        r.state.latency_s = p_.latency_s;
        r.state.loss_p = LossAt(r.rx_mean_dbm);
        r.state.bandwidth_Bps = p_.capacity_bps / 8.0;
        return r;
    }

    double LossAt(double rx_mean_dbm) const {
        const double x = (rx_mean_dbm - TableDbm(0)) / kStepDb;
        if (x <= 0.0) return table_[0];
        if (x >= kTable - 1) return table_[kTable - 1];
        const int i = int(x);
        const double w = x - i;
        return (1.0 - w) * table_[i] + w * table_[i + 1];
    }
    const RfParams& params() const { return p_; }

private:
    static constexpr int kTable = 1601;
    static constexpr double kStepDb = 0.1;
    double TableDbm(int i) const { return p_.noise_floor_dbm - 80.0 + kStepDb * i; }

    RfParams p_;
    std::map<AgentId, AgentTrack> tracks_;
    std::vector<WallTriangle> walls_;
    double table_[kTable];
};

}  // namespace ariadne::collaboration
