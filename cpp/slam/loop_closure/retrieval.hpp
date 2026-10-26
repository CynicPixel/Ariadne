#pragma once
// ---------------------------------------------------------------------------
// Intra-agent loop-candidate retrieval (D73 part 2b; design robust-backend.md §8): which past keyframes might
// show the same place as the current one. A global descriptor per keyframe (one float vector; the runtime adapter
// fills it, today with 1.6's mean-bit descriptor, a labelled PLACEHOLDER until 2.1's EigenPlaces), cosine
// similarity, and a temporal exclusion window: the keyframes just before the query look alike because they ARE
// adjacent, and odometry already relates them. Retrieval only proposes; geometry (verifier -> degeneracy gate ->
// PCM -> GNC) decides. Plain types only: slam may not include collaboration or runtime (arch §3).
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace ariadne::slam {

struct RetrievalParams {
    int exclude_recent = 50;         // keyframes before the query that are never candidates
    double exclude_seconds = 10.0;   // ... nor any keyframe within this many seconds of it
    double min_score = 0.8;          // cosine similarity [ASSUMPTION: placeholder descriptor, calibrated in 2b]
    int top_k = 3;
};

class LoopRetrieval {
public:
    explicit LoopRetrieval(const RetrievalParams& p = {}) : p_(p) {}

    void Add(uint64_t id, int64_t t_ns, std::vector<float> descriptor) {
        double n = 0.0;
        for (float v : descriptor) n += double(v) * v;
        n = std::sqrt(n);
        if (n > 0.0)
            for (float& v : descriptor) v = float(v / n);
        db_.push_back({id, t_ns, std::move(descriptor)});
    }

    /// Up to top_k (id, score) with score >= min_score, best first; ties by smaller id (deterministic).
    std::vector<std::pair<uint64_t, double>> Query(uint64_t id, int64_t t_ns, const std::vector<float>& d) const {
        double n = 0.0;
        for (float v : d) n += double(v) * v;
        n = std::sqrt(n);
        std::vector<std::pair<uint64_t, double>> out;
        if (!(n > 0.0)) return out;
        for (const Entry& e : db_) {
            if (e.id + uint64_t(std::max(p_.exclude_recent, 0)) >= id) continue;
            if (std::fabs(double(t_ns - e.t_ns)) * 1e-9 < p_.exclude_seconds) continue;
            if (e.d.size() != d.size()) continue;
            double s = 0.0;
            for (size_t i = 0; i < d.size(); ++i) s += double(e.d[i]) * d[i];
            s /= n;
            if (s >= p_.min_score) out.push_back({e.id, s});
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });
        if (int(out.size()) > p_.top_k) out.resize(size_t(std::max(p_.top_k, 0)));
        return out;
    }

    size_t size() const { return db_.size(); }

private:
    struct Entry {
        uint64_t id;
        int64_t t_ns;
        std::vector<float> d;
    };
    RetrievalParams p_;
    std::vector<Entry> db_;
};

}  // namespace ariadne::slam
