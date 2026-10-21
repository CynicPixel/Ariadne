#pragma once
// ---------------------------------------------------------------------------
// Place index over unit global descriptors (PERCEPTION_2.1.md §11.2-11.3):
// top-k by cosine, with an admission filter (a loop-closure query admits only
// keyframes older than a time window).
//
//   exact  one cv::gemm of the query against every stored vector. Exact,
//          deterministic, N x 256 multiply-adds (a few us at thousands of
//          keyframes): the reference and the default.
//   annoy  cv::ANNIndex (Annoy: a forest of random-projection trees, approximate),
//          DIST_ANGULAR. Read from the pinned source (§11.2):
//          - addItems numbers items from 0 on EVERY call, so the index is always
//            rebuilt from the full matrix here;
//          - Annoy cannot add after build(): Query() rebuilds when items were
//            added since the last build;
//          - it cannot filter during search: Query() over-fetches
//            k + overfetch and filters after.
//          - Its multithreaded build seeds every thread alike over
//            hardware_concurrency() threads: repeatable on one machine (tested),
//            not guaranteed across core counts (a stated risk).
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/features.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::perception {

class PlaceIndex {
public:
    enum class Backend { kExact, kAnnoy };
    struct Hit {
        uint64_t id = 0;
        float cosine = 0.0f;
    };

    explicit PlaceIndex(int dim, Backend backend = Backend::kExact, int trees = 10, int seed = 1, int overfetch = 50)
        : dim_(dim), backend_(backend), trees_(trees), seed_(seed), overfetch_(overfetch) {
        if (dim <= 0 || trees <= 0 || overfetch < 0) throw std::invalid_argument("PlaceIndex: bad parameters");
    }

    std::string Name() const {
        return backend_ == Backend::kExact ? "exact" : "annoy/trees" + std::to_string(trees_);
    }
    size_t size() const { return ids_.size(); }

    void Add(uint64_t id, const std::vector<float>& d) {
        if (static_cast<int>(d.size()) != dim_) throw std::invalid_argument("PlaceIndex: dimension");
        cv::Mat row(1, dim_, CV_32F, const_cast<float*>(d.data()));
        data_.push_back(row.clone());
        ids_.push_back(id);
        stale_ = true;
    }

    /// Top-k admitted items by cosine, best first (ties: smaller id first).
    std::vector<Hit> Query(const std::vector<float>& q, int k, const std::function<bool(uint64_t)>& admit = {}) {
        if (static_cast<int>(q.size()) != dim_) throw std::invalid_argument("PlaceIndex: dimension");
        std::vector<Hit> out;
        if (ids_.empty() || k <= 0) return out;
        const cv::Mat qm(1, dim_, CV_32F, const_cast<float*>(q.data()));
        std::vector<size_t> cand;
        if (backend_ == Backend::kExact) {
            cand.resize(ids_.size());
            std::iota(cand.begin(), cand.end(), size_t(0));
        } else {
            Rebuild();
            const int n = std::min<int>(static_cast<int>(ids_.size()), k + overfetch_);
            cv::Mat idx, dist;
            ann_->knnSearch(qm, idx, dist, n);
            for (int i = 0; i < idx.cols; ++i) {
                const int j = idx.at<int>(0, i);
                if (j >= 0) cand.push_back(static_cast<size_t>(j));
            }
        }
        // Score the candidates exactly (annoy's distances are approximate only in which items it visits).
        cv::Mat sims;
        cv::gemm(qm, data_, 1.0, cv::noArray(), 0.0, sims, cv::GEMM_2_T);
        for (size_t j : cand) {
            if (admit && !admit(ids_[j])) continue;
            out.push_back({ids_[j], sims.at<float>(0, static_cast<int>(j))});
        }
        std::sort(out.begin(), out.end(), [](const Hit& a, const Hit& b) {
            return a.cosine != b.cosine ? a.cosine > b.cosine : a.id < b.id;
        });
        if (static_cast<int>(out.size()) > k) out.resize(static_cast<size_t>(k));
        return out;
    }

private:
    void Rebuild() {
        if (!stale_ && ann_) return;
        ann_ = cv::ANNIndex::create(dim_, cv::ANNIndex::DIST_ANGULAR);
        ann_->setSeed(seed_);           // before addItems, as the API requires
        ann_->addItems(data_);          // ids 0..n-1 = rows of data_
        ann_->build(trees_);
        stale_ = false;
    }

    int dim_;
    Backend backend_;
    int trees_, seed_, overfetch_;
    cv::Mat data_;                      // n x dim, CV_32F, unit rows
    std::vector<uint64_t> ids_;
    cv::Ptr<cv::ANNIndex> ann_;
    bool stale_ = true;
};

}  // namespace ariadne::perception
