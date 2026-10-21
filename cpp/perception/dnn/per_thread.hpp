#pragma once
// ---------------------------------------------------------------------------
// The DNN execution contract's threading rule (arch §7.4, D91): "one
// cv::dnn::Net instance per thread; Net is not treated as thread-safe; the
// worker pool owns N nets loaded from the same file". Every cv::dnn wrapper
// (XFeatNet, PlaceRecognizer, HardPairMatcher) holds its network state in a
// PerThread<T> instead of a bare member, so ONE wrapper object can be shared by
// a pool and each calling thread runs its own network.
//
// WHY NOT A MUTEX AROUND ONE NET. That is safe too, but it serialises the
// pool. Why not Net copies: cv::dnn::Net is a ref-counted handle (copying shares
// the graph and its buffers), and LightGlueMatcher::clone() shares its Net as
// well (PERCEPTION_2.1 §12.3), so a "copy" is not an instance.
//
// MECHANICS. Local() returns the calling thread's T, building it on that
// thread's first call with the factory (an ONNX load: slow, once per thread).
// The map is guarded; the T itself is touched only by its own thread. A T's
// address is stable (unique_ptr), so the reference stays valid while the
// PerThread lives. An id whose thread has exited may be reused by a later
// thread, which then inherits that T: harmless, the old owner is gone.
//
// SECOND RULE: THE PARALLELISM IS THE CALLERS'. Per-thread instances alone do
// not make outputs identical: every Net shares OpenCV's ONE process-wide pool
// (cv::parallel_for_), and two concurrent forwards get it partitioned
// differently. Measured (D91, 2026-10-22): concurrent ENGINE_NEW ALIKED runs at
// the default pool differ intermittently by up to 6e-5 px / 1.3e-6 per
// descriptor element (match indices identical); at cv::setNumThreads(1) every
// repeat is byte-identical. So a runtime that shares a wrapper across worker
// threads sets cv::setNumThreads(kPoolOpenCvThreads) once (process-wide, so it
// is the runtime's call, not a library's) and records it in the manifest. A
// single caller may keep the pool: sequential runs are byte-identical at 8.
// test_dnn_contract asserts byte identity under the rule for all three wrappers.
// ---------------------------------------------------------------------------
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace ariadne::perception::dnn {

/// OpenCV's own pool size when one wrapper serves several threads (the second rule above).
inline constexpr int kPoolOpenCvThreads = 1;

/// What ran, for the run manifest (§7.4 "record which engine actually executed each model").
struct ModelRecord {
    std::string model;    // e.g. eigenplaces_r50_256@376x240
    std::string file;     // the ONNX / weights file
    std::string engine;   // cv_dnn_classic | cv_dnn_new | opencv_dnn_graph | gemm_reference
};

template <class T>
class PerThread {
public:
    using Factory = std::function<std::unique_ptr<T>()>;

    explicit PerThread(Factory make) : make_(std::move(make)) {
        if (!make_) throw std::invalid_argument("PerThread: empty factory");
    }
    PerThread(const PerThread&) = delete;
    PerThread& operator=(const PerThread&) = delete;

    /// The calling thread's instance (built on its first call).
    T& Local() const {
        const auto id = std::this_thread::get_id();
        {
            std::lock_guard<std::mutex> g(mu_);
            const auto it = by_thread_.find(id);
            if (it != by_thread_.end()) return *it->second;
        }
        std::unique_ptr<T> t = make_();   // outside the lock: loads on different threads overlap
        if (!t) throw std::runtime_error("PerThread: factory returned null");
        std::lock_guard<std::mutex> g(mu_);
        return *by_thread_.emplace(id, std::move(t)).first->second;   // only this thread inserts this id
    }

    /// Number of threads that have an instance.
    size_t Instances() const {
        std::lock_guard<std::mutex> g(mu_);
        return by_thread_.size();
    }

private:
    Factory make_;
    mutable std::mutex mu_;
    mutable std::unordered_map<std::thread::id, std::unique_ptr<T>> by_thread_;
};

}  // namespace ariadne::perception::dnn
