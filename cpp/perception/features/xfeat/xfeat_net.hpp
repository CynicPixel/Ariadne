#pragma once
// ---------------------------------------------------------------------------
// XFeat (Potje et al., "XFeat: Accelerated Features for Lightweight Image
// Matching", CVPR 2024) INFERENCE on OpenCV 5 primitives (D15). The network
// architecture and forward pass are ours; the weights are upstream's
// (verlab/accelerated_features, Apache-2.0), exported by
// python/model_tools/xfeat_reference.py. Training is not attempted.
//
// WHAT THE NETWORK IS. A small CNN on a gray image of size H x W (multiples of
// 32), giving three maps at 1/8 resolution:
//   feats    64 x H/8 x W/8   dense descriptors (L2-normalized per cell later)
//   keypoint 65 x H/8 x W/8   logits: one class per pixel of the 8x8 cell, plus
//                             a "no keypoint here" dustbin (SuperPoint's trick)
//   heatmap   1 x H/8 x W/8   reliability in (0,1)
// The keypoint branch sees the image only through an 8x8 space-to-depth unfold
// (64 channels per cell) and three 1x1 convolutions: keypoints are a cheap,
// local decision, while descriptors come from a 5-level backbone whose coarse
// levels are upsampled and summed back to 1/8 (a feature pyramid).
//
// HOW IT RUNS HERE. BatchNorm has affine=False and fixed running statistics at
// inference, so it is FOLDED into the preceding convolution at load time:
//   y = (W*x - mu) / sqrt(var + eps)  ==  (W/s)*x + (-mu/s),  s = sqrt(var+eps)
// leaving conv + bias + ReLU. Two engines run the SAME folded weights:
//   kOpenCvDnn     the graph is assembled layer by layer with the cv::dnn
//                  graph API (Net::addLayer / connect; Convolution, ReLU,
//                  Pooling, Eltwise, Resize, Sigmoid), so OpenCV 5's
//                  dispatched AVX2/AVX-512/NEON convolution kernels run it.
//                  No ONNX file and no importer: the graph is ours (D15).
//                  The 8x8 unfold is an exact "one-hot" 8x8 stride-8 conv.
//   kGemmReference every op written out explicitly (im2col + cv::gemm): the
//                  readable reference, ~100x slower without an external BLAS.
// Instance norm, softmax, NMS and sampling are written out below to match
// upstream's PyTorch semantics, including its quirks (see Detect()). A cv::dnn
// Net is NOT thread-safe (§7.4): the graphs live in a dnn::PerThread (D91), so one XFeatNet may be shared by
// threads; each calling thread builds and runs its own.
//
// Correctness is not assumed: tests/unit/test_xfeat_reference.cpp checks
// the three maps and the final keypoints/descriptors against upstream's own
// PyTorch outputs on fixed inputs (D15 acceptance test).
// ---------------------------------------------------------------------------
#include "perception/features/xfeat/atb.hpp"

#include "perception/dnn/per_thread.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <map>
#include <utility>

#include <string>
#include <vector>

namespace ariadne::perception::xfeat {

/// A C x H x W feature map stored as a C x (H*W) CV_32F matrix (one row per channel): the layout GEMM wants.
struct FMap {
    int c = 0, h = 0, w = 0;
    cv::Mat data;
};

struct Conv2d {
    cv::Mat W;            // cout x (cin*k*k), CV_32F, BatchNorm folded in
    std::vector<float> b; // cout
    int cin = 0, cout = 0, k = 1, stride = 1, pad = 0;
    bool relu = false;
};

struct NetOutput {
    FMap feats, keypoint_logits, heatmap;
};

struct Detection {
    std::vector<cv::Point2f> keypoints;   // pixel coordinates (x, y) of the input image
    std::vector<float> scores;            // keypoint probability x reliability, > 0
    cv::Mat descriptors;                  // N x 64 CV_32F, unit L2 norm per row
};

struct DetectParams {
    int top_k = 4096;
    float detection_threshold = 0.05f;    // upstream default
    int nms_kernel = 5;                   // upstream default
};

enum class Engine { kOpenCvDnn, kGemmReference };
inline const char* EngineName(Engine e) { return e == Engine::kOpenCvDnn ? "opencv_dnn_graph" : "gemm_reference"; }

class XFeatNet {
public:
    /// Loads upstream's weights from an ATB blob; throws on any missing or mis-shaped tensor.
    explicit XFeatNet(const std::string& weights_atb, Engine engine = Engine::kOpenCvDnn);
    Engine engine() const { return engine_; }

    /// The network on a gray image whose sides are multiples of 32, 8-bit or float, fed as raw 0..255 as upstream
    /// does (float lets a caller resize the way upstream's preprocess_tensor does, without re-quantising to 8 bits).
    NetOutput Forward(const cv::Mat& gray) const;

    /// upstream XFeat.detectAndCompute (sparse): NMS on the 8x8-unshuffled keypoint probabilities, score = probability
    /// x reliability, top-k, bicubic descriptor sampling, L2 normalisation. = Candidates + top-k + SampleDescriptors.
    Detection Detect(const cv::Mat& gray, const DetectParams& p = {}) const;

    /// A pixel of the network input that survives NMS and the threshold, with upstream's score ((0,0) scores -1).
    struct Candidate {
        cv::Point pos;
        float score = 0.0f;
    };
    /// Detect's first half: every NMS survivor, row-major, scored; no top-k (callers select their own way, §7.3).
    static std::vector<Candidate> Candidates(const NetOutput& o, const DetectParams& p);
    /// Detect's second half at ANY pixels (x, y) of the network input: upstream's bicubic sampler on the per-cell
    /// L2-normalised dense map, then L2 normalisation. N x 64 CV_32F, row i for pts[i].
    static cv::Mat SampleDescriptors(const NetOutput& o, const std::vector<cv::Point2f>& pts);

private:
    NetOutput ForwardGemm(const FMap& x) const;
    NetOutput ForwardDnn(const FMap& x) const;
    cv::dnn::Net& DnnFor(int h, int w) const;   // one graph per input size (Resize targets are fixed)

    Engine engine_;
    using Graphs = std::map<std::pair<int, int>, cv::dnn::Net>;
    dnn::PerThread<Graphs> nets_{[] { return std::make_unique<Graphs>(); }};
    Conv2d skip1_;
    std::vector<Conv2d> block1_, block2_, block3_, block4_, block5_, fusion_, heatmap_head_, keypoint_head_;
};

}  // namespace ariadne::perception::xfeat
