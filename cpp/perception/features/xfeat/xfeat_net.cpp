#include "perception/features/xfeat/xfeat_net.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ariadne::perception::xfeat {

namespace {

constexpr float kBnEps = 1e-5f;     // torch.nn.BatchNorm2d default
constexpr float kInEps = 1e-5f;     // torch.nn.InstanceNorm2d default

const Tensor& Need(const TensorMap& m, const std::string& name, size_t ndim) {
    const auto it = m.find(name);
    if (it == m.end()) throw std::runtime_error("xfeat: weights lack " + name);
    if (it->second.shape.size() != ndim || it->second.f32.empty()) throw std::runtime_error("xfeat: bad tensor " + name);
    return it->second;
}

/// A convolution from `prefix`.weight [+ .bias], optionally followed by the BatchNorm at `bn` (folded in).
Conv2d MakeConv(const TensorMap& m, const std::string& weight, const std::string& bias, const std::string& bn,
                int stride, int pad, bool relu) {
    const Tensor& w = Need(m, weight, 4);
    Conv2d c;
    c.cout = w.shape[0];
    c.cin = w.shape[1];
    c.k = w.shape[2];
    if (w.shape[3] != c.k) throw std::runtime_error("xfeat: non-square kernel " + weight);
    c.stride = stride;
    c.pad = pad;
    c.relu = relu;
    const int row = c.cin * c.k * c.k;
    c.W = cv::Mat(c.cout, row, CV_32F);
    std::copy(w.f32.begin(), w.f32.end(), c.W.ptr<float>());   // (cout, cin, k, k) C order == row layout of im2col
    c.b.assign(size_t(c.cout), 0.0f);
    if (!bias.empty()) {
        const Tensor& b = Need(m, bias, 1);
        if (b.shape[0] != c.cout) throw std::runtime_error("xfeat: bias size " + bias);
        c.b = b.f32;
    }
    if (!bn.empty()) {
        const Tensor& mu = Need(m, bn + ".running_mean", 1);
        const Tensor& var = Need(m, bn + ".running_var", 1);
        for (int o = 0; o < c.cout; ++o) {
            const float s = 1.0f / std::sqrt(var.f32[size_t(o)] + kBnEps);
            float* r = c.W.ptr<float>(o);
            for (int j = 0; j < row; ++j) r[j] *= s;
            c.b[size_t(o)] = (c.b[size_t(o)] - mu.f32[size_t(o)]) * s;
        }
    }
    return c;
}

/// upstream BasicLayer: Conv2d(bias=False) -> BatchNorm2d(affine=False) -> ReLU.
Conv2d Basic(const TensorMap& m, const std::string& p, int stride, int pad) {
    return MakeConv(m, p + ".layer.0.weight", "", p + ".layer.1", stride, pad, true);
}

FMap Run(const FMap& in, const Conv2d& L) {
    if (in.c != L.cin) throw std::logic_error("xfeat: channel mismatch");
    const int ho = (in.h + 2 * L.pad - L.k) / L.stride + 1;
    const int wo = (in.w + 2 * L.pad - L.k) / L.stride + 1;
    cv::Mat col;
    if (L.k == 1 && L.stride == 1 && L.pad == 0) {
        col = in.data;
    } else {
        // im2col: row (c, ky, kx) holds input(c, oy*s - pad + ky, ox*s - pad + kx), zero outside.
        col = cv::Mat::zeros(in.c * L.k * L.k, ho * wo, CV_32F);
        for (int c = 0; c < in.c; ++c) {
            const float* src = in.data.ptr<float>(c);
            for (int ky = 0; ky < L.k; ++ky) {
                for (int kx = 0; kx < L.k; ++kx) {
                    float* dst = col.ptr<float>((c * L.k + ky) * L.k + kx);
                    for (int oy = 0; oy < ho; ++oy) {
                        const int iy = oy * L.stride - L.pad + ky;
                        if (iy < 0 || iy >= in.h) continue;
                        const float* s = src + size_t(iy) * size_t(in.w);
                        float* d = dst + size_t(oy) * size_t(wo);
                        for (int ox = 0; ox < wo; ++ox) {
                            const int ix = ox * L.stride - L.pad + kx;
                            if (ix >= 0 && ix < in.w) d[ox] = s[ix];
                        }
                    }
                }
            }
        }
    }
    FMap out{L.cout, ho, wo, cv::Mat()};
    cv::gemm(L.W, col, 1.0, cv::noArray(), 0.0, out.data);
    for (int o = 0; o < L.cout; ++o) {
        float* r = out.data.ptr<float>(o);
        const float b = L.b[size_t(o)];
        for (int j = 0; j < ho * wo; ++j) {
            const float v = r[j] + b;
            r[j] = L.relu ? std::max(v, 0.0f) : v;
        }
    }
    return out;
}

FMap RunSeq(FMap x, const std::vector<Conv2d>& seq) {
    for (const auto& L : seq) x = Run(x, L);
    return x;
}

void AddInPlace(FMap& a, const FMap& b) {
    if (a.c != b.c || a.h != b.h || a.w != b.w) throw std::logic_error("xfeat: add shape mismatch");
    a.data += b.data;
}

/// torch.nn.functional.interpolate(mode='bilinear', align_corners=False): source = scale*(dst+0.5)-0.5, clamped below
/// at 0; the upper neighbour is clamped to the last index.
FMap Bilinear(const FMap& in, int ho, int wo) {
    FMap out{in.c, ho, wo, cv::Mat(in.c, ho * wo, CV_32F)};
    const float sy = float(in.h) / float(ho), sx = float(in.w) / float(wo);
    std::vector<int> x0(static_cast<size_t>(wo)), x1(static_cast<size_t>(wo));
    std::vector<float> lx(static_cast<size_t>(wo));
    for (int x = 0; x < wo; ++x) {
        const float s = std::max(sx * (float(x) + 0.5f) - 0.5f, 0.0f);
        x0[size_t(x)] = int(s);
        x1[size_t(x)] = x0[size_t(x)] + (x0[size_t(x)] < in.w - 1 ? 1 : 0);
        lx[size_t(x)] = s - float(x0[size_t(x)]);
    }
    for (int c = 0; c < in.c; ++c) {
        const float* src = in.data.ptr<float>(c);
        float* dst = out.data.ptr<float>(c);
        for (int y = 0; y < ho; ++y) {
            const float s = std::max(sy * (float(y) + 0.5f) - 0.5f, 0.0f);
            const int y0 = int(s), y1 = y0 + (y0 < in.h - 1 ? 1 : 0);
            const float ly = s - float(y0);
            const float* r0 = src + size_t(y0) * size_t(in.w);
            const float* r1 = src + size_t(y1) * size_t(in.w);
            for (int x = 0; x < wo; ++x) {
                const size_t i = size_t(x);
                const float top = (1.0f - lx[i]) * r0[x0[i]] + lx[i] * r0[x1[i]];
                const float bot = (1.0f - lx[i]) * r1[x0[i]] + lx[i] * r1[x1[i]];
                dst[size_t(y) * size_t(wo) + i] = (1.0f - ly) * top + ly * bot;
            }
        }
    }
    return out;
}

/// The 8x8 space-to-depth unfold (upstream _unfold2d): channel a*8+b of cell (Y, X) is pixel (8Y+a, 8X+b).
FMap Unfold8(const FMap& x) {
    const int hf = x.h / 8, wf = x.w / 8;
    FMap out{64, hf, wf, cv::Mat(64, hf * wf, CV_32F)};
    const float* src = x.data.ptr<float>(0);
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b) {
            float* dst = out.data.ptr<float>(a * 8 + b);
            for (int Y = 0; Y < hf; ++Y)
                for (int X = 0; X < wf; ++X) dst[Y * wf + X] = src[size_t(8 * Y + a) * size_t(x.w) + size_t(8 * X + b)];
        }
    return out;
}

/// Value of a single-channel map at an integer index, zero outside (grid_sample padding_mode='zeros').
inline float At(const float* m, int h, int w, int y, int x) {
    return (x >= 0 && x < w && y >= 0 && y < h) ? m[size_t(y) * size_t(w) + size_t(x)] : 0.0f;
}

/// PyTorch's cubic convolution weights (A = -0.75), as grid_sample(mode='bicubic') uses.
inline void CubicWeights(float t, float w[4]) {
    constexpr float A = -0.75f;
    const auto c1 = [&](float x) { return ((A + 2) * x - (A + 3)) * x * x + 1; };           // |x| <= 1
    const auto c2 = [&](float x) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; };     // 1 < |x| < 2
    w[0] = c2(t + 1);
    w[1] = c1(t);
    w[2] = c1(1 - t);
    w[3] = c2(2 - t);
}

/// Upstream's sampling convention (InterpolateSparse2d): positions are normalised by (W-1, H-1) but sampled with
/// align_corners=False, so a pixel x lands at x * Wm / (W-1) - 0.5 on a map of width Wm. Out-of-map taps read 0.
inline float SampleCoord(float v, int full, int map) { return v * float(map) / float(full - 1) - 0.5f; }

}  // namespace

XFeatNet::XFeatNet(const std::string& weights_atb, Engine engine) : engine_(engine) {
    const TensorMap m = LoadAtb(weights_atb);
    skip1_ = MakeConv(m, "skip1.1.weight", "skip1.1.bias", "", 1, 0, false);
    block1_ = {Basic(m, "block1.0", 1, 1), Basic(m, "block1.1", 2, 1), Basic(m, "block1.2", 1, 1),
               Basic(m, "block1.3", 2, 1)};
    block2_ = {Basic(m, "block2.0", 1, 1), Basic(m, "block2.1", 1, 1)};
    block3_ = {Basic(m, "block3.0", 2, 1), Basic(m, "block3.1", 1, 1), Basic(m, "block3.2", 1, 0)};
    block4_ = {Basic(m, "block4.0", 2, 1), Basic(m, "block4.1", 1, 1), Basic(m, "block4.2", 1, 1)};
    block5_ = {Basic(m, "block5.0", 2, 1), Basic(m, "block5.1", 1, 1), Basic(m, "block5.2", 1, 1),
               Basic(m, "block5.3", 1, 0)};
    fusion_ = {Basic(m, "block_fusion.0", 1, 1), Basic(m, "block_fusion.1", 1, 1),
               MakeConv(m, "block_fusion.2.weight", "block_fusion.2.bias", "", 1, 0, false)};
    heatmap_head_ = {Basic(m, "heatmap_head.0", 1, 0), Basic(m, "heatmap_head.1", 1, 0),
                     MakeConv(m, "heatmap_head.2.weight", "heatmap_head.2.bias", "", 1, 0, false)};
    keypoint_head_ = {Basic(m, "keypoint_head.0", 1, 0), Basic(m, "keypoint_head.1", 1, 0),
                      Basic(m, "keypoint_head.2", 1, 0),
                      MakeConv(m, "keypoint_head.3.weight", "keypoint_head.3.bias", "", 1, 0, false)};
    if (keypoint_head_.back().cout != 65 || heatmap_head_.back().cout != 1 || fusion_.back().cout != 64) {
        throw std::runtime_error("xfeat: unexpected head sizes");
    }
}

NetOutput XFeatNet::Forward(const cv::Mat& gray) const {
    if ((gray.type() != CV_8UC1 && gray.type() != CV_32FC1) || gray.rows % 32 || gray.cols % 32 || gray.empty()) {
        throw std::invalid_argument("xfeat: input must be 8-bit or float gray with sides that are multiples of 32");
    }
    // InstanceNorm2d(1): per-image mean and BIASED variance, eps inside the sqrt.
    FMap x{1, gray.rows, gray.cols, cv::Mat()};
    gray.reshape(1, 1).convertTo(x.data, CV_32F);
    double sum = 0, sq = 0;
    const float* p = x.data.ptr<float>(0);
    const size_t n = size_t(x.h) * size_t(x.w);
    for (size_t i = 0; i < n; ++i) sum += p[i];
    const double mean = sum / double(n);
    for (size_t i = 0; i < n; ++i) sq += (p[i] - mean) * (p[i] - mean);
    const float inv = float(1.0 / std::sqrt(sq / double(n) + kInEps));
    x.data = (x.data - float(mean)) * inv;

    return engine_ == Engine::kOpenCvDnn ? ForwardDnn(x) : ForwardGemm(x);
}

NetOutput XFeatNet::ForwardGemm(const FMap& x) const {
    // skip1: AvgPool2d(4, stride 4) then a 1x1 convolution with bias.
    FMap pooled{1, x.h / 4, x.w / 4, cv::Mat(1, (x.h / 4) * (x.w / 4), CV_32F)};
    {
        const float* s = x.data.ptr<float>(0);
        float* d = pooled.data.ptr<float>(0);
        for (int Y = 0; Y < pooled.h; ++Y)
            for (int X = 0; X < pooled.w; ++X) {
                float acc = 0;
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b) acc += s[size_t(4 * Y + a) * size_t(x.w) + size_t(4 * X + b)];
                d[Y * pooled.w + X] = acc / 16.0f;
            }
    }
    FMap x1 = RunSeq(x, block1_);
    AddInPlace(x1, Run(pooled, skip1_));
    const FMap x2 = RunSeq(x1, block2_);
    const FMap x3 = RunSeq(x2, block3_);
    const FMap x4 = RunSeq(x3, block4_);
    const FMap x5 = RunSeq(x4, block5_);
    FMap fused = x3;
    fused.data = x3.data.clone();
    AddInPlace(fused, Bilinear(x4, x3.h, x3.w));
    AddInPlace(fused, Bilinear(x5, x3.h, x3.w));
    NetOutput o;
    o.feats = RunSeq(fused, fusion_);
    o.heatmap = RunSeq(o.feats, heatmap_head_);
    float* hm = o.heatmap.data.ptr<float>(0);
    for (int i = 0; i < o.heatmap.h * o.heatmap.w; ++i) hm[i] = 1.0f / (1.0f + std::exp(-hm[i]));
    o.keypoint_logits = RunSeq(Unfold8(x), keypoint_head_);
    return o;
}

namespace {

cv::dnn::LayerParams ConvParams(const Conv2d& c) {
    cv::dnn::LayerParams lp;
    lp.type = "Convolution";
    lp.set("kernel_size", c.k);
    lp.set("stride", c.stride);
    lp.set("pad", c.pad);
    lp.set("num_output", c.cout);
    lp.set("bias_term", true);
    const int sz[4] = {c.cout, c.cin, c.k, c.k};
    lp.blobs.push_back(cv::Mat(4, sz, CV_32F, c.W.data).clone());
    lp.blobs.push_back(cv::Mat(1, c.cout, CV_32F, const_cast<float*>(c.b.data())).clone());
    return lp;
}

/// Builds the graph: add(name, type, params, inputs) with inputs as (layer name, output index).
struct GraphBuilder {
    cv::dnn::Net& net;
    int Add(const std::string& name, const std::string& type, cv::dnn::LayerParams lp,
            const std::vector<std::string>& inputs) {
        lp.name = name;
        lp.type = type;
        const int id = net.addLayer(name, type, lp);
        if (id < 0) throw std::runtime_error("xfeat dnn: cannot add layer " + name);
        for (size_t i = 0; i < inputs.size(); ++i) {
            const int src = inputs[i] == "data" ? 0 : net.getLayerId(inputs[i]);
            if (src < 0) throw std::runtime_error("xfeat dnn: unknown input " + inputs[i]);
            net.connect(src, 0, id, int(i));
        }
        return id;
    }
    /// conv (+ ReLU) chain; returns the name of the last layer.
    std::string Seq(const std::string& prefix, const std::vector<Conv2d>& seq, std::string in) {
        for (size_t i = 0; i < seq.size(); ++i) {
            const std::string n = prefix + "_" + std::to_string(i);
            Add(n, "Convolution", ConvParams(seq[i]), {in});
            in = n;
            if (seq[i].relu) {
                Add(n + "_relu", "ReLU", {}, {in});
                in = n + "_relu";
            }
        }
        return in;
    }
};

FMap FromBlob(const cv::Mat& blob) {
    if (blob.dims != 4 || blob.size[0] != 1) throw std::runtime_error("xfeat dnn: unexpected output blob");
    FMap f{blob.size[1], blob.size[2], blob.size[3], cv::Mat()};
    f.data = cv::Mat(f.c, f.h * f.w, CV_32F, const_cast<uchar*>(blob.data)).clone();
    return f;
}

}  // namespace

cv::dnn::Net& XFeatNet::DnnFor(int h, int w) const {
    const auto key = std::make_pair(h, w);
    Graphs& nets = nets_.Local();   // this thread's graphs
    const auto it = nets.find(key);
    if (it != nets.end()) return it->second;
    cv::dnn::Net net;
    GraphBuilder g{net};
    net.setInputsNames({"data"});
    const std::string b1 = g.Seq("b1", block1_, "data");
    cv::dnn::LayerParams pool;
    pool.set("pool", "ave");
    pool.set("kernel_size", 4);
    pool.set("stride", 4);
    g.Add("skip_pool", "Pooling", pool, {"data"});
    g.Add("skip_conv", "Convolution", ConvParams(skip1_), {"skip_pool"});
    cv::dnn::LayerParams sum;
    sum.set("operation", "sum");
    g.Add("b2_in", "Eltwise", sum, {b1, "skip_conv"});
    const std::string b2 = g.Seq("b2", block2_, "b2_in");
    const std::string b3 = g.Seq("b3", block3_, b2);
    const std::string b4 = g.Seq("b4", block4_, b3);
    const std::string b5 = g.Seq("b5", block5_, b4);
    cv::dnn::LayerParams rs;
    rs.set("interpolation", "bilinear");
    rs.set("half_pixel_centers", true);   // == PyTorch align_corners=False
    rs.set("height", h / 8);
    rs.set("width", w / 8);
    g.Add("b4_up", "Resize", rs, {b4});
    g.Add("b5_up", "Resize", rs, {b5});
    g.Add("fuse_in", "Eltwise", sum, {b3, "b4_up", "b5_up"});
    const std::string feats = g.Seq("fusion", fusion_, "fuse_in");
    const std::string heat = g.Seq("heat", heatmap_head_, feats);
    g.Add("heat_sigmoid", "Sigmoid", {}, {heat});
    // The 8x8 space-to-depth unfold as an exact convolution: 64 one-hot 8x8 kernels, stride 8.
    Conv2d unfold;
    unfold.cin = 1;
    unfold.cout = 64;
    unfold.k = 8;
    unfold.stride = 8;
    unfold.W = cv::Mat::zeros(64, 64, CV_32F);
    for (int k = 0; k < 64; ++k) unfold.W.at<float>(k, k) = 1.0f;
    unfold.b.assign(64, 0.0f);
    g.Add("unfold8", "Convolution", ConvParams(unfold), {"data"});
    g.Seq("kpt", keypoint_head_, "unfold8");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return nets.emplace(key, std::move(net)).first->second;
}

NetOutput XFeatNet::ForwardDnn(const FMap& x) const {
    cv::dnn::Net& net = DnnFor(x.h, x.w);
    const int sz[4] = {1, 1, x.h, x.w};
    net.setInput(cv::Mat(4, sz, CV_32F, x.data.data).clone(), "data");
    std::vector<cv::Mat> outs;
    const std::string feats = "fusion_" + std::to_string(fusion_.size() - 1);
    const std::string kpt = "kpt_" + std::to_string(keypoint_head_.size() - 1);
    net.forward(outs, std::vector<std::string>{feats, kpt, "heat_sigmoid"});
    if (outs.size() != 3) throw std::runtime_error("xfeat dnn: expected 3 outputs");
    return {FromBlob(outs[0]), FromBlob(outs[1]), FromBlob(outs[2])};
}

std::vector<XFeatNet::Candidate> XFeatNet::Candidates(const NetOutput& o, const DetectParams& p) {
    const int hf = o.feats.h, wf = o.feats.w, H = 8 * hf, W = 8 * wf;

    // Keypoint probabilities: softmax over the 65 logits of each cell, dustbin dropped, unshuffled to full resolution.
    cv::Mat K(H, W, CV_32F);
    for (int Y = 0; Y < hf; ++Y)
        for (int X = 0; X < wf; ++X) {
            const int cell = Y * wf + X;
            float mx = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < 65; ++k) mx = std::max(mx, o.keypoint_logits.data.at<float>(k, cell));
            float e[65], sum = 0;
            for (int k = 0; k < 65; ++k) sum += e[k] = std::exp(o.keypoint_logits.data.at<float>(k, cell) - mx);
            for (int a = 0; a < 8; ++a)
                for (int b = 0; b < 8; ++b) K.at<float>(8 * Y + a, 8 * X + b) = e[a * 8 + b] / sum;
        }

    // NMS: a pixel survives if it equals the max of its k x k window (padding counts as -inf) and beats the threshold.
    // Survivors in row-major order, as torch.nonzero returns them.
    const int r = p.nms_kernel / 2;
    std::vector<Candidate> out;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float v = K.at<float>(y, x);
            if (!(v > p.detection_threshold)) continue;
            float mx = v;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    const int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < H && xx >= 0 && xx < W) mx = std::max(mx, K.at<float>(yy, xx));
                }
            if (v == mx) out.push_back({cv::Point(x, y), 0.0f});
        }

    const float* heat = o.heatmap.data.ptr<float>(0);
    for (Candidate& c : out) {
        const float px = float(c.pos.x), py = float(c.pos.y);
        // nearest on K (full resolution): std::nearbyint, i.e. round half to even, as PyTorch does
        const int nx = int(std::nearbyint(SampleCoord(px, W, W))), ny = int(std::nearbyint(SampleCoord(py, H, H)));
        const float kv = At(K.ptr<float>(), H, W, ny, nx);
        // bilinear on the reliability map (1/8 resolution)
        const float sx = SampleCoord(px, W, wf), sy = SampleCoord(py, H, hf);
        const int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
        const float tx = sx - float(x0), ty = sy - float(y0);
        const float hv = (1 - ty) * ((1 - tx) * At(heat, hf, wf, y0, x0) + tx * At(heat, hf, wf, y0, x0 + 1)) +
                         ty * ((1 - tx) * At(heat, hf, wf, y0 + 1, x0) + tx * At(heat, hf, wf, y0 + 1, x0 + 1));
        c.score = kv * hv;
        if (c.pos.x == 0 && c.pos.y == 0) c.score = -1.0f;   // upstream marks (0,0) as padding
    }
    return out;
}

cv::Mat XFeatNet::SampleDescriptors(const NetOutput& o, const std::vector<cv::Point2f>& pts) {
    const int hf = o.feats.h, wf = o.feats.w, H = 8 * hf, W = 8 * wf;
    // Per-cell L2-normalised feats, sampled bicubically at each point, then L2-normalised again.
    cv::Mat M = o.feats.data.clone();   // 64 x (hf*wf)
    for (int j = 0; j < hf * wf; ++j) {
        float nn = 0;
        for (int c = 0; c < 64; ++c) nn += M.at<float>(c, j) * M.at<float>(c, j);
        const float s = 1.0f / std::max(std::sqrt(nn), 1e-12f);
        for (int c = 0; c < 64; ++c) M.at<float>(c, j) *= s;
    }
    cv::Mat out(int(pts.size()), 64, CV_32F);
    for (size_t i = 0; i < pts.size(); ++i) {
        const float sx = SampleCoord(pts[i].x, W, wf), sy = SampleCoord(pts[i].y, H, hf);
        const int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
        float wx[4], wy[4];
        CubicWeights(sx - float(x0), wx);
        CubicWeights(sy - float(y0), wy);
        float* desc = out.ptr<float>(int(i));
        float nn = 0;
        for (int c = 0; c < 64; ++c) {
            const float* m = M.ptr<float>(c);
            float acc = 0;
            for (int a = 0; a < 4; ++a) {
                float rowv = 0;
                for (int b = 0; b < 4; ++b) rowv += wx[b] * At(m, hf, wf, y0 - 1 + a, x0 - 1 + b);
                acc += wy[a] * rowv;
            }
            desc[c] = acc;
            nn += acc * acc;
        }
        const float s = 1.0f / std::max(std::sqrt(nn), 1e-12f);
        for (int c = 0; c < 64; ++c) desc[c] *= s;
    }
    return out;
}

Detection XFeatNet::Detect(const cv::Mat& gray, const DetectParams& p) const {
    const NetOutput o = Forward(gray);
    const std::vector<Candidate> cand = Candidates(o, p);
    // Best first; ties broken by detection order (upstream's argsort is unstable, so ties are unspecified there).
    std::vector<size_t> idx(cand.size());
    std::iota(idx.begin(), idx.end(), size_t(0));
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return cand[a].score > cand[b].score; });
    if (int(idx.size()) > p.top_k) idx.resize(size_t(p.top_k));

    Detection d;
    for (size_t i : idx) {
        if (!(cand[i].score > 0)) continue;   // upstream: valid = scores > 0
        d.keypoints.emplace_back(float(cand[i].pos.x), float(cand[i].pos.y));
        d.scores.push_back(cand[i].score);
    }
    d.descriptors = SampleDescriptors(o, d.keypoints);
    return d;
}

}  // namespace ariadne::perception::xfeat
