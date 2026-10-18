#pragma once
// ---------------------------------------------------------------------------
// Windowed stereo bundle adjustment (arch §11.3-§11.6; design LOCAL_SLAM_2.3.md),
// OpenCV-free: residuals, analytic Jacobians, Huber/IRLS assembly, and the
// Schur elimination of landmarks. OpenCV 5's cv::detail::LevMarqBase drives
// it through window_ba_levmarq.hpp (D51), exactly as for the pose graph.
//
// STATE. Keyframe poses T_W_C (left perturbation T <- exp(d) T, d = [rho; phi],
// the PoseNode convention) and landmarks as world points X. The oldest
// keyframe (and any other marked fixed) is not a variable: the gauge (§11.3).
//
// RESIDUAL of an observation z of X in keyframe k (left camera, or the right
// camera of the rectified pair, offset by the baseline along x):
//     Xc = R^T (X - t) - (right ? (B,0,0) : 0),   r = pi(Xc) - z
// JACOBIANS (design §2):
//     dXc/drho = -R^T,   dXc/dphi = R^T [X]x,   dXc/dX = R^T
//     dpi/dXc  = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]
// HUBER on s = |r| / sigma (delta = 2.0, §11.3): rho(s) = s^2/2 (s <= delta),
// delta (s - delta/2) otherwise; IRLS weight w = 1 or delta/s. The energy the
// LM driver sees IS the robust cost, and H, g are the IRLS-weighted ones at
// the linearisation point, so g is exactly the robust cost's gradient.
//
// SCHUR (§11.5). H = [[Hcc, Hcl], [Hlc, Hll]] with Hll block-diagonal (3x3 per
// landmark). The LM driver damps the FULL diagonal; damping is applied first,
// then landmarks are eliminated:
//     S = Hcc - sum_l Hcl Hll^-1 Hlc,  S dc = -(gc - sum_l Hcl Hll^-1 gl),
//     dl = Hll^-1 (-gl - Hlc dc)
// and the FULL step (cameras + landmarks) is returned, because the driver's
// predicted-decrease and step-norm tests use the whole vector. S is dense
// (<= 6 * 14 = 84 rows at window 15): Eigen LDLT with the positive-pivot guard.
//
// A landmark that projects behind a camera (z <= z_floor) makes the energy
// enormous (finite), so the driver rejects that step and raises lambda,
// instead of evaluating pi at a singularity.
//
// INERTIAL (imu.mode full, INERTIAL_2.4.md §4.2). When BaWindow::imu holds
// preintegrations, every window keyframe also carries its body velocity v_W
// and biases b_g, b_a (a 9-vector, additive), and the energy gains
//     sum_k ½ r_k^T Σ_k^-1 r_k        Forster's inertial factor k -> k+1
//                                     (slam/inertial/preintegration.hpp)
//   + sum_k ½ |b_k+1 - b_k|^2 / (σ_rw^2 Δt_k)    bias random walk
//   + ½ (b_0 - b̄)^T Λ (b_0 - b̄)                 bias prior on window[0]
// No robust kernel on these (Gaussian by construction). A pose's fixedness and
// its velocity/bias's are SEPARATE: window[0]'s pose is the gauge, its v and b
// are free. Layout [poses 6 nc | inertial 9 ni | landmarks 3 nl]: landmarks
// never touch v or b, so the Schur elimination is unchanged and only the
// reduced system S grows (dim_c covers poses AND inertial blocks). With no
// inertial blocks every number is what it was before (visual-only windows
// are bit-identical).
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "slam/inertial/preintegration.hpp"

#include <Eigen/Dense>

#include <algorithm>

#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace ariadne::slam {

using core::Mat3;
using core::Mat6;
using core::SE3;
using core::Vec3;
using core::Vec6;
using Vec2 = Eigen::Vector2d;

struct BaCamera {
    double fx = 0, fy = 0, cx = 0, cy = 0, baseline = 0;
};

struct WindowObservation {
    int kf = 0;            // index into the window's keyframes
    uint64_t landmark = 0;
    Vec2 z;                // rectified pixel
    bool right = false;    // observed in the right camera
    double sigma = 1.0;    // pixel standard deviation
};

/// The visual-inertial part of a window (empty `pre`: visual-only). Indexed like BaWindow::T_W_C.
struct BaInertial {
    SE3 T_C_B;                         // IMU body in the rectified left camera (T_W_B = T_W_C T_C_B)
    Vec3 g_W = Vec3::Zero();           // gravity in the VO world: estimated at initialisation, an input here
    std::vector<Vec3> v, bg, ba;       // per keyframe: body velocity in W, gyro bias, accel bias
    std::vector<Preintegrated> pre;    // pre[k]: keyframe k -> k + 1 (size K - 1), at bias (bg, ba)_lin of k
    double gyro_random_walk = 0.0;     // rad/s^2/sqrt(Hz), as the caller inflated it
    double accel_random_walk = 0.0;    // m/s^3/sqrt(Hz)
    bool has_prior = false;            // bias prior on keyframe 0: r = [bg_0; ba_0] - prior_b, information prior_info
    Vec6 prior_b = Vec6::Zero();
    Mat6 prior_info = Mat6::Zero();
    /// v_0 is not a variable (held at its value): set by AnchorWindow when window[0] is the only pose-determined
    /// keyframe of its group, the one case where the IMU's Galilean null space is not removed by vision (maintainer
    /// decision 2026-10-18; INERTIAL_2.4 §4.2). Otherwise false: v_0 is free, biases carry the only prior.
    bool hold_v0 = false;
};

struct BaWindow {
    BaCamera cam;
    std::vector<SE3> T_W_C;          // keyframe poses, oldest first
    std::vector<bool> fixed;         // fixed[k]: not a variable (the gauge is fixed[0])
    std::map<uint64_t, Vec3> X;      // landmarks
    std::set<uint64_t> fixed_landmarks;   // not variables (motion-only refinement; the PnP oracle)
    std::vector<WindowObservation> obs;
    double huber_delta = 2.0;
    double z_floor = 1e-3;           // m: behind / at the camera
    BaInertial imu;

    bool inertial() const { return !imu.pre.empty(); }
};

/// Throws unless the inertial part, when present, matches the window (one preintegration per consecutive pair, a
/// velocity and biases per keyframe).
inline void CheckInertial(const BaWindow& w) {
    if (!w.inertial()) return;
    const size_t K = w.T_W_C.size();
    if (w.imu.pre.size() + 1 != K || w.imu.v.size() != K || w.imu.bg.size() != K || w.imu.ba.size() != K)
        throw std::invalid_argument("BaWindow: inertial part does not match the keyframes");
    if (!(w.imu.gyro_random_walk > 0.0) || !(w.imu.accel_random_walk > 0.0))
        throw std::invalid_argument("BaWindow: bias random walk densities must be > 0");
}

struct ObsLinearization {
    bool valid = false;              // in front of the camera
    Vec2 r;
    Eigen::Matrix<double, 2, 6> Jc;
    Eigen::Matrix<double, 2, 3> Jl;
};

/// Residual and Jacobians of one observation at the given pose and point.
inline ObsLinearization LinearizeObservation(const BaCamera& cam, const SE3& T_W_C, const Vec3& X,
                                             const WindowObservation& o, double z_floor) {
    ObsLinearization l;
    const Mat3 Rt = T_W_C.R.transpose();
    Vec3 Xc = Rt * (X - T_W_C.t);
    if (o.right) Xc.x() -= cam.baseline;
    if (!(Xc.z() > z_floor)) return l;
    l.valid = true;
    const double iz = 1.0 / Xc.z();
    l.r = Vec2(cam.fx * Xc.x() * iz + cam.cx, cam.fy * Xc.y() * iz + cam.cy) - o.z;
    Eigen::Matrix<double, 2, 3> P;
    P << cam.fx * iz, 0.0, -cam.fx * Xc.x() * iz * iz, 0.0, cam.fy * iz, -cam.fy * Xc.y() * iz * iz;
    l.Jc.leftCols<3>() = -P * Rt;
    l.Jc.rightCols<3>() = P * Rt * core::Hat(X);
    l.Jl = P * Rt;
    return l;
}

/// Huber cost and IRLS weight of a whitened residual norm s.
inline void Huber(double s, double delta, double& cost, double& weight) {
    if (s <= delta) {
        cost = 0.5 * s * s;
        weight = 1.0;
    } else {
        cost = delta * (s - 0.5 * delta);
        weight = delta / s;
    }
}

// --- Observability guards (LOCAL_SLAM_2.3.md §5) -----------------------------------------------------------------
// A ray count is not observability. Two failure modes seen on EuRoC (MH_02, MH_04, MH_05, V1_03, V2_0x):
//   1. a landmark whose rays are near-parallel (hover, lost tracking): its own 3x3 information is singular;
//   2. a keyframe, or a group of keyframes, not tied to a fixed keyframe through shared landmarks: the window has
//      a free gauge (a landmark seen from one keyframe moves with it and constrains nothing).

/// Landmarks whose own UNWEIGHTED information sum J_l^T J_l / sigma^2 (poses held) gives a worst-direction standard
/// deviation 1/sqrt(lambda_min) above `max_relative_sigma` x range (range = distance to the nearest observing camera)
/// -- the §9.5 triangulation test (sigma_Z / Z), generalised to N rays. A landmark with an invalid ray fails too.
inline std::set<uint64_t> IllConditionedLandmarks(const BaWindow& w, double max_relative_sigma) {
    std::map<uint64_t, Mat3> info;
    std::map<uint64_t, double> range;
    std::set<uint64_t> ill;
    for (const auto& o : w.obs) {
        const Vec3& X = w.X.at(o.landmark);
        const SE3& T = w.T_W_C[size_t(o.kf)];
        const ObsLinearization l = LinearizeObservation(w.cam, T, X, o, w.z_floor);
        if (!l.valid) {
            ill.insert(o.landmark);
            continue;
        }
        auto it = info.try_emplace(o.landmark, Mat3::Zero()).first;
        it->second += l.Jl.transpose() * l.Jl / (o.sigma * o.sigma);
        auto r = range.try_emplace(o.landmark, std::numeric_limits<double>::infinity()).first;
        r->second = std::min(r->second, (X - T.t).norm());
    }
    for (const auto& [id, I] : info) {
        const double lmin = Eigen::SelfAdjointEigenSolver<Mat3>(I, Eigen::EigenvaluesOnly).eigenvalues()(0);
        if (!(lmin > 0.0) || 1.0 / std::sqrt(lmin) > max_relative_sigma * range.at(id)) ill.insert(id);
    }
    return ill;
}

/// Remove landmarks and every observation of them.
inline void RemoveLandmarks(BaWindow& w, const std::set<uint64_t>& ids) {
    if (ids.empty()) return;
    for (uint64_t id : ids) {
        w.X.erase(id);
        w.fixed_landmarks.erase(id);
    }
    w.obs.erase(std::remove_if(w.obs.begin(), w.obs.end(),
                               [&](const WindowObservation& o) { return ids.count(o.landmark) > 0; }),
                w.obs.end());
}

/// Tie every free keyframe to a fixed one. Two keyframes are covisible when at least `min_shared` landmarks are
/// observed by both; a breadth-first search from the fixed keyframes (ascending index, so the result is
/// deterministic) labels each reached keyframe with the fixed keyframe it hangs from. In every component the search
/// does not reach, the OLDEST keyframe is fixed (its own anchor) and the search continues from it. Returns the
/// anchor index of each keyframe; `newly_fixed` counts the keyframes fixed here.
///
/// INERTIAL windows: the IMU observes relative rotation and, given the velocity, relative position -- but NOT the
/// velocity itself. Adding c to every v_k and c (t_k - t_0) to every p_k leaves every inertial residual unchanged (a
/// Galilean null space), so a chain of inertial factors from ONE pose-determined keyframe does not tie anything. It
/// does once the anchor's group holds two pose-determined keyframes (two positions at different times fix c): an
/// inertial factor then ties consecutive keyframes. So the search follows inertial factors only out of groups with
/// >= 2 keyframes reached through fixedness or covisibility (an inertially reached keyframe does not count).
/// When window[0] (fixed, the gauge) has no covisible partner -- a dark stretch -- holding its velocity at the
/// previous solve's estimate removes the null space with real information instead: w.imu.hold_v0 is set, window[0]'s
/// group ties inertially, and the lost keyframes become IMU-determined rather than held at their predicted poses
/// (where the stiff inertial factors would pull the biases toward the prediction's error).
inline std::vector<int> AnchorWindow(BaWindow& w, int min_shared, int& newly_fixed) {
    const int K = int(w.T_W_C.size());
    std::vector<std::set<uint64_t>> seen(static_cast<size_t>(K));
    for (const auto& o : w.obs) seen[size_t(o.kf)].insert(o.landmark);
    std::vector<std::vector<int>> adj(static_cast<size_t>(K));
    for (int a = 0; a < K; ++a)
        for (int b = a + 1; b < K; ++b) {
            int shared = 0;
            for (uint64_t id : seen[size_t(a)]) shared += seen[size_t(b)].count(id) > 0;
            if (shared >= min_shared) {
                adj[size_t(a)].push_back(b);
                adj[size_t(b)].push_back(a);
            }
        }
    std::vector<int> anchor(static_cast<size_t>(K), -1);
    std::vector<int> pose_determined(static_cast<size_t>(K), 0);   // per anchor: keyframes reached without the IMU
    std::vector<int> queue;
    const bool inertial = w.inertial();
    const auto search = [&]() {
        for (;;) {
            for (size_t q = 0; q < queue.size(); ++q)
                for (int b : adj[size_t(queue[q])])
                    if (anchor[size_t(b)] < 0 && !w.fixed[size_t(b)]) {
                        anchor[size_t(b)] = anchor[size_t(queue[q])];
                        ++pose_determined[size_t(anchor[size_t(b)])];
                        queue.push_back(b);
                    }
            queue.clear();
            if (!inertial) return;
            // One inertial tie (the lowest index first), then the covisibility search again from it.
            for (int k = 0; k < K && queue.empty(); ++k) {
                if (anchor[size_t(k)] >= 0) continue;
                for (int n : {k - 1, k + 1})
                    if (n >= 0 && n < K && anchor[size_t(n)] >= 0 && pose_determined[size_t(anchor[size_t(n)])] >= 2) {
                        anchor[size_t(k)] = anchor[size_t(n)];
                        queue.push_back(k);
                        break;
                    }
            }
            if (queue.empty()) return;
        }
    };
    for (int k = 0; k < K; ++k)
        if (w.fixed[size_t(k)]) {
            anchor[size_t(k)] = k;
            pose_determined[size_t(k)] = 1;
            queue.push_back(k);
        }
    if (inertial) {
        // Covisibility first (no inertial ties yet), to see whether window[0]'s group pins the velocity by itself.
        w.imu.hold_v0 = false;
        std::vector<int> a = anchor, pd = pose_determined, q = queue;
        for (size_t i = 0; i < q.size(); ++i)
            for (int b : adj[size_t(q[i])])
                if (a[size_t(b)] < 0 && !w.fixed[size_t(b)]) {
                    a[size_t(b)] = a[size_t(q[i])];
                    ++pd[size_t(a[size_t(b)])];
                    q.push_back(b);
                }
        if (K > 1 && w.fixed[0] && pd[0] < 2) {
            w.imu.hold_v0 = true;
            pose_determined[0] = 2;   // v_0 known: window[0]'s group ties inertially
        }
    }
    search();
    newly_fixed = 0;
    for (int k = 0; k < K; ++k)
        if (anchor[size_t(k)] < 0) {
            w.fixed[size_t(k)] = true;
            anchor[size_t(k)] = k;
            pose_determined[size_t(k)] = 1;
            ++newly_fixed;
            queue.push_back(k);
            search();
        }
    return anchor;
}

/// The parameter layout: free camera poses (6 each, window order), then inertial blocks [v; bg; ba] (9 each, every
/// keyframe of an inertial window, window order; window[0]'s is [bg; ba] when hold_v0), then landmarks (3 each,
/// ascending id). dim_c() is the reduced system's size: poses and inertial blocks.
class BaLayout {
public:
    explicit BaLayout(const BaWindow& w) {
        if (w.T_W_C.size() != w.fixed.size()) throw std::invalid_argument("BaLayout: poses and fixed flags differ");
        CheckInertial(w);
        for (size_t k = 0; k < w.T_W_C.size(); ++k) cam_index_.push_back(w.fixed[k] ? -1 : n_cams_++);
        int off = 6 * n_cams_;
        for (size_t k = 0; w.inertial() && k < w.T_W_C.size(); ++k) {
            const bool held = k == 0 && w.imu.hold_v0;
            vel_off_.push_back(held ? -1 : off);
            off += held ? 0 : 3;
            bias_off_.push_back(off);
            off += 6;
        }
        n_inertial_ = int(vel_off_.size());
        dim_i_ = off - 6 * n_cams_;
        for (const auto& [id, X] : w.X) lm_index_[id] = w.fixed_landmarks.count(id) ? -1 : n_lms_++;
        if (n_cams_ == 0 && n_lms_ == 0 && n_inertial_ == 0) throw std::invalid_argument("BaLayout: nothing to optimise");
    }
    int n_cams() const { return n_cams_; }
    int n_inertial() const { return n_inertial_; }
    int n_lms() const { return n_lms_; }
    int dim_c() const { return 6 * n_cams_ + dim_i_; }
    int dim_i() const { return dim_i_; }
    int dim() const { return dim_c() + 3 * n_lms_; }
    int cam(int k) const { return cam_index_[size_t(k)]; }
    /// Offsets of keyframe k's velocity and of its biases [bg; ba] in the parameter vector; -1 when not variables
    /// (a visual-only window; v_0 under hold_v0).
    int vel(int k) const { return n_inertial_ > 0 ? vel_off_[size_t(k)] : -1; }
    int bias(int k) const { return n_inertial_ > 0 ? bias_off_[size_t(k)] : -1; }
    int lm(uint64_t id) const { return lm_index_.at(id); }

private:
    int n_cams_ = 0, n_inertial_ = 0, n_lms_ = 0, dim_i_ = 0;
    std::vector<int> cam_index_, vel_off_, bias_off_;
    std::map<uint64_t, int> lm_index_;
};

/// The Gaussian (inertial) part of the energy: inertial factors, bias random walk, bias prior. With `s` and `L`, also
/// accumulates H (into s->Hcc) and g at the current point. ONE function for the energy and the linearisation, so the
/// driver's acceptance test and its model see the same cost.
struct BaSystem;
inline double InertialTerms(const BaWindow& w, const BaLayout* L = nullptr, BaSystem* s = nullptr);

/// Robust energy of the whole window. Behind-camera observations make it huge (finite): the step is rejected.
inline double BaEnergy(const BaWindow& w) {
    double E = w.inertial() ? InertialTerms(w) : 0.0;
    for (const auto& o : w.obs) {
        const ObsLinearization l = LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], w.X.at(o.landmark), o, w.z_floor);
        if (!l.valid) return 1e300;
        double c, wt;
        Huber(l.r.norm() / o.sigma, w.huber_delta, c, wt);
        E += c;
    }
    return E;
}

/// The normal equations in block form (IRLS-weighted at the current point).
struct BaSystem {
    Eigen::MatrixXd Hcc;                                   // dim_c x dim_c
    std::map<std::pair<int, int>, Eigen::Matrix<double, 6, 3>> Hcl;   // (camera, landmark) -> block
    std::vector<Mat3> Hll;                                 // per landmark
    Eigen::VectorXd g;                                     // [gc; gl], +J^T W r
    std::vector<std::vector<int>> lm_cams;                 // per landmark: cameras that see it (sorted)
};

inline double BaLinearize(const BaWindow& w, const BaLayout& L, BaSystem& s) {
    s.Hcc = Eigen::MatrixXd::Zero(L.dim_c(), L.dim_c());
    s.Hcl.clear();
    s.Hll.assign(size_t(L.n_lms()), Mat3::Zero());
    s.g = Eigen::VectorXd::Zero(L.dim());
    s.lm_cams.assign(size_t(L.n_lms()), {});
    double E = 0.0;
    for (const auto& o : w.obs) {
        const ObsLinearization l = LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], w.X.at(o.landmark), o, w.z_floor);
        if (!l.valid) return 1e300;
        double c, wt;
        Huber(l.r.norm() / o.sigma, w.huber_delta, c, wt);
        E += c;
        const double W = wt / (o.sigma * o.sigma);
        const int ci = L.cam(o.kf), li = L.lm(o.landmark);
        if (li >= 0) {
            s.Hll[size_t(li)] += W * l.Jl.transpose() * l.Jl;
            s.g.segment<3>(L.dim_c() + 3 * li) += W * l.Jl.transpose() * l.r;
        }
        if (ci >= 0) {
            s.Hcc.block<6, 6>(6 * ci, 6 * ci) += W * l.Jc.transpose() * l.Jc;
            s.g.segment<6>(6 * ci) += W * l.Jc.transpose() * l.r;
        }
        if (ci >= 0 && li >= 0) {
            const size_t lis = size_t(li);
            auto it = s.Hcl.find({ci, li});
            const Eigen::Matrix<double, 6, 3> B = W * l.Jc.transpose() * l.Jl;
            if (it == s.Hcl.end()) {
                s.Hcl.emplace(std::make_pair(ci, li), B);
                s.lm_cams[lis].push_back(ci);
            } else {
                it->second += B;
            }
        }
    }
    for (auto& v : s.lm_cams) std::sort(v.begin(), v.end());
    if (w.inertial()) E += InertialTerms(w, &L, &s);
    return E;
}

inline double InertialTerms(const BaWindow& w, const BaLayout* L, BaSystem* s) {
    const BaInertial& m = w.imu;
    const bool assemble = L != nullptr && s != nullptr;
    // One block of the stacked Jacobian: (parameter offset, columns).
    struct Block { int off; Eigen::MatrixXd J; };
    const auto add = [&](const Eigen::VectorXd& r, const Eigen::MatrixXd& Om, const std::vector<Block>& blocks) {
        for (const Block& a : blocks) {
            if (a.off < 0) continue;
            const Eigen::MatrixXd JtOm = a.J.transpose() * Om;
            s->g.segment(a.off, a.J.cols()) += JtOm * r;
            for (const Block& b : blocks)
                if (b.off >= 0) s->Hcc.block(a.off, b.off, a.J.cols(), b.J.cols()) += JtOm * b.J;
        }
    };
    double E = 0.0;
    for (size_t k = 0; k + 1 < w.T_W_C.size(); ++k) {
        const Preintegrated& pre = m.pre[k];
        const InertialLinearization l = LinearizeInertial(pre, w.T_W_C[k] * m.T_C_B, m.v[k], m.bg[k], m.ba[k],
                                                          w.T_W_C[k + 1] * m.T_C_B, m.v[k + 1], m.g_W);
        const Mat9 Om = InformationOf<9>(pre.Sigma());
        E += 0.5 * l.r.dot(Om * l.r);
        // Bias random walk k -> k+1 over the same interval: covariance density^2 * dt per axis.
        Vec6 rb;
        rb << m.bg[k + 1] - m.bg[k], m.ba[k + 1] - m.ba[k];
        Vec6 ob;
        ob << Vec3::Constant(1.0 / (m.gyro_random_walk * m.gyro_random_walk * pre.dt())),
            Vec3::Constant(1.0 / (m.accel_random_walk * m.accel_random_walk * pre.dt()));
        E += 0.5 * rb.dot(ob.cwiseProduct(rb));
        if (!assemble) continue;
        const int ci = L->cam(int(k)), cj = L->cam(int(k + 1));
        Eigen::Matrix<double, 9, 6> Jbi;   // [bg_i, ba_i]
        Jbi << l.J_bgi, l.J_bai;
        add(l.r, Om,
            {{ci < 0 ? -1 : 6 * ci, l.J_Ti}, {cj < 0 ? -1 : 6 * cj, l.J_Tj}, {L->vel(int(k)), l.J_vi},
             {L->vel(int(k + 1)), l.J_vj}, {L->bias(int(k)), Jbi}});
        add(rb, Eigen::MatrixXd(ob.asDiagonal()),
            {{L->bias(int(k)), -Mat6::Identity()}, {L->bias(int(k + 1)), Mat6::Identity()}});
    }
    if (m.has_prior) {
        Vec6 rp;
        rp << m.bg[0], m.ba[0];
        rp -= m.prior_b;
        E += 0.5 * rp.dot(m.prior_info * rp);
        if (assemble) {
            add(rp, m.prior_info, {{L->bias(0), Mat6::Identity()}});
        }
    }
    return E;
}

/// Diagonal of the full H (cameras then landmarks), and writing it back (the driver's damping).
inline Eigen::VectorXd BaDiagonal(const BaSystem& s, const BaLayout& L) {
    Eigen::VectorXd d(L.dim());
    d.head(L.dim_c()) = s.Hcc.diagonal();
    for (int l = 0; l < L.n_lms(); ++l) d.segment<3>(L.dim_c() + 3 * l) = s.Hll[size_t(l)].diagonal();
    return d;
}

inline void BaSetDiagonal(BaSystem& s, const BaLayout& L, const Eigen::VectorXd& d) {
    s.Hcc.diagonal() = d.head(L.dim_c());
    for (int l = 0; l < L.n_lms(); ++l) s.Hll[size_t(l)].diagonal() = d.segment<3>(L.dim_c() + 3 * l);
}

/// Schur elimination with guards. Factorize() returns false (never throws) on a non-PD system so the LM driver
/// treats it as a failed step; Solve() returns the FULL step x for (H) x = -right.
class BaSchur {
public:
    bool Factorize(const BaSystem& s, const BaLayout& L) {
        ok_ = false;
        failed_landmark_ = -1;
        Hll_inv_.assign(size_t(L.n_lms()), Mat3::Zero());
        for (int l = 0; l < L.n_lms(); ++l) {
            const Eigen::LDLT<Mat3> ld(s.Hll[size_t(l)]);
            if (ld.info() != Eigen::Success || !(ld.vectorD().array() > 0.0).all()) {
                failed_landmark_ = l;
                return false;
            }
            Hll_inv_[size_t(l)] = ld.solve(Mat3::Identity());
        }
        S_ = s.Hcc;
        for (int l = 0; l < L.n_lms(); ++l) {
            const auto& cams = s.lm_cams[size_t(l)];
            for (int a : cams) {
                const Eigen::Matrix<double, 6, 3> Ba = s.Hcl.at({a, l}) * Hll_inv_[size_t(l)];
                for (int b : cams) S_.block<6, 6>(6 * a, 6 * b) -= Ba * s.Hcl.at({b, l}).transpose();
            }
        }
        if (L.dim_c() > 0) {
            ldlt_.compute(S_);
            if (ldlt_.info() != Eigen::Success || !(ldlt_.vectorD().array() > 0.0).all()) return false;
        }
        ok_ = true;
        return true;
    }

    Eigen::VectorXd Solve(const BaSystem& s, const BaLayout& L, const Eigen::VectorXd& right) const {
        if (!ok_) throw std::logic_error("BaSchur::Solve without a successful factorization");
        const int nc = L.dim_c();
        Eigen::VectorXd b_c = -right.head(nc);
        // b_c -= sum_l Hcl Hll^-1 (-r_l)  ==  rhs of the reduced system for -right
        for (int l = 0; l < L.n_lms(); ++l) {
            const Vec3 bl = -right.segment<3>(nc + 3 * l);
            const Vec3 y = Hll_inv_[size_t(l)] * bl;
            for (int a : s.lm_cams[size_t(l)]) b_c.segment<6>(6 * a) -= s.Hcl.at({a, l}) * y;
        }
        Eigen::VectorXd x(L.dim());
        if (nc > 0) x.head(nc) = ldlt_.solve(b_c);
        for (int l = 0; l < L.n_lms(); ++l) {
            Vec3 bl = -right.segment<3>(nc + 3 * l);
            for (int a : s.lm_cams[size_t(l)]) bl -= s.Hcl.at({a, l}).transpose() * x.segment<6>(6 * a);
            x.segment<3>(nc + 3 * l) = Hll_inv_[size_t(l)] * bl;
        }
        return x;
    }

    /// Marginal covariance (left tangent of T_W_C) of free camera c at the solution: the 6x6 block of S^-1, from
    /// S X = E (never an explicit inverse, §11.6). Call on an UNDAMPED factorization.
    Mat6 CameraCovariance(int c) const { return Marginal<6>(6 * c); }

    /// The N x N block of S^-1 at parameter offset `off` (inside the reduced system: a pose or an inertial block),
    /// from S X = E. Call on an UNDAMPED factorization.
    template <int N>
    Eigen::Matrix<double, N, N> Marginal(int off) const {
        if (!ok_) throw std::logic_error("BaSchur::Marginal without a factorization");
        Eigen::MatrixXd E = Eigen::MatrixXd::Zero(S_.rows(), N);
        E.block<N, N>(off, 0) = Eigen::Matrix<double, N, N>::Identity();
        const Eigen::Matrix<double, N, N> C = ldlt_.solve(E).template block<N, N>(off, 0);
        return 0.5 * (C + C.transpose());
    }

    const Eigen::MatrixXd& S() const { return S_; }
    /// After a failed Factorize: the layout index of the landmark block that was not PD, or -1 (the reduced camera
    /// system was not PD).
    int failed_landmark() const { return failed_landmark_; }

private:
    bool ok_ = false;
    int failed_landmark_ = -1;
    std::vector<Mat3> Hll_inv_;
    Eigen::MatrixXd S_;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
};

/// Apply a full step: cameras by the left exp, landmarks additively.
inline BaWindow BaOplus(const BaWindow& w, const BaLayout& L, const Eigen::VectorXd& x) {
    if (x.size() != L.dim()) throw std::invalid_argument("BaOplus: wrong step size");
    BaWindow out = w;
    for (size_t k = 0; k < w.T_W_C.size(); ++k) {
        const int c = L.cam(int(k));
        if (c < 0) continue;
        out.T_W_C[k] = core::SE3Exp(x.segment<6>(6 * c)) * w.T_W_C[k];
        out.T_W_C[k].normalize();
    }
    for (size_t k = 0; k < out.imu.v.size() && L.n_inertial() > 0; ++k) {
        if (L.vel(int(k)) >= 0) out.imu.v[k] += x.segment<3>(L.vel(int(k)));
        out.imu.bg[k] += x.segment<3>(L.bias(int(k)));
        out.imu.ba[k] += x.segment<3>(L.bias(int(k)) + 3);
    }
    for (auto& [id, X] : out.X)
        if (L.lm(id) >= 0) X += x.segment<3>(L.dim_c() + 3 * L.lm(id));
    return out;
}

}  // namespace ariadne::slam
