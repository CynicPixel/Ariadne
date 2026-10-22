#include "perception/radiometric/falloff.hpp"

#include "core/config/config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ariadne::perception::radiometric {

double LampCalibration::Rid(double theta_deg) const {
    const auto& t = rid_theta_deg;
    const auto& v = rid_value;
    if (t.empty() || theta_deg > t.back()) return 0.0;
    if (theta_deg <= t.front()) return v.front();
    const auto it = std::upper_bound(t.begin(), t.end(), theta_deg);
    const size_t i = static_cast<size_t>(it - t.begin());
    const double w = (theta_deg - t[i - 1]) / (t[i] - t[i - 1]);
    return v[i - 1] + w * (v[i] - v[i - 1]);
}

LampCalibration LoadLampCalibration(const std::string& path) {
    auto s = core::StrictConfig::FromFile(path);
    LampCalibration c;
    const auto fail = [&path](const std::string& m) { throw core::ConfigError(path + ": " + m); };
    c.name = s.Require<std::string>("name");
    c.rig = s.Require<std::string>("rig");
    c.render_transfer = s.Require<std::string>("render_transfer");
    const auto p = s.Require<std::vector<double>>("lamp_in_camera_optical_m");
    const auto a = s.Require<std::vector<double>>("lamp_axis_optical");
    if (p.size() != 3 || a.size() != 3) fail("lamp_in_camera_optical_m and lamp_axis_optical are 3-vectors");
    c.lamp_in_camera = {p[0], p[1], p[2]};
    c.axis = cv::normalize(cv::Vec3d(a[0], a[1], a[2]));
    c.intensity_ref = s.Require<double>("intensity_ref");
    c.calib_albedo = s.Require<double>("calib_albedo");
    c.gain_A = s.Require<double>("gain_A");
    c.c_prime_m2 = s.Require<double>("c_prime_m2");
    c.cos_incidence = s.Require<bool>("cos_incidence");
    auto rid = s.Section("rid");
    c.rid_theta_deg = rid.Require<std::vector<double>>("theta_deg");
    c.rid_value = rid.Require<std::vector<double>>("value");
    rid.RequireAllConsumed();
    s.Section("fit");   // provenance of the fit; read by people, not by the stage
    s.RequireAllConsumed();
    if (c.rid_theta_deg.size() < 2 || c.rid_theta_deg.size() != c.rid_value.size() ||
        !std::is_sorted(c.rid_theta_deg.begin(), c.rid_theta_deg.end())) {
        fail("rid: >= 2 entries, theta_deg increasing, one value per theta");
    }
    if (!(c.gain_A > 0.0) || c.c_prime_m2 < 0.0 || !(c.intensity_ref > 0.0) || !(c.calib_albedo > 0.0)) {
        fail("gain_A > 0, c_prime_m2 >= 0, intensity_ref > 0, calib_albedo > 0");
    }
    return c;
}

AlbedoProxy FalloffNormalize(const cv::Mat& gray, const cv::Mat& z, const cv::Matx33d& K, const LampCalibration& lamp,
                             double lamp_rel, const FalloffParams& p, const cv::Matx33d& R_cam_from_raw) {
    if (gray.channels() != 1 || z.type() != CV_32F || gray.size() != z.size()) {
        throw std::invalid_argument("FalloffNormalize: single-channel gray and CV_32F depth of the same size");
    }
    cv::Mat g;
    if (gray.depth() == CV_8U) gray.convertTo(g, CV_32F, 1.0 / 255.0);
    else if (gray.depth() == CV_32F) g = gray;
    else throw std::invalid_argument("FalloffNormalize: gray must be CV_8U or CV_32F");
    if (!(lamp_rel > 0.0)) throw std::invalid_argument("FalloffNormalize: lamp_rel > 0");

    const int h = z.rows, w = z.cols;
    const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    const cv::Vec3d pl = R_cam_from_raw * lamp.lamp_in_camera;
    const cv::Vec3d ax = R_cam_from_raw * lamp.axis;
    const double scale = lamp.gain_A * lamp_rel;

    const auto zat = [&z](int x, int y) {
        const float v = z.at<float>(y, x);
        return std::isfinite(v) && v > 0.0f ? static_cast<double>(v) : 0.0;
    };
    const auto point = [&](int x, int y, double zz) { return cv::Vec3d((x - cx) / fx * zz, (y - cy) / fy * zz, zz); };

    AlbedoProxy out;
    cv::Mat e(h, w, CV_32F, cv::Scalar(0));
    out.valid = cv::Mat(h, w, CV_8U, cv::Scalar(0));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double zz = zat(x, y);
            if (zz == 0.0) continue;
            const cv::Vec3d X = point(x, y, zz);
            const cv::Vec3d L = X - pl;
            const double r = cv::norm(L);
            const double theta = std::acos(std::clamp(L.dot(ax) / r, -1.0, 1.0)) * (180.0 / M_PI);
            const double rid = lamp.Rid(theta);
            if (rid < p.min_rid) continue;
            double ci = 1.0;
            if (lamp.cos_incidence) {
                // Surface normal from the depth map: cross product of central differences of the 3D points.
                if (x == 0 || y == 0 || x == w - 1 || y == h - 1) continue;
                const double zl = zat(x - 1, y), zr = zat(x + 1, y), zu = zat(x, y - 1), zd = zat(x, y + 1);
                if (zl == 0.0 || zr == 0.0 || zu == 0.0 || zd == 0.0) continue;
                cv::Vec3d n = (point(x + 1, y, zr) - point(x - 1, y, zl)).cross(point(x, y + 1, zd) - point(x, y - 1, zu));
                const double nn = cv::norm(n);
                if (!(nn > 0.0)) continue;
                n /= nn;
                if (n.dot(X) > 0.0) n = -n;                  // face the camera
                ci = n.dot(-L) / r;                           // light arrives along -L
                if (ci < p.min_cos) continue;
            }
            e.at<float>(y, x) = static_cast<float>(scale * rid * ci / (lamp.c_prime_m2 + r * r));
            out.valid.at<uint8_t>(y, x) = 255;
        }
    }
    cv::minMaxLoc(e, nullptr, &out.e_max, nullptr, nullptr, out.valid);
    const double floor_e = out.e_max / p.max_gain;
    out.albedo = cv::Mat(h, w, CV_32F, cv::Scalar(0));
    for (int y = 0; y < h; ++y) {
        const uint8_t* v = out.valid.ptr<uint8_t>(y);
        const float* ee = e.ptr<float>(y);
        const float* gg = g.ptr<float>(y);
        float* a = out.albedo.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            if (!v[x]) continue;
            a[x] = static_cast<float>(gg[x] * lamp.calib_albedo / std::max<double>(ee[x], floor_e));
        }
    }
    return out;
}

}  // namespace ariadne::perception::radiometric
