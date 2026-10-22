// SMOKE §4u (4.1, arch §20.1): OpenCV 5.0.0 cv::Volume(HashTSDF) as the volumetric layer or as its oracle.
//
// What it checks, on a synthetic scene with a known answer (a fronto-parallel wall at Z = 2 m, a 640x360 pinhole):
//   1. integrate(float depth in metres, cam->world pose) at 0.05 m and 0.08 m voxels, raycast from the same pose,
//      report the median and max |raycast Z - 2.0| over valid pixels;
//   2. fetchPointsNormals: the surface point count and the mean |Z - 2.0|.
// The public API's shape (integrate / raycast / fetch / bbox; no voxel or weight read, no per-pixel weight input) is
// read from the installed header, not from this program.
//
//   docker run --rm -v "$PWD/infra/scripts:/w" -w /w ariadne/core-dev:dev sh -c '
//     g++ -std=c++17 -O2 smoke_hashtsdf.cpp -o /tmp/s -I/opt/opencv5/include/opencv5 -L/opt/opencv5/lib \
//       -lopencv_core -lopencv_ptcloud && LD_LIBRARY_PATH=/opt/opencv5/lib /tmp/s'
#include <opencv2/core.hpp>
#include <opencv2/ptcloud/volume.hpp>
#include <opencv2/ptcloud/volume_settings.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int W = 640, H = 360;
    const float fx = 340.f, cx = 319.5f, cy = 179.5f, wall = 2.0f;
    const cv::Matx33f K(fx, 0, cx, 0, fx, cy, 0, 0, 1);
    cv::Mat depth(H, W, CV_32F, cv::Scalar(wall));
    const cv::Matx44f pose = cv::Matx44f::eye();   // camera at the world origin, looking down +Z
    for (float vox : {0.05f, 0.08f}) {
        cv::VolumeSettings s(cv::VolumeType::HashTSDF);
        s.setVoxelSize(vox);
        s.setTsdfTruncateDistance(4 * vox);
        s.setDepthFactor(1.f);   // depth is already metres
        s.setMaxDepth(6.f);
        s.setIntegrateWidth(W), s.setIntegrateHeight(H), s.setRaycastWidth(W), s.setRaycastHeight(H);
        s.setCameraIntegrateIntrinsics(cv::Mat(K)), s.setCameraRaycastIntrinsics(cv::Mat(K));
        cv::Volume v(cv::VolumeType::HashTSDF, s);
        for (int i = 0; i < 5; ++i) v.integrate(depth, pose);
        cv::Mat pts, nrm;
        v.raycast(pose, pts, nrm);
        std::vector<float> err;
        for (int r = 0; r < pts.rows; ++r) {
            for (int c = 0; c < pts.cols; ++c) {
                const cv::Vec4f p = pts.at<cv::Vec4f>(r, c);
                if (std::isfinite(p[2]) && p[2] > 0) err.push_back(std::abs(p[2] - wall));
            }
        }
        std::sort(err.begin(), err.end());
        cv::Mat fp, fn;
        v.fetchPointsNormals(fp, fn);
        double zsum = 0;
        for (int i = 0; i < fp.rows; ++i) zsum += std::abs(fp.at<cv::Vec4f>(i)[2] - wall);
        std::printf("voxel %.2f m: raycast pts type %d (%dx%d), valid %zu/%d, |dZ| median %.4f max %.4f m; "
                    "fetchPointsNormals %d pts, mean |dZ| %.4f m; volume units %zu\n",
                    vox, pts.type(), pts.cols, pts.rows, err.size(), W * H, err.empty() ? -1.f : err[err.size() / 2],
                    err.empty() ? -1.f : err.back(), fp.rows, fp.rows ? zsum / fp.rows : -1.0, v.getTotalVolumeUnits());
    }
    return 0;
}
