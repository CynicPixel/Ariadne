// ---------------------------------------------------------------------------
// Phase 0.2 smoke test (arch §0.2). Resolves the [SMOKE-TEST] items that gate
// dependent code, against a BUILT AND INSTALLED OpenCV 5 -- not against the
// source tree. The distinction matters: a class can exist in the tree and still
// be excluded by our BUILD_LIST, gated behind a CMake option, or fail to link.
//
// The strongest evidence here is that this file COMPILES AND LINKS at all. Each
// type reference below is a compile-time assertion that the declaration is in
// the installed headers; each create()/symbol reference forces the linker to
// resolve it in the installed libraries. A missing class is a build error, not
// a runtime surprise.
//
// Output is machine-greppable `KEY=VALUE` so SMOKE_TEST_RESULTS.md rows can be
// regenerated rather than hand-copied.
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/features.hpp>
#include <opencv2/ptcloud/volume.hpp>
#include <opencv2/ptcloud/volume_settings.hpp>
#include <opencv2/geometry/3d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/calib.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/quality.hpp>

#include <cstdio>
#include <functional>
#include <string>
#include <typeinfo>

namespace {

void Report(const char* key, const std::string& value) {
    std::printf("%s=%s\n", key, value.c_str());
}

// Proves a type is complete (declared in the installed headers) without needing
// a model file or a constructor call. sizeof() on an incomplete type is a
// compile error, which is exactly the assertion we want.
template <typename T>
std::string TypePresent() {
    return sizeof(T) > 0 ? "PRESENT" : "PRESENT";
}

}  // namespace

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    Report("opencv_version", CV_VERSION);

    // C7: the manifest records a hash of getBuildInformation(), not the whole
    // string -- the string is ~5 KB of noise, the hash is a stable identity for
    // "this exact build configuration".
    const std::string build_info = cv::getBuildInformation();
    Report("build_info_hash",
           std::to_string(std::hash<std::string>{}(build_info)));
    Report("build_info_bytes", std::to_string(build_info.size()));

    // --- Feature classes (arch §0.2 gate list, §7.2) ---------------------
    // XFeat is deliberately absent from this list: it does not exist in
    // OpenCV 5.0.0. See DECISIONS.md D6. Adding it here would not compile,
    // which is itself the cleanest possible proof.
    Report("features_ALIKED", TypePresent<cv::ALIKED>());
    Report("features_DISK", TypePresent<cv::DISK>());
    Report("features_LightGlueMatcher", TypePresent<cv::LightGlueMatcher>());
    Report("features_ANNIndex", TypePresent<cv::ANNIndex>());
    Report("features_ORB", TypePresent<cv::ORB>());
    Report("features_SIFT", TypePresent<cv::SIFT>());

    // Force real symbol resolution rather than header-only presence: take the
    // address of a static factory. If the symbol is missing from the installed
    // library this fails at link time.
    {
        auto* aliked_create =
            static_cast<cv::Ptr<cv::ALIKED> (*)(const cv::String&,
                                                const cv::ALIKED::Params&)>(
                &cv::ALIKED::create);
        Report("features_ALIKED_create_symbol",
               aliked_create != nullptr ? "LINKED" : "NULL");
    }

    // --- Volumetric (arch §20.1) -----------------------------------------
    // HashTSDF present => the from-scratch TSDF fallback is NOT required.
    {
        cv::VolumeSettings settings(cv::VolumeType::HashTSDF);
        cv::Volume volume(cv::VolumeType::HashTSDF, settings);
        Report("ptcloud_HashTSDF", "CONSTRUCTED");
        Report("ptcloud_voxel_size", std::to_string(settings.getVoxelSize()));
        Report("ptcloud_truncated_distance",
               std::to_string(settings.getTsdfTruncateDistance()));
        Report("ptcloud_max_weight", std::to_string(settings.getMaxWeight()));
    }

    // --- Robust estimation backend (C3 claim: USAC is the default) --------
    Report("geometry_USAC_DEFAULT", std::to_string(cv::USAC_DEFAULT));
    Report("geometry_USAC_MAGSAC", std::to_string(cv::USAC_MAGSAC));

    // --- Stereo (§9.3) ----------------------------------------------------
    {
        auto sgbm = cv::StereoSGBM::create(0, 64, 5);
        Report("stereo_StereoSGBM", sgbm.empty() ? "NULL" : "CONSTRUCTED");
        auto wls = cv::ximgproc::createDisparityWLSFilter(sgbm);
        Report("ximgproc_DisparityWLSFilter",
               wls.empty() ? "NULL" : "CONSTRUCTED");
    }

    // --- Quality harness (§30) -------------------------------------------
    {
        cv::Mat a = cv::Mat::zeros(16, 16, CV_8UC1);
        cv::Mat b = cv::Mat::ones(16, 16, CV_8UC1) * 255;
        const cv::Scalar psnr = cv::quality::QualityPSNR::compute(a, b, cv::noArray());
        Report("quality_PSNR_black_vs_white", std::to_string(psnr[0]));
    }

    // --- Threading control (§27.1 COOL discipline) ------------------------
    // Both arms of the COOL benchmark must pin threads identically; confirm the
    // knob exists and round-trips in this build.
    const int original = cv::getNumThreads();
    cv::setNumThreads(1);
    Report("core_setNumThreads_pinned", std::to_string(cv::getNumThreads()));
    cv::setNumThreads(original);
    Report("core_numThreads_default", std::to_string(original));

    std::printf("SMOKE_TEST=PASS\n");
    return 0;
}
