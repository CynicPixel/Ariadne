// ---------------------------------------------------------------------------
// ariadne_identity: report what software this process is actually running on.
//
//   ariadne_identity                   KEY=VALUE lines (OpenCV identity)
//   ariadne_identity --build-info      the raw cv::getBuildInformation() text
//   ariadne_identity --manifest PATH   write a run manifest with every field
//                                      that is measurable without a run
//
// The image build calls it once and bakes the output into
// /opt/ariadne/opencv_identity.env. The Phase 0 container gate then compares
// that baked value with the one a manifest records at run time: a mismatch
// means the loader resolved a different OpenCV than the image was built with
// (intended for the COOL arm of the benchmark, and a bug anywhere else).
//
// Provenance that only the launcher knows (git commit, image digest) arrives
// through ARIADNE_* environment variables. An unset variable stays UNRECORDED
// in the manifest instead of being guessed.
// ---------------------------------------------------------------------------
#include "runtime/manifest/build_identity.hpp"
#include "runtime/manifest/determinism.hpp"
#include "runtime/manifest/opencv_identity.hpp"
#include "runtime/manifest/run_manifest.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>

namespace {

using ariadne::runtime::RunManifest;

std::string EnvOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::string(v) : fallback;
}

std::string UtcNow() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

int WriteManifest(const char* path) {
    RunManifest m;
    m.created_utc = UtcNow();
    m.run_id = "identity-" + m.created_utc;
    m.git_commit = EnvOr("ARIADNE_GIT_COMMIT", ariadne::runtime::kUnknownField);
    m.git_dirty = EnvOr("ARIADNE_GIT_DIRTY", "false") == "true";
    m.image_digest = EnvOr("ARIADNE_IMAGE_DIGEST", ariadne::runtime::kUnknownField);
    ariadne::runtime::RecordBuildIdentity(m);
    ariadne::runtime::RecordOpenCvIdentity(m);
    ariadne::runtime::RecordDeterminismDigests(m);

    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "ariadne_identity: cannot write %s\n", path);
        return 1;
    }
    out << ariadne::runtime::ToYaml(m);
    return out.good() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        const auto id = ariadne::runtime::MeasureOpenCvIdentity();
        std::printf("OPENCV_VERSION=%s\n", id.version.c_str());
        std::printf("OPENCV_BUILD_HASH=%s\n", id.build_hash.c_str());
        std::printf("OPENCV_CPU_FEATURES=\"%s\"\n", id.cpu_features.c_str());
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--build-info") == 0) {
        std::fputs(ariadne::runtime::OpenCvBuildInformation().c_str(), stdout);
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--manifest") == 0) {
        return WriteManifest(argv[2]);
    }
    std::fprintf(stderr, "usage: %s [--build-info | --manifest PATH]\n", argv[0]);
    return 2;
}
