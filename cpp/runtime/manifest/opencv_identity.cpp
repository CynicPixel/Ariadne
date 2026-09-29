#include "runtime/manifest/opencv_identity.hpp"

#include "core/determinism/digests.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <cstdio>

namespace ariadne::runtime {

std::string OpenCvBuildInformation() { return cv::getBuildInformation(); }

OpenCvIdentity MeasureOpenCvIdentity() {
    // FNV-1a rather than a cryptographic hash: this is an identity check
    // against accidental difference, not against an adversary, and it reuses
    // the one hash implementation the determinism digests already trust.
    core::Fnv1a64 h;
    for (unsigned char c : OpenCvBuildInformation()) h.AbsorbByte(c);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h.value()));

    OpenCvIdentity id;
    id.version = cv::getVersionString();
    // Prefixed with the algorithm so a later switch to another hash cannot be
    // mistaken for a different build.
    id.build_hash = std::string("fnv1a64:") + hex;
    id.cpu_features = cv::getCPUFeaturesLine();
    return id;
}

void RecordOpenCvIdentity(RunManifest& m) {
    const OpenCvIdentity id = MeasureOpenCvIdentity();
    m.opencv_version = id.version;
    m.opencv_build_hash = id.build_hash;
    m.opencv_cpu_features = id.cpu_features;
}

}  // namespace ariadne::runtime
