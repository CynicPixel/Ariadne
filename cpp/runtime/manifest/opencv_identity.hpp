#pragma once
// ---------------------------------------------------------------------------
// OpenCV build identity for the run manifest (arch §27.1, C7).
//
// WHAT IS MEASURED. cv::getBuildInformation() returns the configuration text
// that the LOADED libopencv_core carries: version, module list, SIMD baseline
// and dispatch set, compiler, flags. Its hash identifies the OpenCV build that
// actually ran. "Loaded", not "linked": the COOL benchmark (Invariant 2) runs
// one binary against two OpenCVs by changing only what the dynamic loader
// resolves, so a value taken at compile time would record stock OpenCV for
// both arms. This is always read at run time, from the library in memory.
//
// WHAT THE HASH IS NOT. The text contains a build `Timestamp:` and `Host:`
// line, so the hash changes on every rebuild of the same tag, and it differs
// between amd64 and arm64. It identifies a build artifact, never "OpenCV
// 5.0.0 with these flags". Never commit a golden value of it (D33).
//
// CPU FEATURES. OpenCV compiles several SIMD variants of hot kernels (the
// "dispatched" set: SSE4/AVX2/AVX512 on amd64) and picks one per call from the
// CPU it runs on. The same build can therefore produce different last bits on
// two machines. cv::getCPUFeaturesLine() records what this machine offered, so
// a cross-machine difference in an OpenCV output can be attributed rather than
// chased.
//
// Separate from run_manifest.hpp so the manifest itself stays OpenCV-free and
// the OpenCV-free layers keep building on a host without OpenCV.
// ---------------------------------------------------------------------------
#include "runtime/manifest/run_manifest.hpp"

#include <string>

namespace ariadne::runtime {

struct OpenCvIdentity {
    std::string version;       // cv::getVersionString(), e.g. "5.0.0"
    std::string build_hash;    // "fnv1a64:<16 hex digits>" of the build text
    std::string cpu_features;  // cv::getCPUFeaturesLine()
};

/// Measure the identity of the OpenCV that is loaded in this process.
OpenCvIdentity MeasureOpenCvIdentity();

/// The raw text the hash is computed over, for dumping and debugging.
std::string OpenCvBuildInformation();

/// Fill the manifest's OpenCV fields by measurement.
void RecordOpenCvIdentity(RunManifest& m);

}  // namespace ariadne::runtime
