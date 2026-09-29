#pragma once
// Compile-time facts about THIS binary, for the run manifest (§28.2). They
// describe the executable, so the compiler knows them; one definition shared
// by every tool that writes a manifest.
#include "runtime/manifest/run_manifest.hpp"

#include <string>

namespace ariadne::runtime {

inline std::string CompilerId() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#else
    return kUnknownField;
#endif
}

inline std::string ArchitectureId() {
#if defined(__aarch64__)
    return "arm64";
#elif defined(__x86_64__)
    return "amd64";
#else
    return kUnknownField;
#endif
}

inline void RecordBuildIdentity(RunManifest& m) {
    m.compiler = CompilerId();
    m.architecture = ArchitectureId();
}

}  // namespace ariadne::runtime
