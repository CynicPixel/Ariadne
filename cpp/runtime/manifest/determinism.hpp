#pragma once
// Fill a run manifest's determinism digests by MEASURING them on this host at
// run time (D21, D32). Copying the committed constants into the manifest
// instead would make "a result produced by drifted math is identifiable after
// the fact" an empty guarantee: the manifest would claim the recorded values
// whatever the machine actually computed.
#include "core/determinism/digests.hpp"
#include "perception/degradation/degradation_digest.hpp"
#include "runtime/manifest/run_manifest.hpp"

#include <string>

namespace ariadne::runtime {

inline void RecordDeterminismDigests(RunManifest& m) {
    m.det_math_digest = std::to_string(core::DetMathDigest());
    m.philox_digest = std::to_string(core::PhiloxDigest());
    m.degradation_digest = std::to_string(perception::DegradeDigest());
}

}  // namespace ariadne::runtime
