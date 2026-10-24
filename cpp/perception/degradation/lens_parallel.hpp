#pragma once
// ---------------------------------------------------------------------------
// D226 (shape fixed after the Kaggle test_unit break): ApplyLens with its rows under cv::parallel_for_. Lives in
// its own header so lens.hpp - the determinism substrate, on the manifest/determinism include path - stays
// OpenCV-free (arch §3 dependency rule; ariadne_perception is deliberately OpenCV-less). Every output pixel is an
// independent pure function of the inputs, so any row partition is BIT-IDENTICAL to the serial ApplyLens
// (verified: ma_smoke3 replay trace sha unchanged at any thread count). Include this only from OpenCV-linked
// consumers (stereo_source, sim tools); everything else keeps ApplyLens.
// ---------------------------------------------------------------------------
#include "perception/degradation/lens.hpp"

#include <opencv2/core.hpp>

#include <stdexcept>

namespace ariadne::perception {

inline core::ImageU8 ApplyLensParallel(const core::ImageU8& clean, const LensMap& map) {
    if (clean.width != map.width || clean.height != map.height) {
        throw std::invalid_argument("lens: image size does not match the map");
    }
    core::ImageU8 out(map.width, map.height, clean.channels);
    cv::parallel_for_(cv::Range(0, map.height), [&](const cv::Range& rows) {
        ApplyLensRowSpan(clean, map, rows.start, rows.end, out);
    });
    return out;
}

}  // namespace ariadne::perception
