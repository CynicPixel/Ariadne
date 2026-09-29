#pragma once
// The degradation operator's determinism digest: D applied to a fixed synthetic
// image with dust enabled, so the whole chain (Poisson and Gaussian samplers,
// PRNU, vignetting, the dust field) contributes. Single implementation shared
// by tests, check_determinism.sh and the run manifest (see
// core/determinism/digests.hpp for why a digest must exist exactly once).
#include "core/determinism/digests.hpp"
#include "perception/degradation/degradation.hpp"

#include <cstdint>

namespace ariadne::perception {

inline uint64_t DegradeDigest() {
    ImageU8 in(64, 64, 3);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            for (int c = 0; c < 3; ++c)
                in.At(x, y, c) = static_cast<uint8_t>((x * 3 + y * 5 + c * 7) & 0xFF);
    ImageF32 depth(64, 64);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) depth.At(x, y) = 0.5f + 0.1f * static_cast<float>(x);

    DegradationParams p;
    p.radiance_scale = 0.08;
    p.photon_scale = 8000.0;
    p.read_noise_electrons = 3.0;
    p.vignette_strength = 0.35;
    p.prnu_sigma = 0.02;
    p.analog_gain = 2.0;
    p.enable_dust = true;
    p.dust_beta = 0.25;
    p.dust_airlight = 0.55;
    p.dust_field_strength = 0.6;

    const ImageU8 out = Degrade(in, depth, p, 20260906ull, 7ull);
    core::Fnv1a64 h;
    for (uint8_t v : out.data) h.AbsorbByte(v);
    return h.value();
}

}  // namespace ariadne::perception
