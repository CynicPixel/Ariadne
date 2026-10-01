#pragma once
// The read-time lens operator's determinism digest (D41): the sampling map (its
// raw IEEE bits, so the Brown-Conrady inverse is covered) and ApplyLens on a fixed
// synthetic RGB image, through a lens with every coefficient non-zero. Single
// implementation shared by tests, check_determinism.sh and the run manifest (see
// core/determinism/digests.hpp for why a digest must exist exactly once).
#include "core/determinism/digests.hpp"
#include "perception/degradation/lens.hpp"

#include <cstdint>

namespace ariadne::perception {

inline uint64_t LensDigest() {
    LensModel m;                          // rig_stereo_v1's lens scaled to 320 x 180, plus k3
    m.width = 320;
    m.height = 180;
    m.render_fx = m.render_fy = 160.0;
    m.render_cx = 159.5;
    m.render_cy = 89.5;
    m.fx = m.fy = 172.0;
    m.cx = 159.5;
    m.cy = 89.5;
    m.k1 = -0.05;
    m.k2 = 0.01;
    m.p1 = 0.0005;
    m.p2 = -0.0002;
    m.k3 = 0.001;
    const LensMap map = BuildLensMap(m);
    core::ImageU8 in(m.width, m.height, 3);
    for (int y = 0; y < m.height; ++y)
        for (int x = 0; x < m.width; ++x)
            for (int c = 0; c < 3; ++c)
                in.At(x, y, c) = static_cast<uint8_t>((x * 7 + y * 3 + c * 11 + ((x ^ y) & 31)) & 0xFF);
    core::Fnv1a64 h;
    for (size_t i = 0; i < map.src_u.size(); ++i) {
        h.AbsorbU64(core::DoubleToBits(map.src_u[i]));
        h.AbsorbU64(core::DoubleToBits(map.src_v[i]));
    }
    for (uint8_t v : ApplyLens(in, map).data) h.AbsorbByte(v);
    return h.value();
}

}  // namespace ariadne::perception
