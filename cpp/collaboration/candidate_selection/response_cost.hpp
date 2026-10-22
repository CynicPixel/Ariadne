#pragma once
// ---------------------------------------------------------------------------
// Predicted wire bytes of fetching one keyframe's features (design SPECTRAL_3.4 §5, COLLAB_3.2 §7): the
// MATCH_REQUEST plus the MATCH_RESPONSE, first transmission, as the ledger will count them.
//
//   payload   exact: a response of n features x feature_bytes is encoded through the real codec (FlatBuffers size,
//             padding and envelope included) and its pre-compression size taken;
//   compress  the per-class ratio model of COLLAB_3.2 §7 (calibrated on runs disjoint from the gate runs; 1.0 =
//             uncalibrated, and then the prediction is an upper bound, stated); the codec keeps a compressed form only
//             when it is smaller, so transmitted = min(payload, ratio * payload);
//   framing   the codec's own formula (frame header, then fragment header + datagram overhead per fragment).
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace ariadne::collaboration {

struct ResponseCostModel {
    double response_ratio = 1.0;   // transmitted / payload for MATCH_RESPONSE; 1.0 = uncalibrated (an upper bound)
};

/// Per class: the byte gate is per class (COLLAB_3.2 §7), so the two halves are kept apart.
struct FetchCost {
    double request = 0, response = 0;
    double total() const { return request + response; }
};

class FetchCostPredictor {
public:
    FetchCostPredictor(const core::WireCodec& codec, ResponseCostModel m) : codec_(codec), m_(m) {}

    FetchCost Predict(uint32_t n_features, uint16_t feature_bytes) const {
        const auto key = std::make_pair(n_features, feature_bytes);
        const auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;
        core::FeatureResponseValue f;
        f.request_id = ~0ull;
        f.keyframe_id = ~0ull;
        f.keypoints.assign(size_t(n_features) * 2, 0.0f);
        f.descriptors.assign(size_t(n_features) * feature_bytes, 0);
        f.points3d.assign(size_t(n_features) * 3, 0.0f);
        f.camera.assign(5, 0.0f);
        // Envelope fields as on a real message: FlatBuffers omits zero-valued fields, so a sample encoded with seq 0,
        // t 0 and agent 0 comes out smaller than any real one (measured: requests 112 B predicted vs 135.6 B sent).
        const uint32_t a = 1;
        const core::Timestamp ts{1'000'000'000};
        const uint64_t seq = (1ull << 40) | 1ull;   // a reliable-class sequence (incarnation << 40 | n)
        const uint64_t payload = codec_.EncodeFeatureResponse(f, a, a, ts, seq).record.payload_bytes;
        const double transmitted = std::min(double(payload), std::ceil(m_.response_ratio * double(payload)));
        const double response = Framed(transmitted);
        core::MatchRequestValue r{~0ull, 0, ~0ull, 0};
        const double request = double(codec_.EncodeMatchRequest(r, a, a, ts, seq).record.wire_bytes);
        return cache_[key] = FetchCost{request, response};
    }

private:
    double Framed(double transmitted) const {
        const uint64_t framed = core::kFrameHeaderBytes + uint64_t(transmitted);
        return codec_.framing() == core::Framing::kWholeMessage
                   ? double(framed + core::kDatagramOverheadBytes * core::DatagramCount(framed))
                   : double(framed + (core::kDatagramOverheadBytes + core::kFragmentHeaderBytes) *
                                         core::FragmentCount(framed));
    }
    const core::WireCodec& codec_;
    ResponseCostModel m_;
    mutable std::map<std::pair<uint32_t, uint16_t>, FetchCost> cache_;
};

}  // namespace ariadne::collaboration
