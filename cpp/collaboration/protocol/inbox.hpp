#pragma once
// ---------------------------------------------------------------------------
// The receive side of §15.3: "every message is idempotent; duplicates are
// ignored; out-of-order messages are stored until prerequisites arrive".
//
// Every message carries (sender, class, monotonic sequence), sequences starting
// at 0 per (sender, class). The inbox keeps a high-water mark per stream plus a
// bounded out-of-order buffer: a message at the mark is delivered (followed by
// any buffered successors), one below it or already buffered is a duplicate
// (acknowledged and dropped), one above it waits. Delivery order per stream is
// therefore the sender's order regardless of what the link did.
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ariadne::collaboration {

using core::DecodedMessage;
using core::MessageClass;

class Inbox {
public:
    explicit Inbox(size_t max_buffered_per_stream = 4096) : max_buffered_(max_buffered_per_stream) {}

    /// Offer one decoded message; returns those now deliverable, in order.
    std::vector<DecodedMessage> Offer(DecodedMessage m) {
        const Key k{m.sender, static_cast<uint8_t>(m.cls)};
        uint64_t& next = next_[k];
        auto& buf = pending_[k];
        std::vector<DecodedMessage> out;
        if (m.sequence < next || buf.count(m.sequence)) {
            ++duplicates_;
            return out;
        }
        if (m.sequence > next) {
            if (buf.size() >= max_buffered_) throw std::runtime_error("inbox: out-of-order buffer overflow");
            buf.emplace(m.sequence, std::move(m));
            ++buffered_;
            return out;
        }
        out.push_back(std::move(m));
        ++next;
        for (auto it = buf.find(next); it != buf.end(); it = buf.find(next)) {
            out.push_back(std::move(it->second));
            buf.erase(it);
            ++next;
        }
        delivered_ += out.size();
        return out;
    }

    uint64_t duplicates() const { return duplicates_; }
    uint64_t buffered() const { return buffered_; }     // arrivals that had to wait
    uint64_t delivered() const { return delivered_; }
    size_t waiting() const {
        size_t n = 0;
        for (const auto& [k, b] : pending_) n += b.size();
        return n;
    }

private:
    using Key = std::pair<uint32_t, uint8_t>;
    size_t max_buffered_;
    std::map<Key, uint64_t> next_;
    std::map<Key, std::map<uint64_t, DecodedMessage>> pending_;
    uint64_t duplicates_ = 0, buffered_ = 0, delivered_ = 0;
};

/// The send side: one monotonic counter per (receiver, class).
class SequenceCounter {
public:
    uint64_t Next(uint32_t receiver, MessageClass cls) { return next_[{receiver, static_cast<uint8_t>(cls)}]++; }

private:
    std::map<std::pair<uint32_t, uint8_t>, uint64_t> next_;
};

}  // namespace ariadne::collaboration
