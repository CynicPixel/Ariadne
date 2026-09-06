#pragma once
// ---------------------------------------------------------------------------
// Record / replay over the single serialization boundary (arch §5.9, §4.4).
//
// A record file is a length-delimited sequence of framed messages, written
// through the SAME WireCodec that the network uses. That is deliberate: a
// replayed run therefore exercises the identical encode/decode path and the
// identical byte accounting as a live run, so a bug in the wire format cannot
// hide behind a separate file format.
//
// Layout, repeated to end of file:
//     [uint32 LE frame_length][frame_length bytes of framed message]
//
// The outer length is required because the in-frame header records the
// UNCOMPRESSED payload size, which does not tell a reader how many bytes the
// possibly-compressed body occupies.
//
// This is not MCAP. §5.9 specifies MCAP as the container for simulator
// recordings, and that remains open work; this file is the deterministic
// substrate the Phase 0 gate needs, and MCAP will wrap the same framed
// messages when it lands.
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace ariadne::runtime {

using core::ByteLedger;
using core::SerializedMessage;

class RecordWriter {
public:
    explicit RecordWriter(const std::string& path)
        : stream_(path, std::ios::binary | std::ios::trunc) {}

    bool Ok() const { return stream_.good(); }

    /// Append one message and charge its bytes to the ledger. Writing and
    /// accounting happen together so a message cannot be recorded without
    /// being counted.
    bool Write(SerializedMessage& msg, ByteLedger* ledger) {
        const uint32_t length = static_cast<uint32_t>(msg.bytes.size());
        uint8_t header[4] = {
            static_cast<uint8_t>(length & 0xFF),
            static_cast<uint8_t>((length >> 8) & 0xFF),
            static_cast<uint8_t>((length >> 16) & 0xFF),
            static_cast<uint8_t>((length >> 24) & 0xFF)};
        stream_.write(reinterpret_cast<const char*>(header), 4);
        stream_.write(reinterpret_cast<const char*>(msg.bytes.data()), length);
        if (ledger != nullptr) ledger->Record(msg.record);
        return stream_.good();
    }

    void Close() { stream_.close(); }

private:
    std::ofstream stream_;
};

class RecordReader {
public:
    explicit RecordReader(const std::string& path)
        : stream_(path, std::ios::binary) {}

    bool Ok() const { return stream_.good(); }

    /// Read the next framed message. Returns false at clean end of file and on
    /// a truncated record; `truncated` distinguishes the two, because a partial
    /// trailing record is a corrupt bag, not an empty one.
    bool Next(std::vector<uint8_t>* frame, bool* truncated = nullptr) {
        if (truncated != nullptr) *truncated = false;
        uint8_t header[4];
        stream_.read(reinterpret_cast<char*>(header), 4);
        if (stream_.gcount() == 0) return false;              // clean EOF
        if (stream_.gcount() != 4) {
            if (truncated != nullptr) *truncated = true;
            return false;
        }
        const uint32_t length = static_cast<uint32_t>(header[0]) |
                                (static_cast<uint32_t>(header[1]) << 8) |
                                (static_cast<uint32_t>(header[2]) << 16) |
                                (static_cast<uint32_t>(header[3]) << 24);
        frame->resize(length);
        stream_.read(reinterpret_cast<char*>(frame->data()), length);
        if (static_cast<uint32_t>(stream_.gcount()) != length) {
            if (truncated != nullptr) *truncated = true;
            return false;
        }
        return true;
    }

private:
    std::ifstream stream_;
};

}  // namespace ariadne::runtime
