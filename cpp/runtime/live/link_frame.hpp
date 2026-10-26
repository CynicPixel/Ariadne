#pragma once
// ---------------------------------------------------------------------------
// The live link's wire format (design live-runtime.md D150): one Unix stream
// socket between the ROS edge (ros2/ariadne_bridge RuntimeLink, sim image) and the
// ROS-free runtime (ariadne_live, core image). std + POSIX only, so both images
// compile the SAME definition.
//
//   frame = u32 magic "ARL1" | u32 topic_len | topic | u32 payload_len | payload
//   (little-endian; payload = the ROS message's CDR bytes, encapsulation header
//    included, exactly as a rosbag2 MCAP stores it, so ros_cdr decodes it.)
//
// Topics starting with '#' are control frames, not ROS messages:
//   #ready  edge -> runtime: every bridged topic has a publisher, every command
//           topic a subscriber; payload = JSON listing them. The world is
//           unpaused only after this (infra/scripts/live.sh).
//   #stats  edge -> runtime: JSON per-topic counters (received, forwarded).
// ---------------------------------------------------------------------------
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ariadne::runtime::live {

constexpr uint32_t kLinkMagic = 0x314C5241u;   // bytes "ARL1"
constexpr uint32_t kMaxTopic = 256;
constexpr uint32_t kMaxPayload = 64u << 20;    // a 1280x720 rgb8 image is 2.8 MB

struct LinkFrame {
    std::string topic;
    std::vector<uint8_t> payload;
};

namespace detail {

inline bool WriteAll(int fd, const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    while (n > 0) {
        const ssize_t k = ::send(fd, b, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        b += k;
        n -= size_t(k);
    }
    return true;
}

/// false = clean EOF before the first byte; throws on EOF mid-object.
inline bool ReadAll(int fd, void* p, size_t n, bool eof_ok) {
    auto* b = static_cast<uint8_t*>(p);
    size_t got = 0;
    while (got < n) {
        const ssize_t k = ::recv(fd, b + got, n - got, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k < 0) throw std::runtime_error(std::string("link: recv: ") + std::strerror(errno));
        if (k == 0) {
            if (got == 0 && eof_ok) return false;
            throw std::runtime_error("link: connection closed mid-frame");
        }
        got += size_t(k);
    }
    return true;
}

inline uint32_t Le32(const uint8_t* b) { return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24; }
inline void PutLe32(uint8_t* b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b[i] = uint8_t(v >> (8 * i));
}

}  // namespace detail

/// One frame; false if the peer has gone (EPIPE). Callers serialize concurrent writers.
inline bool WriteFrame(int fd, const std::string& topic, const uint8_t* data, size_t n) {
    if (topic.size() > kMaxTopic || n > kMaxPayload) throw std::length_error("link: frame too large");
    uint8_t h[8];
    detail::PutLe32(h, kLinkMagic);
    detail::PutLe32(h + 4, uint32_t(topic.size()));
    uint8_t l[4];
    detail::PutLe32(l, uint32_t(n));
    return detail::WriteAll(fd, h, 8) && detail::WriteAll(fd, topic.data(), topic.size()) &&
           detail::WriteAll(fd, l, 4) && (n == 0 || detail::WriteAll(fd, data, n));
}

/// false = clean EOF between frames. Throws on a malformed frame (wrong magic, oversize).
inline bool ReadFrame(int fd, LinkFrame& f) {
    uint8_t h[8];
    if (!detail::ReadAll(fd, h, 8, true)) return false;
    if (detail::Le32(h) != kLinkMagic) throw std::runtime_error("link: bad magic");
    const uint32_t tl = detail::Le32(h + 4);
    if (tl > kMaxTopic) throw std::runtime_error("link: topic length " + std::to_string(tl));
    f.topic.resize(tl);
    detail::ReadAll(fd, f.topic.data(), tl, false);
    uint8_t l[4];
    detail::ReadAll(fd, l, 4, false);
    const uint32_t n = detail::Le32(l);
    if (n > kMaxPayload) throw std::runtime_error("link: payload length " + std::to_string(n));
    f.payload.resize(n);
    if (n) detail::ReadAll(fd, f.payload.data(), n, false);
    return true;
}

inline sockaddr_un UnixAddress(const std::string& path) {
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    if (path.size() + 1 > sizeof a.sun_path) throw std::length_error("link: socket path too long: " + path);
    std::memcpy(a.sun_path, path.c_str(), path.size() + 1);
    return a;
}

/// Server side (the runtime): bind + listen, accept ONE peer. Removes a stale socket file first.
inline int ListenAndAccept(const std::string& path) {
    const int s = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) throw std::runtime_error("link: socket");
    ::unlink(path.c_str());
    const sockaddr_un a = UnixAddress(path);
    if (::bind(s, reinterpret_cast<const sockaddr*>(&a), sizeof a) != 0 || ::listen(s, 1) != 0) {
        ::close(s);
        throw std::runtime_error("link: bind/listen " + path + ": " + std::strerror(errno));
    }
    const int c = ::accept(s, nullptr, nullptr);
    ::close(s);
    if (c < 0) throw std::runtime_error("link: accept");
    return c;
}

/// Client side (the ROS edge): connect, retrying until timeout (the runtime may start second).
inline int Connect(const std::string& path, std::chrono::milliseconds timeout) {
    const auto until = std::chrono::steady_clock::now() + timeout;
    const sockaddr_un a = UnixAddress(path);
    for (;;) {
        const int s = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (s < 0) throw std::runtime_error("link: socket");
        if (::connect(s, reinterpret_cast<const sockaddr*>(&a), sizeof a) == 0) return s;
        ::close(s);
        if (std::chrono::steady_clock::now() > until) throw std::runtime_error("link: cannot connect to " + path);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

}  // namespace ariadne::runtime::live
