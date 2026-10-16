#pragma once
// ---------------------------------------------------------------------------
// ATB ("Ariadne tensor blob"): the dependency-free file format the XFeat weights
// and reference outputs travel in (python/model_tools/xfeat_reference.py writes
// it). Little-endian:
//   "ATB1", u32 count, then per tensor: u16 name length, name, u8 dtype
//   (0 = f32, 1 = i64), u8 ndim, u32 dims[ndim], raw C-order data.
// Every malformed input is an exception, never a partial read.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::perception::xfeat {

struct Tensor {
    std::vector<int> shape;
    std::vector<float> f32;     // dtype 0
    std::vector<int64_t> i64;   // dtype 1
    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= size_t(d);
        return n;
    }
};

using TensorMap = std::map<std::string, Tensor>;

inline TensorMap LoadAtb(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("atb: cannot open " + path);
    const auto read = [&](void* p, size_t n) {
        if (!f.read(static_cast<char*>(p), std::streamsize(n))) throw std::runtime_error("atb: truncated " + path);
    };
    char magic[4];
    read(magic, 4);
    if (std::memcmp(magic, "ATB1", 4) != 0) throw std::runtime_error("atb: bad magic in " + path);
    uint32_t count = 0;
    read(&count, 4);
    TensorMap out;
    for (uint32_t t = 0; t < count; ++t) {
        uint16_t nl = 0;
        read(&nl, 2);
        std::string name(nl, '\0');
        read(name.data(), nl);
        uint8_t dtype = 0, ndim = 0;
        read(&dtype, 1);
        read(&ndim, 1);
        if (dtype > 1 || ndim > 8) throw std::runtime_error("atb: bad header for " + name);
        Tensor x;
        for (int d = 0; d < ndim; ++d) {
            uint32_t v = 0;
            read(&v, 4);
            x.shape.push_back(int(v));
        }
        if (dtype == 0) {
            x.f32.resize(x.numel());
            read(x.f32.data(), x.numel() * 4);
        } else {
            x.i64.resize(x.numel());
            read(x.i64.data(), x.numel() * 8);
        }
        if (!out.emplace(name, std::move(x)).second) throw std::runtime_error("atb: duplicate tensor " + name);
    }
    if (f.peek() != EOF) throw std::runtime_error("atb: trailing bytes in " + path);
    return out;
}

}  // namespace ariadne::perception::xfeat
