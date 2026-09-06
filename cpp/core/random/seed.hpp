#pragma once
// ---------------------------------------------------------------------------
// Deterministic seed plumbing (Phase 0.5).
//
// THE RULE: there is no global RNG anywhere in Ariadne. Every stochastic
// subsystem derives its own seed from the run's single root seed plus a stable
// purpose identifier, and draws from a counter-based generator keyed on it.
//
// Why derivation rather than "seed everything with the run seed": if two
// subsystems shared a seed they would draw the SAME numbers, which correlates
// noise that should be independent. Why a stable purpose id rather than an
// incrementing counter: adding a new subsystem must not shift the streams of
// existing ones, or every ablation baseline silently changes the moment an
// unrelated feature lands. This is the same argument that gives each stage of
// the degradation operator its own stream.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <string>

namespace ariadne::core {

/// Stable 64-bit FNV-1a over a purpose name. Chosen because the mapping from
/// name to seed must never change: a purpose id is effectively part of the
/// experiment's identity once results exist.
inline constexpr uint64_t HashPurpose(const char* name) {
    uint64_t hash = 1469598103934665603ull;
    while (*name != '\0') {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(*name));
        hash *= 1099511628211ull;
        ++name;
    }
    return hash;
}

/// Purposes that draw randomness. Names, not indices, so inserting a new one
/// leaves every existing stream untouched.
namespace seed_purpose {
inline constexpr const char* kDegradation = "degradation";
inline constexpr const char* kTrajectoryNoise = "trajectory_noise";
inline constexpr const char* kCommsLoss = "comms_loss";
inline constexpr const char* kCommsLatency = "comms_latency";
inline constexpr const char* kWorldGeneration = "world_generation";
inline constexpr const char* kFalseClosureInjection = "false_closure_injection";
inline constexpr const char* kRansac = "ransac";
inline constexpr const char* kGaussianInit = "gaussian_init";
inline constexpr const char* kTaskAllocation = "task_allocation";
}  // namespace seed_purpose

/// Derive a subsystem seed. Mixing uses splitmix64's finalizer, which
/// decorrelates nearby root seeds: run seeds 1 and 2 must not produce
/// subsystem seeds that share structure.
inline constexpr uint64_t DeriveSeed(uint64_t root_seed, uint64_t purpose_hash) {
    uint64_t z = root_seed ^ purpose_hash;
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline uint64_t DeriveSeed(uint64_t root_seed, const char* purpose) {
    return DeriveSeed(root_seed, HashPurpose(purpose));
}

/// The run's seed set. Constructed once from the root seed recorded in the run
/// manifest (§28.2), then passed explicitly to whatever needs it. Passing it
/// explicitly is deliberate: a subsystem that reaches for a global would be
/// invisible to the manifest and therefore unreproducible.
class SeedSet {
public:
    explicit SeedSet(uint64_t root_seed) : root_(root_seed) {}

    uint64_t Root() const { return root_; }
    uint64_t For(const char* purpose) const { return DeriveSeed(root_, purpose); }

private:
    uint64_t root_;
};

}  // namespace ariadne::core
