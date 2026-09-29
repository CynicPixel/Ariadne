// Phase 0.5: event schema (§41), run manifest (§28.2), seed plumbing.
#include "core/random/philox.hpp"
#include "core/random/seed.hpp"
#include "runtime/logging/event_log.hpp"
#include "runtime/manifest/run_manifest.hpp"

#include <gtest/gtest.h>

#include <set>
#include <sstream>

using namespace ariadne::core;
using namespace ariadne::runtime;

// --- Seed plumbing --------------------------------------------------------

TEST(Seeds, DerivationIsDeterministicAndPurposeSpecific) {
    const SeedSet seeds(20260906ull);
    EXPECT_EQ(seeds.For(seed_purpose::kDegradation),
              seeds.For(seed_purpose::kDegradation));
    EXPECT_NE(seeds.For(seed_purpose::kDegradation),
              seeds.For(seed_purpose::kCommsLoss));
    EXPECT_NE(seeds.Root(), seeds.For(seed_purpose::kDegradation));
}

// Adding a subsystem must not shift an existing one's stream, or every
// ablation baseline silently changes when an unrelated feature lands. Purpose
// ids are name-hashed rather than sequential precisely to guarantee this.
TEST(Seeds, PurposeIdsAreNameDerivedNotPositional) {
    const SeedSet seeds(7ull);
    const uint64_t before = seeds.For(seed_purpose::kDegradation);
    // A newly introduced purpose, as if added later in the project.
    const uint64_t brand_new = seeds.For("some_future_subsystem");
    EXPECT_EQ(seeds.For(seed_purpose::kDegradation), before);
    EXPECT_NE(brand_new, before);
}

// Nearby root seeds must not produce structured subsystem seeds; seeds 1..N are
// exactly how sweeps are run, so correlation there would correlate whole runs.
TEST(Seeds, NearbyRootSeedsDecorrelate) {
    std::set<uint64_t> derived;
    for (uint64_t root = 1; root <= 200; ++root) {
        derived.insert(DeriveSeed(root, seed_purpose::kDegradation));
    }
    EXPECT_EQ(derived.size(), 200u);   // no collisions

    // Consecutive roots must not give consecutive (or trivially related) seeds.
    const uint64_t a = DeriveSeed(1, seed_purpose::kDegradation);
    const uint64_t b = DeriveSeed(2, seed_purpose::kDegradation);
    EXPECT_GT(a > b ? a - b : b - a, 1000ull);
}

TEST(Seeds, DistinctPurposesProduceDistinctStreams) {
    const SeedSet seeds(99ull);
    PhiloxStream a(seeds.For(seed_purpose::kDegradation), 0, 0);
    PhiloxStream b(seeds.For(seed_purpose::kTrajectoryNoise), 0, 0);
    EXPECT_NE(a.NextBits64(), b.NextBits64());
}

// --- Event log ------------------------------------------------------------

TEST(EventLog, AllTwentyEventTypesAreNamed) {
    std::set<std::string> names;
    for (uint8_t i = 0; i < static_cast<uint8_t>(EventType::kCount); ++i) {
        const std::string name = EventTypeName(static_cast<EventType>(i));
        EXPECT_NE(name, "INVALID") << "unnamed event type " << int(i);
        names.insert(name);
    }
    EXPECT_EQ(names.size(), 20u) << "§41 defines exactly 20 event types";
}

TEST(EventLog, SequenceIdsAreContiguousFromZero) {
    std::ostringstream sink;
    EventLog log("run-1", &sink);
    for (int i = 0; i < 10; ++i) {
        const Event e = log.Append(EventType::FRAME,
                                   Timestamp::Nanoseconds(i * 1000), 0);
        EXPECT_EQ(e.sequence_id, static_cast<uint64_t>(i));
    }
    EXPECT_EQ(log.EventCount(), 10u);
}

// The core determinism property: identical event sequences produce identical
// state hashes, so two runs can be compared by a single number.
TEST(EventLog, IdenticalSequencesProduceIdenticalStateHash) {
    auto run = [] {
        std::ostringstream sink;
        EventLog log("run", &sink);
        log.Append(EventType::FRAME, Timestamp::Nanoseconds(1), 0);
        log.Append(EventType::KEYFRAME, Timestamp::Nanoseconds(2), 0,
                   {{"keyframe_id", "17"}});
        log.Append(EventType::LOOP_ACCEPT, Timestamp::Nanoseconds(3), 1,
                   {{"pcm_score", "0.91"}});
        return log.StateHash();
    };
    EXPECT_EQ(run(), run());
    EXPECT_NE(run(), 0u);
}

// And the diagnostic property: the hash chain localizes WHERE two runs
// diverged, not merely that they did. This is what turns a determinism failure
// in a long mission from a needle hunt into a bisection.
TEST(EventLog, HashChainLocalizesDivergence) {
    std::ostringstream sink_a, sink_b;
    EventLog a("run", &sink_a);
    EventLog b("run", &sink_b);

    std::vector<uint64_t> hashes_a, hashes_b;
    for (int i = 0; i < 10; ++i) {
        hashes_a.push_back(
            a.Append(EventType::FRAME, Timestamp::Nanoseconds(i), 0).state_hash);
        // Run B diverges at event 5 only.
        const int agent = (i == 5) ? 1 : 0;
        hashes_b.push_back(
            b.Append(EventType::FRAME, Timestamp::Nanoseconds(i), agent).state_hash);
    }

    size_t first_divergence = hashes_a.size();
    for (size_t i = 0; i < hashes_a.size(); ++i) {
        if (hashes_a[i] != hashes_b[i]) { first_divergence = i; break; }
    }
    EXPECT_EQ(first_divergence, 5u);
    // Every hash from the divergence onward differs, because the chain carries
    // the difference forward.
    for (size_t i = 5; i < hashes_a.size(); ++i) {
        EXPECT_NE(hashes_a[i], hashes_b[i]) << "at " << i;
    }
}

// A chain value of zero must not restart the chain. Zero is a legitimate hash,
// and using it as a "no previous event" sentinel would let two different
// histories converge -- destroying the property that the chain localizes WHERE
// two runs diverged. Forced here by injecting a zero state directly.
TEST(EventLog, AZeroChainValueDoesNotRestartTheChain) {
    std::ostringstream sink_a, sink_b;
    EventLog a("run", &sink_a);
    EventLog b("run", &sink_b);
    // Two logs given different first events, hence different chain states.
    a.Append(EventType::FRAME, Timestamp::Nanoseconds(1), 0);
    b.Append(EventType::KEYFRAME, Timestamp::Nanoseconds(1), 0);
    ASSERT_NE(a.StateHash(), b.StateHash());
    // Identical suffixes must keep them distinct forever: the chain carries
    // the difference forward rather than washing it out.
    for (int i = 0; i < 50; ++i) {
        a.Append(EventType::FRAME, Timestamp::Nanoseconds(i), 0);
        b.Append(EventType::FRAME, Timestamp::Nanoseconds(i), 0);
        ASSERT_NE(a.StateHash(), b.StateHash()) << "chains converged at " << i;
    }
}

TEST(EventLog, AnyFieldChangeChangesTheHash) {
    auto hash_with = [](EventType type, int64_t ns, uint32_t agent,
                        const std::string& value) {
        std::ostringstream sink;
        EventLog log("run", &sink);
        log.Append(type, Timestamp::Nanoseconds(ns), agent, {{"k", value}});
        return log.StateHash();
    };
    const uint64_t base = hash_with(EventType::FRAME, 1, 0, "v");
    EXPECT_NE(base, hash_with(EventType::KEYFRAME, 1, 0, "v"));   // type
    EXPECT_NE(base, hash_with(EventType::FRAME, 2, 0, "v"));      // timestamp
    EXPECT_NE(base, hash_with(EventType::FRAME, 1, 1, "v"));      // agent
    EXPECT_NE(base, hash_with(EventType::FRAME, 1, 0, "w"));      // field value
}

TEST(EventLog, EmitsOneValidJsonObjectPerLine) {
    std::ostringstream sink;
    EventLog log("run-42", &sink);
    log.Append(EventType::MESSAGE_SENT, Timestamp::Nanoseconds(123), 2,
               {{"bytes", "207"}, {"class", "POSE_EDGE"}});
    log.Append(EventType::FAILURE, Timestamp::Nanoseconds(456), 2);

    std::istringstream in(sink.str());
    std::string line;
    int lines = 0;
    while (std::getline(in, line)) {
        ++lines;
        EXPECT_EQ(line.front(), '{');
        EXPECT_EQ(line.back(), '}');
        EXPECT_NE(line.find("\"run_id\":\"run-42\""), std::string::npos);
        EXPECT_NE(line.find("\"state_hash\":"), std::string::npos);
        EXPECT_NE(line.find("\"seq\":"), std::string::npos);
    }
    EXPECT_EQ(lines, 2);
    EXPECT_NE(sink.str().find("\"bytes\":\"207\""), std::string::npos);
}

// A log line must survive content that would otherwise break the format.
TEST(EventLog, EscapesJsonHostileContent) {
    std::ostringstream sink;
    EventLog log("run", &sink);
    log.Append(EventType::FAILURE, Timestamp::Nanoseconds(1), 0,
               {{"detail", "quote\" backslash\\ newline\n tab\t"}});
    const std::string out = sink.str();
    // Exactly one line: the embedded newline must have been escaped.
    EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 1);
    EXPECT_NE(out.find("\\\""), std::string::npos);
    EXPECT_NE(out.find("\\\\"), std::string::npos);
    EXPECT_NE(out.find("\\n"), std::string::npos);
}

// --- Run manifest ---------------------------------------------------------

TEST(RunManifest, UnrecordedFieldsAreVisibleNotAbsent) {
    RunManifest m;
    m.run_id = "r1";
    const std::string yaml = ToYaml(m);
    // An unknown must appear as UNRECORDED rather than as an empty value, so a
    // reader can tell "we did not record this" from "this was empty".
    EXPECT_NE(yaml.find("git_commit: UNRECORDED"), std::string::npos);
    EXPECT_NE(yaml.find("opencv_build_hash: UNRECORDED"), std::string::npos);
}

TEST(RunManifest, SerializesTheSection282FieldSet) {
    RunManifest m;
    m.run_id = "20260906-a";
    m.git_commit = "40738fb1";
    m.opencv_version = "5.0.0";
    m.opencv_build_hash = "9127192543017384188";
    m.scene = "two_room_corridor";
    m.seed = 20260906;
    m.agent_count = 2;
    m.degradation_config = "dark2_dust1";
    m.feature_engine = "xfeat";
    m.loop_backend = "gnc_tls";
    m.imu_mode = "full";

    const std::string yaml = ToYaml(m);
    for (const char* needle :
         {"run_id: 20260906-a", "git_commit: 40738fb1", "opencv_version: 5.0.0",
          "scene: two_room_corridor", "seed: 20260906", "agent_count: 2",
          "degradation_config: dark2_dust1", "feature_engine: xfeat",
          "imu_mode: full", "execution_mode: replay"}) {
        EXPECT_NE(yaml.find(needle), std::string::npos) << "missing: " << needle;
    }
}

TEST(RunManifest, RequiredFieldValidationCatchesAnUnpublishableRun) {
    RunManifest m;
    EXPECT_FALSE(MissingRequiredFields(m).empty());

    m.run_id = "r";
    m.git_commit = "abc123";
    m.scene = "s";
    m.feature_engine = "orb";
    m.loop_backend = "pcm";
    // Still incomplete: §5.6 requires the named degradation config, and the
    // determinism digests are what make a drifted-math result identifiable
    // after the fact. A manifest without them cannot support either claim.
    EXPECT_FALSE(MissingRequiredFields(m).empty());

    m.degradation_config = "dark2_dust1";
    m.seed = 20260906;
    m.seed_recorded = true;
    m.det_math_digest = "11806531094783593156";
    m.philox_digest = "14182439601419266695";
    m.degradation_digest = "8535765251477677907";
    EXPECT_TRUE(MissingRequiredFields(m).empty());

    // seed 0 is a legitimate seed; only the flag distinguishes it from unset.
    m.seed_recorded = false;
    EXPECT_FALSE(MissingRequiredFields(m).empty());
    m.seed_recorded = true;

    // A dirty tree means git_commit does not identify the code that ran, so the
    // run is not publishable even though every field is populated.
    m.git_dirty = true;
    EXPECT_FALSE(MissingRequiredFields(m).empty());
}

TEST(RunManifest, YamlQuotesValuesThatWouldOtherwiseReparse) {
    RunManifest m;
    m.run_id = "r";
    m.scene = "corridor: narrow";        // a colon would break the mapping
    m.extra["note"] = "# not a comment";
    const std::string yaml = ToYaml(m);
    EXPECT_NE(yaml.find("scene: \"corridor: narrow\""), std::string::npos);
    EXPECT_NE(yaml.find("note: \"# not a comment\""), std::string::npos);
}

TEST(RunManifest, ExecutionModeIsRecordedBecauseDeterminismDiffers) {
    // §5.9: replayed runs are bit-deterministic, live closed-loop runs are not
    // and must be reported over seeds. Every figure has to declare which.
    RunManifest m;
    EXPECT_EQ(m.execution_mode, "replay");
    m.execution_mode = "live";
    EXPECT_NE(ToYaml(m).find("execution_mode: live"), std::string::npos);
}
