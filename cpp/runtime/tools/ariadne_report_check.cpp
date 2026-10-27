// ---------------------------------------------------------------------------
// ariadne_report_check: the §30.2 gate (D31; design evaluation-harness.md). The metric
// registry in runtime/metrics/metric_registry.hpp is the single source of truth;
// the Python harness never re-lists the ids.
//
//   ariadne_report_check --dump-registry FILE   write id / group / unit / source (YAML)
//   ariadne_report_check --check REPORT.yaml    MissingMetrics + UnknownMetrics on its
//                                               `metrics:` map; exit 0 iff both empty
// ---------------------------------------------------------------------------
#include "runtime/metrics/metric_registry.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

namespace {
const char* GroupName(ariadne::runtime::MetricGroup g) {
    using G = ariadne::runtime::MetricGroup;
    switch (g) {
        case G::kLocalization: return "localization";
        case G::kRegistration: return "registration";
        case G::kMapping: return "mapping";
        case G::kCommunication: return "communication";
        case G::kMission: return "mission";
    }
    return "unknown";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: ariadne_report_check --dump-registry FILE | --check REPORT.yaml");
        const std::string mode = argv[1], path = argv[2];
        using namespace ariadne::runtime;
        if (mode == "--dump-registry") {
            std::ofstream o(path);
            o << "# The §30.2 metric registry (runtime/metrics/metric_registry.hpp). Generated; do not hand-edit.\n"
              << "metrics:\n";
            for (const MetricSpec& m : kCoreMetrics) {
                o << "  - {id: " << m.id << ", group: " << GroupName(m.group) << ", unit: \"" << m.unit
                  << "\", source: \"" << m.source << "\"}\n";
            }
            return 0;
        }
        if (mode != "--check") throw std::invalid_argument("unknown mode " + mode);
        const YAML::Node r = YAML::LoadFile(path)["metrics"];
        if (!r || !r.IsMap()) throw std::runtime_error("report has no `metrics:` map");
        MetricReport report;
        for (const auto& kv : r) {
            const YAML::Node v = kv.second;
            double x = std::numeric_limits<double>::quiet_NaN();
            if (v.IsScalar()) {
                try {
                    x = v.as<double>();
                } catch (const YAML::Exception&) {   // null / text: not a number, so the metric is missing
                }
            }
            report[kv.first.as<std::string>()] = x;
        }
        const auto missing = MissingMetrics(report), unknown = UnknownMetrics(report);
        std::printf("registry: %zu metrics; report: %zu entries\n", kCoreMetrics.size(), report.size());
        std::printf("missing (absent or non-finite): %zu\n", missing.size());
        for (const auto& m : missing) std::printf("  MISSING %s\n", m.c_str());
        std::printf("unknown ids: %zu\n", unknown.size());
        for (const auto& u : unknown) std::printf("  UNKNOWN %s\n", u.c_str());
        const bool pass = missing.empty() && unknown.empty();
        std::printf("VERDICT %s\n", pass ? "PASS (every §30.2 metric emits a number)" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_report_check: %s\n", e.what());
        return 2;
    }
}
