// OpenCV build identity (arch §27.1, C7). Built only where OpenCV 5 exists,
// i.e. inside the ariadne-core image.
#include "runtime/manifest/opencv_identity.hpp"

#include <gtest/gtest.h>

#include <set>
#include <sstream>
#include <string>

using namespace ariadne::runtime;

// C7: one OpenCV, built from the pinned 5.x tag. An OpenCV 4 here means an apt
// package or a stray install won the lookup.
TEST(OpenCvIdentity, IsOpenCv5) {
    EXPECT_EQ(MeasureOpenCvIdentity().version.rfind("5.", 0), 0u);
}

// The loaded build must carry the contrib modules C7 names and the core modules
// the pipeline links. Checked against the build text, not the headers, because
// a module can have headers installed and still be missing from the library.
TEST(OpenCvIdentity, LoadedBuildHasThePinnedModules) {
    const std::string info = OpenCvBuildInformation();
    const auto pos = info.find("To be built:");
    ASSERT_NE(pos, std::string::npos);
    // Exact tokens, not substrings: " video" is also a prefix of " videoio".
    std::istringstream line(info.substr(pos + 12, info.find('\n', pos) - pos - 12));
    std::set<std::string> built;
    for (std::string m; line >> m;) built.insert(m);
    for (const char* mod : {"features", "geometry", "calib", "stereo", "ptcloud", "video",
                            "ximgproc", "xphoto", "optflow", "quality"}) {
        EXPECT_EQ(built.count(mod), 1u) << "missing module " << mod;
    }
}

TEST(OpenCvIdentity, HashIsPrefixedAndStable) {
    const OpenCvIdentity a = MeasureOpenCvIdentity();
    const OpenCvIdentity b = MeasureOpenCvIdentity();
    EXPECT_EQ(a.build_hash, b.build_hash);
    ASSERT_EQ(a.build_hash.size(), std::string("fnv1a64:").size() + 16);
    EXPECT_EQ(a.build_hash.rfind("fnv1a64:", 0), 0u);
    EXPECT_EQ(a.build_hash.find_first_not_of("0123456789abcdef", 8), std::string::npos);
}

TEST(OpenCvIdentity, RecordFillsTheManifest) {
    RunManifest m;
    RecordOpenCvIdentity(m);
    EXPECT_NE(m.opencv_version, kUnknownField);
    EXPECT_NE(m.opencv_build_hash, kUnknownField);
    EXPECT_NE(m.opencv_cpu_features, kUnknownField);
    const std::string yaml = ToYaml(m);
    // The value contains ':' so the writer quotes it; compare the escaped form.
    EXPECT_NE(yaml.find("opencv_build_hash: " + detail::YamlEscape(m.opencv_build_hash)),
              std::string::npos);
}
