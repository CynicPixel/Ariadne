#include "runtime/live/demo_camera.hpp"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <unistd.h>

TEST(DemoCamera, PublishesMatchedDownsampledImageAndFeatures) {
    const auto dir = std::filesystem::temp_directory_path() / ("ariadne-camera-test-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    {
        ariadne::runtime::live::DemoCamera camera(dir, 2);
        ASSERT_TRUE(camera.Wants(100000000));
        ASSERT_FALSE(camera.Wants(150000000));
        cv::Mat rgb(720, 1280, CV_8UC3, cv::Scalar(220, 50, 20));
        camera.Put(100000000, rgb, {{200, 100}, {600, 400}});
    } // destructor flushes the last pending packet
    const auto manifest = YAML::LoadFile((dir / "cam_2.json").string());
    ASSERT_EQ(manifest["t_ns"].as<int64_t>(), 100000000);
    ASSERT_EQ(manifest["agent"].as<int>(), 2);
    const auto jpg = cv::imread((dir / manifest["image"].as<std::string>()).string());
    EXPECT_EQ(jpg.cols, 640);
    EXPECT_EQ(jpg.rows, 360);
    EXPECT_FLOAT_EQ(manifest["pts"][0][0].as<float>(), 100);
    EXPECT_FLOAT_EQ(manifest["pts"][0][1].as<float>(), 50);
    EXPECT_GT(jpg.at<cv::Vec3b>(0, 0)[2], jpg.at<cv::Vec3b>(0, 0)[0]); // RGB to JPEG BGR
    std::filesystem::remove_all(dir);
}
