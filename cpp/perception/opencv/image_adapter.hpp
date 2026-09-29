#pragma once
// ---------------------------------------------------------------------------
// The cv::Mat boundary for Ariadne's raw image buffers (core/image/image.hpp).
//
// Everything upstream of here (the bag reader, the degradation operator D) is
// OpenCV-free by design; this is the one place pixels cross into OpenCV.
//
// CHANNEL ORDER IS EXPLICIT. Gazebo and ROS publish rgb8; OpenCV's convention is
// BGR. Converting rgb8 with COLOR_BGR2GRAY does not fail, it silently swaps the
// red and blue luma weights (0.299 vs 0.114), and every downstream number
// shifts. So the pixel format travels with the image and there is no default.
// ---------------------------------------------------------------------------
#include "core/image/image.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace ariadne::perception {

enum class PixelFormat { kGray8, kRgb8, kBgr8 };

/// Zero-copy view of an ImageU8 as a cv::Mat. The Mat does NOT own the pixels:
/// `image` must outlive it and must not be resized while it is in use. Use
/// .clone() when the result has to outlive the buffer.
inline cv::Mat WrapU8(const core::ImageU8& image) {
    if (image.Empty()) return {};
    return cv::Mat(image.height, image.width, CV_8UC(image.channels),
                   const_cast<uint8_t*>(image.data.data()));
}

/// Grayscale for the geometry path (§6.3). BT.601 luma weights, as OpenCV's
/// cvtColor applies them, with the channel order taken from `format`.
inline cv::Mat ToGray(const cv::Mat& image, PixelFormat format) {
    switch (format) {
        case PixelFormat::kGray8:
            if (image.type() != CV_8UC1) throw std::invalid_argument("ToGray: expected CV_8UC1");
            return image;
        case PixelFormat::kRgb8:
        case PixelFormat::kBgr8: {
            if (image.type() != CV_8UC3) throw std::invalid_argument("ToGray: expected CV_8UC3");
            cv::Mat gray;
            cv::cvtColor(image, gray,
                         format == PixelFormat::kRgb8 ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
            return gray;
        }
    }
    throw std::invalid_argument("ToGray: unknown pixel format");
}

}  // namespace ariadne::perception
