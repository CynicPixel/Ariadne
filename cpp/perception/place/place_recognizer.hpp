#pragma once
// ---------------------------------------------------------------------------
// Global place descriptor (arch §13, §7.4; perception.md §11): EigenPlaces
// R50-256 (Berton et al., ICCV 2023; gmberton/EigenPlaces, MIT) exported to
// ONNX by python/model_tools/eigenplaces_reference.py and run by cv::dnn.
//
// WHAT IT IS. A ResNet-50 trained so that images of the same place from
// different viewpoints map to nearby vectors. Its last feature map (2048 x
// H/32 x W/32) is L2-normalised per location, GeM-pooled (the generalised
// mean, (mean x^p)^(1/p), between average pooling at p = 1 and max pooling as
// p -> inf; p is learned), projected to 256-D and L2-normalised: one unit
// vector per image. Place similarity = cosine = dot product. It fills the
// GLOBAL_DESCRIPTOR contract the mean-bit placeholder fills today (256-D unit
// vector, cosine; collaboration/retrieval/mean_bit_descriptor.hpp).
//
// PREPROCESSING, exactly as the exporter's: gray / 255 -> INTER_AREA to the
// export's fixed size (identity at the native size) -> the gray channel
// replicated to RGB -> ImageNet mean/std. The sim rig's RGB path
// (DescribeRgb) takes the image in RGB ORDER (StereoFrame::left_rgb, rgb8 as
// published): /255 -> INTER_AREA on all three channels -> mean/std per
// channel. A BGR image passed there is not an error but a different input
// (the R/B means differ), so the channel order is in the method name. The ONNX graph has a FIXED input
// size (GeM's kernel is the map size); the file name carries it.
//
// COST (SMOKE §4r, indicative): 0.2-0.5 s at 376x240, 1-2.7 s at 752x480 on
// this CPU. Not for every keyframe: describe a spaced subset.
// A cv::dnn::Net is not thread-safe (§7.4): the Net lives in a dnn::PerThread (D91), so one PlaceRecognizer may
// be shared by threads; each calling thread loads and runs its own.
// ---------------------------------------------------------------------------
#include "perception/dnn/per_thread.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

namespace ariadne::perception {

struct PlaceRecognizerParams {
    std::string onnx;            // eigenplaces_r50_256_<W>x<H>.onnx
    int width = 376, height = 240;
    std::string engine = "classic";   // classic | new (cv::dnn ENGINE_CLASSIC / ENGINE_NEW; both exact, SMOKE §4r)
    // Retrieval-path enhancement (arch §6.3, D92): none | clahe. CLAHE (clip 2.0, 8x8 tiles, fixed before the 3.1
    // ablation) on a COPY at the input resolution, before the resize: gray directly, RGB on Lab's L channel. The
    // caller's frame is never modified, so the geometry path keeps its calibrated-only pixels.
    std::string enhance = "none";
};

class PlaceRecognizer {
public:
    explicit PlaceRecognizer(const PlaceRecognizerParams& p);

    /// 256-D unit vector for an 8-bit gray image (any size; resized to the export's).
    std::vector<float> Describe(const cv::Mat& gray) const;

    /// 256-D unit vector for an 8-bit image in RGB order (CV_8UC3; any size; resized to the export's).
    std::vector<float> DescribeRgb(const cv::Mat& rgb) const;
    /// The exact network input: 1 x 3 x H x W CV_32F (NCHW).
    static cv::Mat Preprocess(const cv::Mat& gray, int width, int height);
    static cv::Mat PreprocessRgb(const cv::Mat& rgb, int width, int height);
    /// The retrieval-path enhancement alone (a new image; the input is untouched). none returns the input itself.
    static cv::Mat Enhance(const cv::Mat& image, const std::string& enhance, bool rgb);

    /// Manifest id, e.g. "eigenplaces_r50_256@376x240/cv_dnn_classic".
    std::string Name() const;
    /// The manifest record (§7.4): model, file, engine.
    dnn::ModelRecord Record() const;

private:
    std::vector<float> Run(const cv::Mat& blob) const;
    PlaceRecognizerParams p_;
    dnn::PerThread<cv::dnn::Net> net_;
};

}  // namespace ariadne::perception
