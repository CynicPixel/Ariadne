// ---------------------------------------------------------------------------
// ariadne_charuco_assets: the ChArUco calibration boards placed in every
// generated scene's entrance rooms (design §3/§6), drawn by OpenCV itself so
// the detector and the drawing agree by construction.
//
//   ariadne_charuco_assets --out DIR
//
// Writes board_<k>.pgm (8-bit, white margin included) and boards.json. Board k
// uses marker ids [k*n, (k+1)*n) of DICT_5X5_250, with n the number of markers
// on one board: DISJOINT id ranges, so the two entrance boards can never be
// confused for one another (a false inter-agent loop, design §1).
//
// Why PGM: worldgen's PNG reader handles only filter-type-0 files (it exists to
// verify its own output); OpenCV's PNG writer uses adaptive filters. PGM is a
// raw raster with a text header, trivially and exactly readable anywhere.
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kSquaresX = 7;
constexpr int kSquaresY = 5;
constexpr float kSquareM = 0.08f;          // 8 cm squares
constexpr float kMarkerM = 0.06f;          // 6 cm markers
constexpr int kPxPerSquare = 80;           // 1000 px/m: sharp edges at 0.5-3 m
constexpr int kMarginSquares = 1;          // white quiet zone, one square wide
constexpr int kBoards = 2;                 // one per entrance

}  // namespace

int main(int argc, char** argv) {
    std::string out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--out") out = argv[i + 1];
    }
    if (out.empty()) {
        std::fprintf(stderr, "usage: ariadne_charuco_assets --out DIR\n");
        return 2;
    }
    std::filesystem::create_directories(out);
    const cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    const cv::aruco::CharucoBoard probe(cv::Size(kSquaresX, kSquaresY), kSquareM, kMarkerM, dict);
    const int n = static_cast<int>(probe.getIds().size());
    const int margin_px = kMarginSquares * kPxPerSquare;
    const cv::Size img_size((kSquaresX + 2 * kMarginSquares) * kPxPerSquare,
                            (kSquaresY + 2 * kMarginSquares) * kPxPerSquare);

    std::ofstream js(out + "/boards.json");
    js << "{\n \"generator\": \"cpp/runtime/tools/ariadne_charuco_assets.cpp\",\n"
       << " \"opencv\": \"" << CV_VERSION << "\",\n"
       << " \"dictionary\": \"DICT_5X5_250\",\n"
       << " \"squares_xy\": [" << kSquaresX << ", " << kSquaresY << "],\n"
       << " \"square_m\": " << kSquareM << ",\n \"marker_m\": " << kMarkerM << ",\n"
       << " \"margin_squares\": " << kMarginSquares << ",\n \"px_per_square\": " << kPxPerSquare << ",\n"
       << " \"image_wh\": [" << img_size.width << ", " << img_size.height << "],\n"
       << " \"legacy_pattern\": false,\n \"markers_per_board\": " << n << ",\n \"boards\": [\n";
    for (int k = 0; k < kBoards; ++k) {
        std::vector<int> ids(n);
        std::iota(ids.begin(), ids.end(), k * n);
        const cv::aruco::CharucoBoard board(cv::Size(kSquaresX, kSquaresY), kSquareM, kMarkerM, dict, ids);
        cv::Mat img;
        board.generateImage(img_size, img, margin_px, 1);
        CV_Assert(img.type() == CV_8UC1 && img.size() == img_size);
        const std::string name = "board_" + std::to_string(k) + ".pgm";
        if (!cv::imwrite(out + "/" + name, img)) {
            std::fprintf(stderr, "failed to write %s\n", name.c_str());
            return 1;
        }
        js << "  {\"index\": " << k << ", \"file\": \"" << name << "\", \"first_id\": " << k * n
           << ", \"last_id\": " << (k + 1) * n - 1 << "}" << (k + 1 < kBoards ? "," : "") << "\n";
    }
    js << " ]\n}\n";
    std::printf("wrote %d boards (%d markers each, %dx%d px) to %s\n", kBoards, n, img_size.width,
                img_size.height, out.c_str());
    return 0;
}
