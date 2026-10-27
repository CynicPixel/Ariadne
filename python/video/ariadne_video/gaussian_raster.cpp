// CPU back-to-front alpha compositing for projected elliptical Gaussians.
#include <algorithm>
#include <cmath>
#include <cstdint>
extern "C" void raster_gaussians(int n, int width, int height,
    const double* xy, const double* inverse, const int32_t* extent,
    const double* rgba, const int64_t* order, float* out) {
  for (int k = 0; k < n; ++k) {
    const auto i = order[k];
    const double cx = xy[2*i], cy = xy[2*i+1];
    const int x0 = std::max(0, int(std::floor(cx-extent[2*i])));
    const int x1 = std::min(width, int(std::ceil(cx+extent[2*i]))+1);
    const int y0 = std::max(0, int(std::floor(cy-extent[2*i+1])));
    const int y1 = std::min(height, int(std::ceil(cy+extent[2*i+1]))+1);
    for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
      const double dx = x+0.5-cx, dy = y+0.5-cy;
      const double power = inverse[4*i]*dx*dx + 2*inverse[4*i+1]*dx*dy + inverse[4*i+3]*dy*dy;
      if (power > 9) continue;
      const float a = std::min(0.99, rgba[4*i+3]*std::exp(-0.5*power));
      float* pixel = out + 4*(y*width+x);
      for (int c = 0; c < 3; ++c) pixel[c] = float(rgba[4*i+c])*a + pixel[c]*(1-a);
      pixel[3] = a + pixel[3]*(1-a);
    }
  }
}
