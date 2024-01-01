#include "cpu-new-forward.h"

void conv_forward_cpu(float *output, const float *input, const float *weight,
                      const int n_sample, const int channel_out,
                      const int channel_in, const int height_in,
                      const int width_in, const int height_kernel) {
  const int height_out = height_in - height_kernel + 1;
  const int width_out = width_in - height_kernel + 1;

#define y4d(i3, i2, i1, i0)                                                    \
  output[(i3) * (channel_out * height_out * width_out) +                       \
         (i2) * (height_out * width_out) + (i1) * (width_out) + i0]
#define x4d(i3, i2, i1, i0)                                                    \
  input[(i3) * (channel_in * height_in * width_in) +                           \
        (i2) * (height_in * width_in) + (i1) * (width_in) + i0]
#define k4d(i3, i2, i1, i0)                                                    \
  weight[(i3) * (channel_in * height_kernel * height_kernel) +                 \
         (i2) * (height_kernel * height_kernel) + (i1) * (height_kernel) + i0]

  for (int b = 0; b < n_sample; b++) {
    for (int m = 0; m < channel_out; m++) {
      for (int h = 0; h < height_out; h++) {
        for (int w = 0; w < width_out; w++) {
          y4d(b, m, h, w) = 0;
          for (int c = 0; c < channel_in; c++) {
            for (int p = 0; p < height_kernel; p++) {
              for (int q = 0; q < height_kernel; q++) {
                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
              }
            }
          }
        }
      }
    }
  }
#undef y4d
#undef x4d
#undef k4d
}