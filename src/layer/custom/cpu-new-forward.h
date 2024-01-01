#ifndef SRC_LAYER_CPU_NEW_FORWARD_H
#define SRC_LAYER_CPU_NEW_FORWARD_H

void conv_forward_cpu(float *output, const float *input, const float *weight,
                      const int n_sample, const int channel_out,
                      const int channel_in, const int height_in,
                      const int width_in, const int height_kernel);

#endif // SRC_LAYER_CPU_NEW_FORWARD_H