#ifndef SRC_LAYER_GPU_NEW_FORWARD_H
#define SRC_LAYER_GPU_NEW_FORWARD_H

class GPUInterface {
public:
  void conv_forward_gpu_prolog(const float *output, const float *input,
                               const float *weight, float **d_output,
                               float **d_input, float **d_weight,
                               const int n_sample, const int channel_out, const int channel_in,
                               const int height_in, const int width_in, const int height_kernel);
  void conv_forward_gpu(float *d_output, const float *d_input,
                        const float *d_weight, const int n_sample, const int channel_out,
                        const int channel_in, const int height_in, const int width_in, const int height_kernel);
  void conv_forward_gpu_epilog(float *output, float *d_output, float *d_input,
                               float *d_weight, const int n_sample, const int channel_out,
                               const int channel_in, const int height_in, const int width_in,
                               const int height_kernel);
};

#endif
