#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define TILE_WIDTH 16

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}


__global__ void conv_forward_kernel(float *d_output, const float *d_input, const float *d_weight,
                                    const int n_sample, const int channel_out, const int channel_in,
                                    const int height_in, const int width_in, const int height_kernel)
{
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) d_output[(i3) * (channel_out * height_out * width_out) + (i2) * (height_out * width_out) + (i1) * (width_out) + i0]
#define x4d(i3, i2, i1, i0) d_input[(i3) * (channel_in * height_in * width_in) + (i2) * (height_in * width_in) + (i1) * (width_in) + i0]
#define k4d(i3, i2, i1, i0) d_weight[(i3) * (channel_in * height_kernel * height_kernel) + (i2) * (height_kernel * height_kernel) + (i1) * (height_kernel) + i0]

    int height_grid = ceil(1.0*height_out / TILE_WIDTH);
    int width_grid = ceil(1.0*width_out / TILE_WIDTH); 
    
    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature
    int h = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int w = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix
    
    float accum = 0.0f;

    if (h < height_out && w < width_out) 
    {
        for(int c=0; c<channel_in; c++)             // sum over all input features
        {
            for(int p=0; p<height_kernel; p++)         // KxK filter 
                for(int q=0; q<height_kernel; q++)
                    accum += x4d(b, c, h+p, w+q) * k4d(m, c, p, q); // 4 dimensions macro resolve thread index
        }
        y4d(b,m,h,w) = accum;
    } // endif (h < H_out && w < W_out)

    #undef y4d
    #undef x4d
    #undef k4d
}
	
void GPUInterface::conv_forward_gpu_prolog(
    const float *output, const float *input,
                               const float *weight, float **d_output,
                               float **d_input, float **d_weight,
                               const int n_sample, const int channel_out, const int channel_in,
                               const int height_in, const int width_in, const int height_kernel) {
  // Allocate memory and copy over the relevant data structures to the GPU

  // We pass double pointers for you to initialize the relevant device pointers,
  //  which are passed to the other two functions.

  const int height_out = height_in - height_kernel + 1;
  const int width_out = width_in - height_kernel + 1;

  int inputSize = n_sample * channel_in * width_in * height_in * sizeof(float); // input features map is C
  int outputSize = n_sample * channel_out * height_out * width_out * sizeof(float);    // output feature map is M
  int maskSize = channel_out * channel_in * height_kernel * height_kernel * sizeof(float); // C * M filter Maps of size K*K

  CHECK(cudaMalloc(d_input, inputSize));
  CHECK(cudaMalloc(d_output, outputSize));
  CHECK(cudaMalloc(d_weight, maskSize));

  // Copy Input data to device
  CHECK(cudaMemcpy(*d_input, input, inputSize, cudaMemcpyHostToDevice));

  // Copy Mask data to device
  CHECK(cudaMemcpy(*d_weight, weight, maskSize, cudaMemcpyHostToDevice));
}


void GPUInterface::conv_forward_gpu(
    float *d_output, const float *d_input,
                        const float *d_weight, const int n_sample, const int channel_out,
                        const int channel_in, const int height_in, const int width_in, const int height_kernel) {
  // Set the kernel dimensions and call the kernel

  const int height_out = height_in - height_kernel + 1;
  const int width_out = width_in - height_kernel + 1;

  int X = ceil(1.0 * height_out / TILE_WIDTH);
  int Y = ceil(1.0 * width_out / TILE_WIDTH);
  int Z = X * Y;

  // Block dimensions = #of threads in the block
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);

  // Grid Dimension = #of Blocks: Batch Size * Num_Output_Features *
  dim3 gridSize(n_sample, channel_out, Z);

  // launch the kernel
  conv_forward_kernel<<<gridSize, blockSize>>>(
      d_output, d_input, d_weight, n_sample, channel_out, channel_in, height_in, width_in, height_kernel);
}

void
GPUInterface::conv_forward_gpu_epilog(float *output, float *d_output, float *d_input,
                               float *d_weight, const int n_sample, const int channel_out,
                               const int channel_in, const int height_in, const int width_in,
                               const int height_kernel) {
  // Copy the output back to host

  const int height_out = height_in - height_kernel + 1;
  const int width_out = width_in - height_kernel + 1;

  int outputSize = n_sample * channel_out * height_out * width_out * sizeof(float);

  CHECK(cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  CHECK(cudaFree(d_weight));
}


