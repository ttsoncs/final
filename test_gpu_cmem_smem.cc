/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/conv.h"
#include "src/layer/conv_gpu.h"
#include "src/layer/conv_gpu_cmem.h"
#include "src/layer/conv_gpu_cmem_smem.h"
#include "src/layer/fully_connected.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/loss/mse_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

int main() {
  // data
  MNIST dataset("../data/mnist/");
  dataset.read_test_data();
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn;
  Layer *conv1 = new Conv_GPU_CMEM_SMEM(1, 28, 28, 6, 5, 5);
  Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
  Layer *conv2 = new Conv_GPU_CMEM_SMEM(6, 12, 12, 16, 5, 5);
  Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
  Layer *fc3 = new FullyConnected(pool2->output_dim(), 120);
  Layer *fc4 = new FullyConnected(120, 84);
  Layer *fc5 = new FullyConnected(84, 10);
  Layer *relu1 = new ReLU;
  Layer *relu2 = new ReLU;
  Layer *relu3 = new ReLU;
  Layer *relu4 = new ReLU;
  Layer *softmax = new Softmax;
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool1);
  dnn.add_layer(conv2);
  dnn.add_layer(relu2);
  dnn.add_layer(pool2);
  dnn.add_layer(fc3);
  dnn.add_layer(relu3);
  dnn.add_layer(fc4);
  dnn.add_layer(relu4);
  dnn.add_layer(fc5);
  dnn.add_layer(softmax);
  // loss
  Loss *loss = new CrossEntropy;
  dnn.add_loss(loss);

  // test
  dnn.load_parameters("../model/weights.bin");
  dnn.forward(dataset.test_data);
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout << "test accuracy: " << acc * 100 << "%" << std::endl;
  return 0;
}
