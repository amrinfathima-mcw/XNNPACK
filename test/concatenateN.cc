// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdint.h>
#include <vector>
#include <iostream>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

// template <typename T>
// class ConcatenateNTest : public ::testing::Test {
// protected:
//   ConcatenateNTest() {
//     shape_dist = std::uniform_int_distribution<size_t>(1, XNN_MAX_TENSOR_DIMS);
//     dim_dist = std::uniform_int_distribution<size_t>(1, 9);
//     f32dist = std::uniform_real_distribution<float>();
//     i8dist = std::uniform_int_distribution<int32_t>(
//         std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
//     u8dist = std::uniform_int_distribution<int32_t>(
//         std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
//     scale_dist = std::uniform_real_distribution<float>(0.1f, 5.0f);

//     num_inputs = RandomNumInputs();
//     input_dims.resize(num_inputs);
//     output_dims = RandomShape();

//     for (size_t i = 0; i < num_inputs; ++i) {
//       axis = RandomAxis(output_dims);
//       input_dims[i] = RandomShape(output_dims, axis);
//       output_dims[axis] += input_dims[i][axis];  // Update output size for concatenation
//     }

//     inputs.resize(num_inputs);
//     for (size_t i = 0; i < num_inputs; ++i) {
//       inputs[i] = std::vector<T>(NumElements(input_dims[i]));
//     }

//     operator_output = std::vector<T>(NumElements(output_dims));
//     subgraph_output = std::vector<T>(NumElements(output_dims));

//     signed_zero_point = i8dist(rng);
//     unsigned_zero_point = u8dist(rng);
//     scale = scale_dist(rng);

  //   batch_size = 1;
  //   std::vector<size_t> channel_sizes(num_inputs, 1); // Initialize channel sizes for each input
     
  //   // Calculate batch size based on dimensions before the axis
  //   for (size_t i = 0; i < axis; i++) {
  //       batch_size *= output_dims[i];
  //   }

  //   // Calculate channels for each input based on dimensions after the axis
  //   for (size_t input_index = 0; input_index < num_inputs; input_index++) {
  //       const auto& input_dim = input_dims[input_index]; // Assuming input_dims_list contains all input dimensions
  //       for (size_t i = axis; i < input_dim.size(); i++) {
  //           channel_sizes[input_index] *= input_dim[i];
  //       }
  //       std::cout<<"input idx "<<input_index<<" channel size "<<channel_sizes[input_index]<<std::endl;
  //   }
  //   channels.resize(channel_sizes.size());
  //   channels = channel_sizes;
  //   output_stride = std::accumulate(channel_sizes.begin(), channel_sizes.end(), size_t(0));
  // }

//   size_t RandomNumInputs() {
//     return std::uniform_int_distribution<size_t>(2, 5)(rng);  // You can adjust the range
//   }

//   std::vector<size_t> RandomShape() {
//     std::vector<size_t> dims(shape_dist(rng));
//     std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
//     return dims;
//   }

//   std::vector<size_t> RandomShape(const std::vector<size_t> base_dims, size_t axis) {
//     auto dims = base_dims;
//     dims[axis] = dim_dist(rng);
//     return dims;
//   }

//   size_t RandomAxis(const std::vector<size_t>& dims) {
//     return std::uniform_int_distribution<size_t>(0, dims.size() - 1)(rng);
//   }

//   size_t NumElements(const std::vector<size_t>& dims) {
//     return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
//   }
//   xnnpack::ReplicableRandomDevice rng;
//   std::uniform_int_distribution<size_t> shape_dist;
//   std::uniform_int_distribution<size_t> dim_dist;
//   std::uniform_real_distribution<float> f32dist;
//   std::uniform_int_distribution<int32_t> i8dist;
//   std::uniform_int_distribution<int32_t> u8dist;
//   std::uniform_real_distribution<float> scale_dist;

//   size_t num_inputs;
//   size_t axis;
//   size_t batch_size;
//   std::vector<size_t> channels;
//   size_t output_stride;

//   //uint32_t input1_ids;
//   uint32_t output_id;

//   std::vector<std::vector<size_t>> input_dims;
//   std::vector<size_t> output_dims;

//   int32_t signed_zero_point;
//   int32_t unsigned_zero_point;
//   float scale;

//   std::vector<std::vector<T>> inputs;
//   std::vector<T> operator_output;
//   std::vector<T> subgraph_output;
// };

template <typename T>
class ConcatenateNTest : public ::testing::Test {
protected:
  // Static configuration
  const size_t num_inputs = 3;  // Number of inputs
  const size_t axis = 0;  // Concatenation axis

  // Predefined input dimensions for each input tensor
  const std::vector<std::vector<size_t>> input_dims = {
    {2, 3},  // Input 0
    {2, 4},  // Input 1
    {2, 2}   // Input 2
  };

  // Predefined output dimensions based on the concatenation
  const std::vector<size_t> output_dims = {2, 9};  // Concatenated along axis 0

  uint32_t output_id;
  std::vector<std::vector<T>> inputs;
  int32_t signed_zero_point = 0;  
  int32_t unsigned_zero_point = 0; 
  float scale = 1.0f; 
  std::vector<T> operator_output;  
  std::vector<T> subgraph_output;  
  size_t output_stride; 
  size_t batch_size = 1;  
  std::vector<size_t> channels;
  std::vector<size_t> channel_sizes; // Declare channel_sizes as a member variable

  void SetUp() override {
    // Initialize channel sizes for each input
    channel_sizes.resize(num_inputs, 1);

    // Calculate batch size based on dimensions before the axis
    for (size_t i = 0; i < axis; i++) {
      batch_size *= output_dims[i];
    }

    // Calculate channels for each input based on dimensions after the axis
    for (size_t input_index = 0; input_index < num_inputs; input_index++) {
      const auto& input_dim = input_dims[input_index]; // Assuming input_dims contains all input dimensions
      for (size_t i = axis; i < input_dim.size(); i++) {
        channel_sizes[input_index] *= input_dim[i];
      }
      std::cout << "input idx " << input_index << " channel size " << channel_sizes[input_index] << std::endl;
    }

    // Initialize input data for testing
    inputs.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      inputs[i] = std::vector<T>(NumElements(input_dims[i]), static_cast<T>(1));  // Fill with static data
    }

    // Initialize output buffers
    operator_output.resize(NumElements(output_dims), static_cast<T>(0));
    subgraph_output.resize(NumElements(output_dims), static_cast<T>(0));

    // Calculate output stride
    output_stride = CalculateOutputStride();

    // Initialize channels based on channel sizes
    channels.resize(num_inputs);
    for (size_t input_index = 0; input_index < num_inputs; ++input_index) {
      channels[input_index] = channel_sizes[input_index];
    }
  }

  size_t NumElements(const std::vector<size_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  size_t CalculateOutputStride() {
    size_t stride = 1;
    for (size_t i = axis + 1; i < output_dims.size(); ++i) {
      stride *= output_dims[i];
    }
    return stride;
  }
};

using ConcatenateNTestQS8 = ConcatenateNTest<int8_t>;
using ConcatenateNTestQU8 = ConcatenateNTest<uint8_t>;
using ConcatenateNTestF16 = ConcatenateNTest<xnn_float16>;
using ConcatenateNTestF32 = ConcatenateNTest<float>;


TEST_F(ConcatenateNTestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
        subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, num_inputs,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate(subgraph, axis, num_inputs, input_ids.data(), output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  std::cout<<"node inputs "<<node->num_inputs<<"  num inputs "<<num_inputs<<std::endl; 
  ASSERT_EQ(node->type, xnn_node_type_concatenate_n);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);
  
  for (size_t i = 0; i < num_inputs; ++i) {
    ASSERT_EQ(node->inputs[i], input_ids[i]);
  }
  
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateNTestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
        subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, num_inputs,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate(subgraph, axis, num_inputs, input_ids.data(), output_id, /*flags=*/0));

  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate_n);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);
//   ASSERT_EQ(node->inputs[0], input1_id);
//   ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateNTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
        subgraph, xnn_datatype_fp16, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }
  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate(subgraph, axis, num_inputs, input_ids.data(), output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate_n);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);
//   ASSERT_EQ(node->inputs[0], input1_id);
//   ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateNTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
        subgraph, xnn_datatype_fp32, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate(subgraph, axis, num_inputs, input_ids.data(), output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate_n);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);
//   ASSERT_EQ(node->inputs[0], input1_id);
//   ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateNTestQS8, matches_operator_api)
{
  // Generate input data for multiple tensors
  // for (size_t i = 0; i < num_inputs; ++i) {
  //   std::generate(inputs[i].begin(), inputs[i].end(), [&]() { return i8dist(rng); });
  //   std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));
  // }
     inputs.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        // Use static values instead of random generation
        inputs[i] = std::vector<int8_t>(NumElements(input_dims[i]), static_cast<int8_t>(i + 1)); // For example, fill with i + 1

    }
    for(int i=0;i<inputs[0].size();++i){
      std::cout<<"input 1: "<<int(inputs[0][i])<<std::endl;
    }
    for(int i=0;i<inputs[1].size();++i){
      std::cout<<"input 2: "<<int(inputs[1][i])<<std::endl;
    }
    for(int i=0;i<inputs[2].size();++i){
      std::cout<<"input 3: "<<int(inputs[2][i])<<std::endl;
    }
    std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  std::vector<xnn_operator_t> operators(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    std::cout<<"before xnn create copy operator "<<std::endl;
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &operators[i]));
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(operators[i], xnn_delete_operator);
    std::cout << "i: " << i << ", batch_size: " << batch_size
          << ", channel_sizes[i]: " << channels[i]
          << ", output_stride: " << output_stride << std::endl;
          std::cout<<"before xnn reshape operator "<<std::endl;
    ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(operators[i], batch_size, channels[i], channels[i], output_stride, /*threadpool=*/nullptr));
    std::cout<<"before xnn setup operator "<<std::endl;
    ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x8(operators[i], inputs[i].data(), operator_output.data() + (i * input_dims[i][1])));
    std::cout<<"before xnn run operator "<<std::endl;
    ASSERT_EQ(xnn_status_success, xnn_run_operator(operators[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
        subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, num_inputs,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate(subgraph, axis, num_inputs, input_ids.data(), output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  
  std::vector<xnn_external_value> external(1 + num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    external[i] = xnn_external_value{input_ids[i], inputs[i].data()};
  }
  external[num_inputs] = xnn_external_value{output_id, subgraph_output.data()};

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  std::cout<<"subgrpah output size "<<subgraph_output.size()<<" operator_output size "<<operator_output.size()<<std::endl;
  for(int i=0;i<10;++i){
    std::cout<<"subgraph output "<<int(subgraph_output[i])<<"  operator output: "<<int(operator_output[i])<<std::endl;
  }
  ASSERT_EQ(subgraph_output, operator_output);
}


// TEST_F(Concatenate2TestQU8, matches_operator_api)
// {
//   std::generate(input1.begin(), input1.end(), [&]() { return u8dist(rng); });
//   std::generate(input2.begin(), input2.end(), [&]() { return u8dist(rng); });
//   std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
//   std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));

//   ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

//   xnn_operator_t op1 = nullptr;
//   xnn_operator_t op2 = nullptr;

//   // Call operator API.
//   ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op1));
//   std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
//   ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op2));
//   std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);

//   ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op1, batch_size, channels_1, channels_1, output_stride, /*threadpool=*/nullptr));
//   ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op2, batch_size, channels_2, channels_2, output_stride, /*threadpool=*/nullptr));

//   ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x8(op1, input1.data(), operator_output.data()));
//   ASSERT_EQ(
//     xnn_status_success,
//     xnn_setup_copy_nc_x8(op2, input2.data(), (uint8_t*) operator_output.data() + op1->channels));

//   ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
//   ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));

//   // Call subgraph API.
//   xnn_subgraph_t subgraph = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
//   std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

//   input1_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success,
//     xnn_define_quantized_tensor_value(
//       subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr, 0,
//       /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
//   ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

//   input2_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success,
//     xnn_define_quantized_tensor_value(
//       subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr, 1,
//       /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
//   ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

//   output_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success,
//     xnn_define_quantized_tensor_value(
//       subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, 2,
//       /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
//   ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

//   ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

//   xnn_runtime_t runtime = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
//   ASSERT_NE(nullptr, runtime);
//   std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
//   std::array<xnn_external_value, 3> external = {
//     xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
//     xnn_external_value{output_id, subgraph_output.data()}};
//   ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
//   ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

//   // Check outputs match.
//   ASSERT_EQ(subgraph_output, operator_output);
// }

// TEST_F(Concatenate2TestF16, matches_operator_api)
// {
//   std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
//   std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
//   std::fill(operator_output.begin(), operator_output.end(), std::nanf(""));
//   std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

//   ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

//   xnn_operator_t op1 = nullptr;
//   xnn_operator_t op2 = nullptr;

//   // Call operator API.
//   ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op1));
//   std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
//   ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op2));
//   std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);

//   ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op1, batch_size, channels_1, channels_1, output_stride, /*threadpool=*/nullptr));
//   ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op2, batch_size, channels_2, channels_2, output_stride, /*threadpool=*/nullptr));

//   ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x16(op1, input1.data(), operator_output.data()));
//   ASSERT_EQ(
//     xnn_status_success,
//     xnn_setup_copy_nc_x16( op2, input2.data(), (xnn_float16*) operator_output.data() + op1->channels));

//   ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
//   ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));

//   // Call subgraph API.
//   xnn_subgraph_t subgraph = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
//   std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

//   input1_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp16, input1_dims.size(), input1_dims.data(), nullptr, 0,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
//   ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

//   input2_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp16, input2_dims.size(), input2_dims.data(), nullptr, 1,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
//   ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

//   output_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, 2,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
//   ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

//   ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

//   xnn_runtime_t runtime = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
//   ASSERT_NE(nullptr, runtime);
//   std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
//   std::array<xnn_external_value, 3> external = {
//     xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
//     xnn_external_value{output_id, subgraph_output.data()}};
//   ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
//   ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

//   // Check outputs match.
//   ASSERT_EQ(subgraph_output, operator_output);
// }

// TEST_F(Concatenate2TestF32, matches_operator_api)
// {
//   std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
//   std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
//   std::fill(operator_output.begin(), operator_output.end(), std::nanf(""));
//   std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

//   ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

//   xnn_operator_t op1 = nullptr;
//   xnn_operator_t op2 = nullptr;

//   // Call operator API.
//   ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op1));
//   std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
//   ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op2));
//   std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);

//   ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op1, batch_size, channels_1, channels_1, output_stride, /*threadpool=*/nullptr));
//   ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op2, batch_size, channels_2, channels_2, output_stride, /*threadpool=*/nullptr));

//   ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x32(op1, input1.data(), operator_output.data()));
//   ASSERT_EQ(
//     xnn_status_success,
//     xnn_setup_copy_nc_x32( op2, input2.data(), (float*) operator_output.data() + op1->channels));

//   ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
//   ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));

//   // Call subgraph API.
//   xnn_subgraph_t subgraph = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
//   std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

//   input1_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr, 0,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
//   ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

//   input2_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr, 1,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
//   ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

//   output_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 2,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
//   ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

//   ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

//   xnn_runtime_t runtime = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
//   ASSERT_NE(nullptr, runtime);
//   std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
//   std::array<xnn_external_value, 3> external = {
//     xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
//     xnn_external_value{output_id, subgraph_output.data()}};
//   ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
//   ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

//   // Check outputs match.
//   ASSERT_EQ(subgraph_output, operator_output);
// }

// TEST_F(Concatenate2TestF32, Reshape)
// {
//   ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

//   xnn_subgraph_t subgraph = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
//   std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

//   input1_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr, 0,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
//   ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

//   input2_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr, 1,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
//   ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

//   output_id = XNN_INVALID_NODE_ID;
//   ASSERT_EQ(
//     xnn_status_success, xnn_define_tensor_value(
//                           subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 2,
//                           /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
//   ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

//   ASSERT_EQ(
//     xnn_status_success,
//     xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));


//   ASSERT_EQ(subgraph->num_nodes, 1);
//   struct xnn_node* node = &subgraph->nodes[0];
//   ASSERT_EQ(node->type, xnn_node_type_concatenate2);
//   ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
//   ASSERT_EQ(node->num_inputs, 2);
//   ASSERT_EQ(node->inputs[0], input1_id);
//   ASSERT_EQ(node->inputs[1], input2_id);
//   ASSERT_EQ(node->num_outputs, 1);
//   ASSERT_EQ(node->outputs[0], output_id);
//   ASSERT_EQ(node->flags, 0);

//   xnn_runtime_t runtime = nullptr;
//   ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
//   ASSERT_NE(nullptr, runtime);
//   std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

//   ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr), xnn_status_success);

//   input1_dims[axis] += 1;
//   ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input1_id, input1_dims.size(), input1_dims.data()));

//   ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
//   const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
//   ASSERT_EQ(output_shape->dim[axis], input1_dims[axis] + input2_dims[axis]);
//   for (size_t i = 0; i < input1_dims.size(); ++i) {
//     if (i == axis) continue;
//     ASSERT_EQ(output_shape->dim[i], input1_dims[i]);
//   }

//   for (size_t i = 0; i < input1_dims.size(); ++i) {
//     if (i == axis) continue;
//     input1_dims[i] += 1;
//     input2_dims[i] += 1;
//     ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input1_id, input1_dims.size(), input1_dims.data()));
//     ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input2_id, input2_dims.size(), input2_dims.data()));

//     ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
//     ASSERT_EQ(output_shape->dim[axis], input1_dims[axis] + input2_dims[axis]);
//     for (size_t i = 0; i < input1_dims.size(); ++i) {
//       if (i == axis) continue;
//       ASSERT_EQ(output_shape->dim[i], input1_dims[i]);
//     }
//   }
//   for (size_t i = 0; i < input1_dims.size(); ++i) {
//     if (i == axis) continue;
//     input1_dims[i] -= 1;
//     input2_dims[i] -= 1;
//     ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input1_id, input1_dims.size(), input1_dims.data()));
//     ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input2_id, input2_dims.size(), input2_dims.data()));

//     ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
//     ASSERT_EQ(output_shape->dim[axis], input1_dims[axis] + input2_dims[axis]);
//     for (size_t i = 0; i < input1_dims.size(); ++i) {
//       if (i == axis) continue;
//       ASSERT_EQ(output_shape->dim[i], input1_dims[i]);
//     }
//   }
// }