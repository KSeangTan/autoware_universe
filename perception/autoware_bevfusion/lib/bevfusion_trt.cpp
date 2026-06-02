// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/bevfusion/bevfusion_trt.hpp"

#include "autoware/bevfusion/bevfusion_config.hpp"

#include <autoware/cuda_utils/cuda_utils.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Contains the common base of the BEVFusion detector family: the shared main TensorRT engine, the
// output buffers and post-processing, and the detection skeleton. The branch-specific engine setup
// (initTrt) and pre-processing (preProcess) are implemented by the per-modality detectors; the base
// only provides reusable output/engine helpers and never accesses the branches itself.
namespace autoware::bevfusion
{

BEVFusionTRT::BEVFusionTRT(const BEVFusionConfig & base_config) : base_config_(base_config)
{
  // Create and init cuda streams
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));

  stop_watch_ptr_ = std::make_unique<autoware_utils_system::StopWatch<std::chrono::milliseconds>>();
  stop_watch_ptr_->tic("processing/inner");
}

BEVFusionTRT::~BEVFusionTRT()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

void BEVFusionTRT::initOutputs(const BEVFusionLidarConfig & lidar_config)
{
  // output of TRT -- input of post-process
  bbox_pred_size_ = base_config_.num_proposals_ * base_config_.num_box_values_;
  label_pred_output_d_ =
    autoware::cuda_utils::make_unique<std::int64_t[]>(base_config_.num_proposals_);
  bbox_pred_output_d_ = autoware::cuda_utils::make_unique<float[]>(bbox_pred_size_);
  score_output_d_ = autoware::cuda_utils::make_unique<float[]>(base_config_.num_proposals_);

  post_ptr_ = std::make_unique<PostprocessCuda>(base_config_, lidar_config, stream_);
}

void BEVFusionTRT::addOutputNetworkIO(
  std::vector<autoware::tensorrt_common::NetworkIO> & network_io) const
{
  network_io.emplace_back(
    "bbox_pred", nvinfer1::Dims{2, {base_config_.num_box_values_, base_config_.num_proposals_}});
  network_io.emplace_back("score", nvinfer1::Dims{1, {base_config_.num_proposals_}});
  network_io.emplace_back("label_pred", nvinfer1::Dims{1, {base_config_.num_proposals_}});
}

void BEVFusionTRT::createNetwork(
  const TrtBEVFusionConfig & trt_config,
  std::vector<autoware::tensorrt_common::NetworkIO> network_io,
  std::vector<autoware::tensorrt_common::ProfileDims> profile_dims)
{
  auto network_io_ptr =
    std::make_unique<std::vector<autoware::tensorrt_common::NetworkIO>>(std::move(network_io));
  auto profile_dims_ptr =
    std::make_unique<std::vector<autoware::tensorrt_common::ProfileDims>>(std::move(profile_dims));

  network_trt_ptr_ = std::make_unique<autoware::tensorrt_common::TrtCommon>(
    trt_config.common, std::make_shared<autoware::tensorrt_common::Profiler>(),
    std::vector<std::string>{base_config_.plugins_path_});

  if (!network_trt_ptr_->setup(std::move(profile_dims_ptr), std::move(network_io_ptr))) {
    throw std::runtime_error("Failed to setup TRT engine.");
  }
}

void BEVFusionTRT::setOutputTensorAddresses()
{
  network_trt_ptr_->setTensorAddress("label_pred", label_pred_output_d_.get());
  network_trt_ptr_->setTensorAddress("bbox_pred", bbox_pred_output_d_.get());
  network_trt_ptr_->setTensorAddress("score", score_output_d_.get());
}

bool BEVFusionTRT::detect(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
  const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
  const std::vector<float> & camera_masks, const tf2_ros::Buffer & tf_buffer,
  std::vector<Box3D> & det_boxes3d, std::unordered_map<std::string, double> & proc_timing,
  bool & is_num_voxels_within_range)
{
  stop_watch_ptr_->toc("processing/inner", true);
  if (!preProcess(
        pc_msg_ptr, camera_data_ptrs, camera_masks, tf_buffer, is_num_voxels_within_range)) {
    RCLCPP_ERROR(rclcpp::get_logger("bevfusion"), "Pre-process failed. Skipping detection.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/preprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  if (!inference()) {
    RCLCPP_ERROR(rclcpp::get_logger("bevfusion"), "Inference failed. Skipping detection.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/inference_ms", stop_watch_ptr_->toc("processing/inner", true));

  if (!postProcess(det_boxes3d)) {
    RCLCPP_ERROR(rclcpp::get_logger("bevfusion"), "Post-process failed. Skipping detection");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/postprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  return true;
}

void BEVFusionTRT::setIntrinsicsExtrinsics(
  std::vector<sensor_msgs::msg::CameraInfo> &, std::vector<Matrix4fRowM> &)
{
  // No cameras in the base / lidar-only detector.
}

void BEVFusionTRT::clearOutputMemory()
{
  using autoware::cuda_utils::clear_async;

  // TODO(knzo25): these should be able to be removed as they are filled by TensorRT
  clear_async(label_pred_output_d_.get(), base_config_.num_proposals_, stream_);
  clear_async(bbox_pred_output_d_.get(), bbox_pred_size_, stream_);
  clear_async(score_output_d_.get(), base_config_.num_proposals_, stream_);
}

bool BEVFusionTRT::inference()
{
  // The base detector only runs the shared main network. The camera / camera-lidar detectors
  // override this to run the image backbone first.
  return runMainNetwork();
}

bool BEVFusionTRT::runMainNetwork()
{
  auto status = network_trt_ptr_->enqueueV3(stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  return status;
}

bool BEVFusionTRT::postProcess(std::vector<Box3D> & det_boxes3d)
{
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(post_ptr_->generateDetectedBoxes3D_launch(
    label_pred_output_d_.get(), bbox_pred_output_d_.get(), score_output_d_.get(), det_boxes3d,
    stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  return true;
}

}  //  namespace autoware::bevfusion
