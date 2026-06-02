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

#include "autoware/bevfusion/camera_lidar/bevfusion_camera_lidar_trt.hpp"

#include "autoware/bevfusion/camera/bevfusion_camera_trt.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_trt.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Contains the camera-lidar fusion BEVFusion detector, which composes a lidar branch and a camera
// branch as independent objects and implements its own engine setup / pre-processing using both.
namespace autoware::bevfusion
{

BEVFusionCameraLidarTRT::BEVFusionCameraLidarTRT(
  const TrtBEVFusionConfig & trt_config, const DensificationParam & densification_param,
  const BEVFusionConfig & base_config, const BEVFusionLidarConfig & lidar_config,
  const BEVFusionCameraConfig & camera_config)
: BEVFusionTRT(base_config)
{
  // The lidar and camera branches are independent objects, each initialized separately while
  // sharing the same CUDA stream owned by the base.
  lidar_branch_ = std::make_unique<LidarTrtBranch>(densification_param, base_config, lidar_config);
  camera_branch_ = std::make_unique<CameraTrtBranch>(base_config, lidar_config, camera_config);
  lidar_branch_->initialize(stream_);
  camera_branch_->initialize(stream_);
  initOutputs(lidar_branch_->config());
  initTrt(trt_config);
}

void BEVFusionCameraLidarTRT::initTrt(const TrtBEVFusionConfig & trt_config)
{
  // The camera branch builds its own image backbone engine when sensor fusion is enabled.
  camera_branch_->setupImageBackbone(trt_config);

  std::vector<autoware::tensorrt_common::NetworkIO> network_io;
  lidar_branch_->addNetworkIO(network_io);
  camera_branch_->addNetworkIO(network_io);
  addOutputNetworkIO(network_io);

  std::vector<autoware::tensorrt_common::ProfileDims> profile_dims;
  lidar_branch_->addProfileDims(profile_dims);
  camera_branch_->addProfileDims(profile_dims);

  createNetwork(trt_config, std::move(network_io), std::move(profile_dims));

  lidar_branch_->setTensorAddresses(network_trt_ptr_.get());
  // The camera fusion inputs reuse the lidar raw points buffer for the "points" tensor.
  camera_branch_->setTensorAddresses(network_trt_ptr_.get(), lidar_branch_->points_d());
  setOutputTensorAddresses();
}

bool BEVFusionCameraLidarTRT::preProcess(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
  const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
  const std::vector<float> & camera_masks, const tf2_ros::Buffer & tf_buffer,
  bool & is_num_voxels_within_range)
{
  is_num_voxels_within_range = true;

  if (!lidar_branch_->validatePointCloud(pc_msg_ptr)) {
    return false;
  }

  if (!lidar_branch_->enqueuePointCloud(pc_msg_ptr, tf_buffer)) {
    return false;
  }

  clearOutputMemory();
  lidar_branch_->clearDeviceMemory(stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  const bool sensor_fusion = camera_branch_->sensor_fusion();

  // Process images if sensor fusion is enabled
  if (sensor_fusion) {
    // Check if all image cameras are ready
    if (!camera_branch_->checkImageCameraMatricesReady(camera_data_ptrs)) {
      rclcpp::Clock clock{RCL_ROS_TIME};
      RCLCPP_WARN_THROTTLE(
        rclcpp::get_logger("bevfusion"), clock, 1000,
        "Image and camera matrices are not ready. Skipping pre-processing.");
      return false;
    }

    // Process Images
    if (!camera_branch_->processImages(camera_data_ptrs, camera_masks)) {
      RCLCPP_ERROR(
        rclcpp::get_logger("bevfusion"), "Failed to process images. Skipping detection.");
      return false;
    }
  }

  const auto num_points = lidar_branch_->generateSweepPoints();
  if (num_points == 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("bevfusion"),
      "Empty sweep points (check the capacity of the buffer). Skipping detection.");
    return false;
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  const std::int64_t num_voxels = lidar_branch_->voxelize(num_points, is_num_voxels_within_range);
  if (num_voxels < 0) {
    return false;
  }

  lidar_branch_->configureInputShapes(network_trt_ptr_.get(), num_voxels);

  if (sensor_fusion) {
    camera_branch_->configureInputShapes(network_trt_ptr_.get(), num_points);
    camera_branch_->debugSaveRoi();
  }

  return true;
}

bool BEVFusionCameraLidarTRT::inference()
{
  // Fusion model: run image backbone first, then the shared main network.
  if (camera_branch_->sensor_fusion()) {
    if (!camera_branch_->runImageBackbone()) {
      return false;
    }
  }

  return runMainNetwork();
}

void BEVFusionCameraLidarTRT::setIntrinsicsExtrinsics(
  std::vector<sensor_msgs::msg::CameraInfo> & camera_info_vector,
  std::vector<Matrix4fRowM> & lidar2camera_vector)
{
  camera_branch_->setIntrinsicsExtrinsics(camera_info_vector, lidar2camera_vector);
}

}  //  namespace autoware::bevfusion
