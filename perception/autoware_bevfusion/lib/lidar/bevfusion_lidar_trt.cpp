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

#include "autoware/bevfusion/lidar/bevfusion_lidar_trt.hpp"

#include "autoware/bevfusion/bevfusion_config.hpp"
#include "autoware/bevfusion/preprocess/point_type.hpp"

#include <autoware/cuda_utils/cuda_utils.hpp>
#include <autoware/point_types/memory.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Contains the lidar inference branch and the lidar-only BEVFusion detector.
namespace autoware::bevfusion
{

LidarTrtBranch::LidarTrtBranch(
  const DensificationParam & densification_param, const BEVFusionConfig & base_config,
  const BEVFusionLidarConfig & lidar_config)
: base_config_(base_config), lidar_config_(lidar_config), densification_param_(densification_param)
{
}

void LidarTrtBranch::initialize(cudaStream_t stream)
{
  stream_ = stream;

  vg_ptr_ =
    std::make_unique<VoxelGenerator>(densification_param_, base_config_, lidar_config_, stream_);

  // point cloud to voxels
  voxel_features_size_ = lidar_config_.max_num_voxels_ * lidar_config_.max_points_per_voxel_ *
                         lidar_config_.num_point_feature_size_;
  voxel_coords_size_ = 3 * lidar_config_.max_num_voxels_;

  voxel_features_d_ = autoware::cuda_utils::make_unique<float[]>(voxel_features_size_);
  voxel_coords_d_ = autoware::cuda_utils::make_unique<std::int32_t[]>(voxel_coords_size_);
  num_points_per_voxel_d_ =
    autoware::cuda_utils::make_unique<std::int32_t[]>(lidar_config_.max_num_voxels_);
  points_d_ = autoware::cuda_utils::make_unique<float[]>(
    lidar_config_.cloud_capacity_ * lidar_config_.num_point_feature_size_);

  pre_ptr_ = std::make_unique<PreprocessCuda>(base_config_, lidar_config_, stream_, true);
}

void LidarTrtBranch::addNetworkIO(
  std::vector<autoware::tensorrt_common::NetworkIO> & network_io) const
{
  network_io.emplace_back(
    "voxels",
    nvinfer1::Dims{
      3, {-1, lidar_config_.max_points_per_voxel_, lidar_config_.num_point_feature_size_}});
  network_io.emplace_back("num_points_per_voxel", nvinfer1::Dims{1, {-1}});
  network_io.emplace_back("coors", nvinfer1::Dims{2, {-1, BEVFusionConfig::kNum3DCoords}});
}

void LidarTrtBranch::addProfileDims(
  std::vector<autoware::tensorrt_common::ProfileDims> & profile_dims) const
{
  profile_dims.emplace_back(
    "voxels",
    nvinfer1::Dims{
      3,
      {lidar_config_.voxels_num_[0], lidar_config_.max_points_per_voxel_,
       lidar_config_.num_point_feature_size_}},
    nvinfer1::Dims{
      3,
      {lidar_config_.voxels_num_[1], lidar_config_.max_points_per_voxel_,
       lidar_config_.num_point_feature_size_}},
    nvinfer1::Dims{
      3,
      {lidar_config_.voxels_num_[2], lidar_config_.max_points_per_voxel_,
       lidar_config_.num_point_feature_size_}});

  profile_dims.emplace_back(
    "num_points_per_voxel", nvinfer1::Dims{1, {lidar_config_.voxels_num_[0]}},
    nvinfer1::Dims{1, {lidar_config_.voxels_num_[1]}},
    nvinfer1::Dims{1, {lidar_config_.voxels_num_[2]}});

  profile_dims.emplace_back(
    "coors", nvinfer1::Dims{2, {lidar_config_.voxels_num_[0], BEVFusionConfig::kNum3DCoords}},
    nvinfer1::Dims{2, {lidar_config_.voxels_num_[1], BEVFusionConfig::kNum3DCoords}},
    nvinfer1::Dims{2, {lidar_config_.voxels_num_[2], BEVFusionConfig::kNum3DCoords}});
}

void LidarTrtBranch::setTensorAddresses(autoware::tensorrt_common::TrtCommon * network)
{
  network->setTensorAddress("voxels", voxel_features_d_.get());
  network->setTensorAddress("num_points_per_voxel", num_points_per_voxel_d_.get());
  network->setTensorAddress("coors", voxel_coords_d_.get());
}

void LidarTrtBranch::configureInputShapes(
  autoware::tensorrt_common::TrtCommon * network, std::int64_t num_voxels) const
{
  network->setInputShape(
    "voxels",
    nvinfer1::Dims{
      3, {num_voxels, lidar_config_.max_points_per_voxel_, lidar_config_.num_point_feature_size_}});
  network->setInputShape("num_points_per_voxel", nvinfer1::Dims{1, {num_voxels}});
  network->setInputShape("coors", nvinfer1::Dims{2, {num_voxels, BEVFusionConfig::kNum3DCoords}});
}

bool LidarTrtBranch::validatePointCloud(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr) const
{
  if (!autoware::point_types::is_data_layout_compatible_with_point_xyzirc(pc_msg_ptr->fields)) {
    RCLCPP_ERROR(rclcpp::get_logger("bevfusion"), "Invalid point type. Skipping detection.");
    return false;
  }

  if (pc_msg_ptr->height * pc_msg_ptr->width == 0) {
    RCLCPP_ERROR(rclcpp::get_logger("bevfusion"), "Empty pointcloud. Skipping detection.");
    return false;
  }

  return true;
}

bool LidarTrtBranch::enqueuePointCloud(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
  const tf2_ros::Buffer & tf_buffer)
{
  return vg_ptr_->enqueuePointCloud(pc_msg_ptr, tf_buffer);
}

void LidarTrtBranch::clearDeviceMemory(cudaStream_t stream)
{
  using autoware::cuda_utils::clear_async;

  clear_async(voxel_features_d_.get(), voxel_features_size_, stream);
  clear_async(voxel_coords_d_.get(), voxel_coords_size_, stream);
  clear_async(num_points_per_voxel_d_.get(), lidar_config_.max_num_voxels_, stream);
  clear_async(
    points_d_.get(), lidar_config_.cloud_capacity_ * lidar_config_.num_point_feature_size_, stream);
}

std::size_t LidarTrtBranch::generateSweepPoints()
{
  return vg_ptr_->generateSweepPoints(points_d_);
}

std::int64_t LidarTrtBranch::voxelize(std::size_t num_points, bool & is_num_voxels_within_range)
{
  auto num_voxels = static_cast<std::int64_t>(pre_ptr_->generateVoxels(
    points_d_.get(), num_points, voxel_features_d_.get(), voxel_coords_d_.get(),
    num_points_per_voxel_d_.get()));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  if (num_voxels < lidar_config_.min_num_voxels_) {
    RCLCPP_ERROR_STREAM(
      rclcpp::get_logger("bevfusion"), "Too few voxels ("
                                         << num_voxels << ") for the actual optimization profile ("
                                         << lidar_config_.min_num_voxels_ << ")");
    return -1;
  }

  // Check the actual number of pillars after inference to avoid unnecessary synchronization.
  if (num_voxels >= lidar_config_.max_num_voxels_) {
    rclcpp::Clock clock{RCL_ROS_TIME};
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("bevfusion"), clock, 1000,
      "The actual number of voxels (%lu) exceeds its maximum value (%zu). "
      "Please considering increasing it since it may limit the detection performance. Clipping for "
      "now.",
      num_voxels, lidar_config_.max_num_voxels_);
    is_num_voxels_within_range = false;
    num_voxels = lidar_config_.max_num_voxels_;
  }

  return num_voxels;
}

BEVFusionLidarTRT::BEVFusionLidarTRT(
  const TrtBEVFusionConfig & trt_config, const DensificationParam & densification_param,
  const BEVFusionConfig & base_config, const BEVFusionLidarConfig & lidar_config)
: BEVFusionTRT(base_config)
{
  lidar_branch_ = std::make_unique<LidarTrtBranch>(densification_param, base_config, lidar_config);
  lidar_branch_->initialize(stream_);
  initOutputs(lidar_branch_->config());
  initTrt(trt_config);
}

void BEVFusionLidarTRT::initTrt(const TrtBEVFusionConfig & trt_config)
{
  std::vector<autoware::tensorrt_common::NetworkIO> network_io;
  lidar_branch_->addNetworkIO(network_io);
  addOutputNetworkIO(network_io);

  std::vector<autoware::tensorrt_common::ProfileDims> profile_dims;
  lidar_branch_->addProfileDims(profile_dims);

  createNetwork(trt_config, std::move(network_io), std::move(profile_dims));

  lidar_branch_->setTensorAddresses(network_trt_ptr_.get());
  setOutputTensorAddresses();
}

bool BEVFusionLidarTRT::preProcess(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
  const std::vector<std::unique_ptr<CameraData>> &, const std::vector<float> &,
  const tf2_ros::Buffer & tf_buffer, bool & is_num_voxels_within_range)
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

  return true;
}

}  //  namespace autoware::bevfusion
