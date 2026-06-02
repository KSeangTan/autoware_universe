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

#include "autoware/bevfusion/camera/bevfusion_camera_trt.hpp"

#include "autoware/bevfusion/bevfusion_config.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_trt.hpp"
#include "autoware/bevfusion/preprocess/precomputed_features.hpp"

#include <autoware/cuda_utils/cuda_utils.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Contains the camera inference branch (image backbone engine, camera buffers and lidar-to-image
// geometry) and the camera-only BEVFusion detector.
namespace autoware::bevfusion
{

CameraTrtBranch::CameraTrtBranch(
  const BEVFusionConfig & base_config, const BEVFusionLidarConfig & lidar_config,
  const BEVFusionCameraConfig & camera_config)
: base_config_(base_config), lidar_config_(lidar_config), camera_config_(camera_config)
{
}

void CameraTrtBranch::initialize(cudaStream_t stream)
{
  stream_ = stream;

  if (!camera_config_.sensor_fusion_) {
    return;
  }

  lidar2image_d_ = autoware::cuda_utils::make_unique<float[]>(
    camera_config_.num_cameras_ * BEVFusionConfig::kTransformMatrixDim *
    BEVFusionConfig::kTransformMatrixDim);
  std::int64_t num_geom_feats = camera_config_.num_cameras_ * camera_config_.features_height_ *
                                camera_config_.features_width_ * camera_config_.num_depth_features_;
  geom_feats_d_ = autoware::cuda_utils::make_unique<std::int32_t[]>(
    BEVFusionConfig::kTransformMatrixDim * num_geom_feats);
  kept_d_ = autoware::cuda_utils::make_unique<std::uint8_t[]>(num_geom_feats);
  ranks_d_ = autoware::cuda_utils::make_unique<std::int64_t[]>(num_geom_feats);
  indices_d_ = autoware::cuda_utils::make_unique<std::int64_t[]>(num_geom_feats);

  // image branch
  roi_tensor_d_ = autoware::cuda_utils::make_unique<std::uint8_t[]>(
    camera_config_.num_cameras_ * camera_config_.roi_height_ * camera_config_.roi_width_ *
    BEVFusionConfig::kNumRGBChannels);
  camera_masks_d_ = autoware::cuda_utils::make_unique<float[]>(camera_config_.num_cameras_);

  // buffers for fusion model with separate image backbone
  image_feats_d_ = autoware::cuda_utils::make_unique<float[]>(
    camera_config_.num_cameras_ * camera_config_.image_feature_channel_ *
    camera_config_.features_height_ * camera_config_.features_width_);
  img_aug_matrix_d_ = autoware::cuda_utils::make_unique<float[]>(
    camera_config_.num_cameras_ * BEVFusionConfig::kTransformMatrixDim *
    BEVFusionConfig::kTransformMatrixDim);
}

void CameraTrtBranch::setupImageBackbone(const TrtBEVFusionConfig & trt_config)
{
  if (!camera_config_.sensor_fusion_ || !trt_config.image_backbone.has_value()) {
    return;
  }

  std::vector<autoware::tensorrt_common::NetworkIO> image_backbone_io;
  image_backbone_io.emplace_back(
    "imgs", nvinfer1::Dims{
              4,
              {-1, BEVFusionConfig::kNumRGBChannels, camera_config_.roi_height_,
               camera_config_.roi_width_}});
  image_backbone_io.emplace_back(
    "image_feats", nvinfer1::Dims{
                     4,
                     {-1, camera_config_.image_feature_channel_, camera_config_.features_height_,
                      camera_config_.features_width_}});

  std::vector<autoware::tensorrt_common::ProfileDims> image_backbone_profiles;
  image_backbone_profiles.emplace_back(
    "imgs",
    nvinfer1::Dims{
      4,
      {1, BEVFusionConfig::kNumRGBChannels, camera_config_.roi_height_, camera_config_.roi_width_}},
    nvinfer1::Dims{
      4,
      {camera_config_.num_cameras_, BEVFusionConfig::kNumRGBChannels, camera_config_.roi_height_,
       camera_config_.roi_width_}},
    nvinfer1::Dims{
      4,
      {camera_config_.num_cameras_, BEVFusionConfig::kNumRGBChannels, camera_config_.roi_height_,
       camera_config_.roi_width_}});

  auto image_backbone_io_ptr =
    std::make_unique<std::vector<autoware::tensorrt_common::NetworkIO>>(image_backbone_io);
  auto image_backbone_profiles_ptr =
    std::make_unique<std::vector<autoware::tensorrt_common::ProfileDims>>(image_backbone_profiles);

  image_backbone_trt_ptr_ = std::make_unique<autoware::tensorrt_common::TrtCommon>(
    trt_config.image_backbone.value(), std::make_shared<autoware::tensorrt_common::Profiler>(),
    std::vector<std::string>{base_config_.plugins_path_});

  if (!image_backbone_trt_ptr_->setup(
        std::move(image_backbone_profiles_ptr), std::move(image_backbone_io_ptr))) {
    throw std::runtime_error("Failed to setup image backbone TRT engine.");
  }

  image_backbone_trt_ptr_->setTensorAddress("imgs", roi_tensor_d_.get());
  image_backbone_trt_ptr_->setTensorAddress("image_feats", image_feats_d_.get());
}

void CameraTrtBranch::addNetworkIO(
  std::vector<autoware::tensorrt_common::NetworkIO> & network_io) const
{
  if (!camera_config_.sensor_fusion_) {
    return;
  }

  network_io.emplace_back("points", nvinfer1::Dims{2, {-1, lidar_config_.num_point_feature_size_}});
  network_io.emplace_back(
    "image_feats", nvinfer1::Dims{
                     4,
                     {-1, camera_config_.image_feature_channel_, camera_config_.features_height_,
                      camera_config_.features_width_}});
  network_io.emplace_back(
    "img_aug_matrix",
    nvinfer1::Dims{
      3, {-1, BEVFusionConfig::kTransformMatrixDim, BEVFusionConfig::kTransformMatrixDim}});
  network_io.emplace_back(
    "lidar2image",
    nvinfer1::Dims{
      3, {-1, BEVFusionConfig::kTransformMatrixDim, BEVFusionConfig::kTransformMatrixDim}});

  network_io.emplace_back(
    "geom_feats", nvinfer1::Dims{2, {-1, BEVFusionConfig::kTransformMatrixDim}});
  network_io.emplace_back("kept", nvinfer1::Dims{1, {-1}});
  network_io.emplace_back("ranks", nvinfer1::Dims{1, {-1}});
  network_io.emplace_back("indices", nvinfer1::Dims{1, {-1}});
}

void CameraTrtBranch::addProfileDims(
  std::vector<autoware::tensorrt_common::ProfileDims> & profile_dims) const
{
  if (!camera_config_.sensor_fusion_) {
    return;
  }

  profile_dims.emplace_back(
    "points",
    nvinfer1::Dims{2, {lidar_config_.voxels_num_[0], lidar_config_.num_point_feature_size_}},
    nvinfer1::Dims{2, {lidar_config_.voxels_num_[1], lidar_config_.num_point_feature_size_}},
    nvinfer1::Dims{2, {lidar_config_.cloud_capacity_, lidar_config_.num_point_feature_size_}});

  profile_dims.emplace_back(
    "image_feats",
    nvinfer1::Dims{
      4,
      {1, camera_config_.image_feature_channel_, camera_config_.features_height_,
       camera_config_.features_width_}},
    nvinfer1::Dims{
      4,
      {camera_config_.num_cameras_, camera_config_.image_feature_channel_,
       camera_config_.features_height_, camera_config_.features_width_}},
    nvinfer1::Dims{
      4,
      {camera_config_.num_cameras_, camera_config_.image_feature_channel_,
       camera_config_.features_height_, camera_config_.features_width_}});

  profile_dims.emplace_back(
    "img_aug_matrix",
    nvinfer1::Dims{
      3, {1, BEVFusionConfig::kTransformMatrixDim, BEVFusionConfig::kTransformMatrixDim}},
    nvinfer1::Dims{
      3,
      {camera_config_.num_cameras_, BEVFusionConfig::kTransformMatrixDim,
       BEVFusionConfig::kTransformMatrixDim}},
    nvinfer1::Dims{
      3,
      {camera_config_.num_cameras_, BEVFusionConfig::kTransformMatrixDim,
       BEVFusionConfig::kTransformMatrixDim}});

  profile_dims.emplace_back(
    "lidar2image",
    nvinfer1::Dims{
      3, {1, BEVFusionConfig::kTransformMatrixDim, BEVFusionConfig::kTransformMatrixDim}},
    nvinfer1::Dims{
      3,
      {camera_config_.num_cameras_, BEVFusionConfig::kTransformMatrixDim,
       BEVFusionConfig::kTransformMatrixDim}},
    nvinfer1::Dims{
      3,
      {camera_config_.num_cameras_, BEVFusionConfig::kTransformMatrixDim,
       BEVFusionConfig::kTransformMatrixDim}});

  const std::int64_t num_geom_feats =
    camera_config_.num_cameras_ * camera_config_.features_height_ * camera_config_.features_width_ *
    camera_config_.num_depth_features_;

  profile_dims.emplace_back(
    "geom_feats", nvinfer1::Dims{2, {0, BEVFusionConfig::kTransformMatrixDim}},
    nvinfer1::Dims{2, {num_geom_feats, BEVFusionConfig::kTransformMatrixDim}},
    nvinfer1::Dims{2, {num_geom_feats, BEVFusionConfig::kTransformMatrixDim}});

  profile_dims.emplace_back(
    "kept", nvinfer1::Dims{1, {0}}, nvinfer1::Dims{1, {num_geom_feats}},
    nvinfer1::Dims{1, {num_geom_feats}});

  profile_dims.emplace_back(
    "ranks", nvinfer1::Dims{1, {0}}, nvinfer1::Dims{1, {num_geom_feats}},
    nvinfer1::Dims{1, {num_geom_feats}});

  profile_dims.emplace_back(
    "indices", nvinfer1::Dims{1, {0}}, nvinfer1::Dims{1, {num_geom_feats}},
    nvinfer1::Dims{1, {num_geom_feats}});
}

void CameraTrtBranch::setTensorAddresses(
  autoware::tensorrt_common::TrtCommon * network, float * points_d)
{
  if (!camera_config_.sensor_fusion_) {
    return;
  }

  network->setTensorAddress("points", points_d);
  network->setTensorAddress("image_feats", image_feats_d_.get());
  network->setTensorAddress("img_aug_matrix", img_aug_matrix_d_.get());
  network->setTensorAddress("lidar2image", lidar2image_d_.get());
  network->setTensorAddress("geom_feats", geom_feats_d_.get());
  network->setTensorAddress("kept", kept_d_.get());
  network->setTensorAddress("ranks", ranks_d_.get());
  network->setTensorAddress("indices", indices_d_.get());
}

void CameraTrtBranch::configureInputShapes(
  autoware::tensorrt_common::TrtCommon * network, std::size_t num_points) const
{
  if (!camera_config_.sensor_fusion_) {
    return;
  }

  network->setInputShape(
    "points", nvinfer1::Dims{
                2, {static_cast<std::int64_t>(num_points), lidar_config_.num_point_feature_size_}});

  // Separate image backbone: set image_feats and img_aug_matrix inputs
  network->setInputShape(
    "image_feats", nvinfer1::Dims{
                     4,
                     {camera_config_.num_cameras_, camera_config_.image_feature_channel_,
                      camera_config_.features_height_, camera_config_.features_width_}});
  network->setInputShape(
    "img_aug_matrix", nvinfer1::Dims{
                        3,
                        {camera_config_.num_cameras_, BEVFusionConfig::kTransformMatrixDim,
                         BEVFusionConfig::kTransformMatrixDim}});
  network->setInputShape(
    "lidar2image", nvinfer1::Dims{
                     3,
                     {camera_config_.num_cameras_, BEVFusionConfig::kTransformMatrixDim,
                      BEVFusionConfig::kTransformMatrixDim}});

  network->setInputShape(
    "geom_feats", nvinfer1::Dims{2, {num_ranks_, BEVFusionConfig::kTransformMatrixDim}});
  network->setInputShape("kept", nvinfer1::Dims{1, {num_kept_}});
  network->setInputShape("ranks", nvinfer1::Dims{1, {num_ranks_}});
  network->setInputShape("indices", nvinfer1::Dims{1, {num_indices_}});
}

void CameraTrtBranch::setIntrinsicsExtrinsics(
  std::vector<sensor_msgs::msg::CameraInfo> & camera_info_vector,
  std::vector<Matrix4fRowM> & lidar2camera_vector)
{
  roi_start_y_vector_.clear();
  img_aug_matrices_.clear();

  for (std::int64_t i = 0; i < camera_config_.num_cameras_; i++) {
    int crop_h = static_cast<int>(camera_config_.resized_height_ - camera_config_.roi_height_);
    int crop_w = std::max(
                   0, static_cast<int>(camera_config_.resized_width_) -
                        static_cast<int>(camera_config_.roi_width_)) /
                 2;

    Matrix4fRowM img_aug_matrix = Matrix4fRowM::Identity();
    img_aug_matrix(0, 0) = camera_config_.img_aug_scale_x_;
    img_aug_matrix(1, 1) = camera_config_.img_aug_scale_y_;
    img_aug_matrix(0, 3) = -static_cast<float>(crop_w);
    img_aug_matrix(1, 3) = -static_cast<float>(crop_h);

    img_aug_matrices_.push_back(img_aug_matrix);
    roi_start_y_vector_.push_back(crop_h);
  }

  auto [lidar2images_flattened, geom_feats, kept, ranks, indices] =
    precomputeFeatures(lidar2camera_vector, img_aug_matrices_, camera_info_vector, camera_config_);

  assert(
    static_cast<std::int64_t>(lidar2images_flattened.size()) ==
    camera_config_.num_cameras_ * BEVFusionConfig::kTransformMatrixDim *
      BEVFusionConfig::kTransformMatrixDim);

  assert(
    static_cast<std::int64_t>(geom_feats.size()) <=
    camera_config_.num_cameras_ * BEVFusionConfig::kTransformMatrixDim *
      camera_config_.features_height_ * camera_config_.features_width_ *
      camera_config_.num_depth_features_);
  assert(
    static_cast<std::int64_t>(kept.size()) ==
    camera_config_.num_cameras_ * camera_config_.features_height_ * camera_config_.features_width_ *
      camera_config_.num_depth_features_);
  assert(
    static_cast<std::int64_t>(ranks.size()) <=
    camera_config_.num_cameras_ * camera_config_.features_height_ * camera_config_.features_width_ *
      camera_config_.num_depth_features_);
  assert(
    static_cast<std::int64_t>(indices.size()) <=
    camera_config_.num_cameras_ * camera_config_.features_height_ * camera_config_.features_width_ *
      camera_config_.num_depth_features_);

  num_geom_feats_ = static_cast<std::int64_t>(geom_feats.size());
  num_kept_ = static_cast<std::int64_t>(kept.size());
  num_ranks_ = static_cast<std::int64_t>(ranks.size());
  num_indices_ = static_cast<std::int64_t>(indices.size());

  assert(num_geom_feats_ == BEVFusionConfig::kTransformMatrixDim * num_ranks_);
  assert(num_ranks_ == num_indices_);

  CHECK_CUDA_ERROR(cudaMemcpy(
    lidar2image_d_.get(), lidar2images_flattened.data(),
    camera_config_.num_cameras_ * BEVFusionConfig::kTransformMatrixDim *
      BEVFusionConfig::kTransformMatrixDim * sizeof(float),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(
    geom_feats_d_.get(), geom_feats.data(), num_geom_feats_ * sizeof(std::int32_t),
    cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(
    kept_d_.get(), kept.data(), num_kept_ * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(
    ranks_d_.get(), ranks.data(), num_ranks_ * sizeof(std::int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(
    indices_d_.get(), indices.data(), num_indices_ * sizeof(std::int64_t), cudaMemcpyHostToDevice));

  // Copy img_aug_matrix data for fusion model (fusion model always uses separate image backbone)
  // Each Matrix4fRowM is contiguous, copy each matrix directly to its position in device memory
  if (camera_config_.sensor_fusion_) {
    const std::size_t matrix_size =
      BEVFusionConfig::kTransformMatrixDim * BEVFusionConfig::kTransformMatrixDim;

    for (std::int64_t i = 0; i < camera_config_.num_cameras_; i++) {
      CHECK_CUDA_ERROR(cudaMemcpy(
        img_aug_matrix_d_.get() + i * matrix_size, img_aug_matrices_[i].data(),
        matrix_size * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
}

bool CameraTrtBranch::checkImageCameraMatricesReady(
  const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs) const
{
  for (std::int64_t camera_id = 0; camera_id < camera_config_.num_cameras_; camera_id++) {
    if (!camera_data_ptrs[camera_id]->is_camera_matrices_ready()) {
      return false;
    }
  }
  return true;
}

bool CameraTrtBranch::processImages(
  const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
  const std::vector<float> & camera_masks)
{
  for (std::int64_t camera_id = 0; camera_id < camera_config_.num_cameras_; camera_id++) {
    // Check if the image encoding is supported
    if (!camera_data_ptrs[camera_id]->is_image_encoding_supported()) {
      rclcpp::Clock clock{RCL_ROS_TIME};
      RCLCPP_WARN_THROTTLE(
        rclcpp::get_logger("bevfusion"), clock, 1000,
        "Only RGB8 encoding is supported, and the image encoding is %s.",
        camera_data_ptrs[camera_id]->image_msg()->encoding.c_str());
      return false;
    }
    // Preprocess the image
    auto roi_tensor_offset = camera_data_ptrs[camera_id]->output_img_offset();
    if (!camera_data_ptrs[camera_id]->preprocess_image(&roi_tensor_d_[roi_tensor_offset])) {
      return false;
    }
  }

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    camera_masks_d_.get(), camera_masks.data(), camera_config_.num_cameras_ * sizeof(float),
    cudaMemcpyHostToDevice, stream_));

  for (std::int64_t camera_id = 0; camera_id < camera_config_.num_cameras_; camera_id++) {
    CHECK_CUDA_ERROR(camera_data_ptrs[camera_id]->sync_cuda_stream());
  }
  return true;
}

bool CameraTrtBranch::runImageBackbone()
{
  image_backbone_trt_ptr_->setInputShape(
    "imgs", nvinfer1::Dims{
              4,
              {camera_config_.num_cameras_, BEVFusionConfig::kNumRGBChannels,
               camera_config_.roi_height_, camera_config_.roi_width_}});

  auto image_status = image_backbone_trt_ptr_->enqueueV3(stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  return image_status;
}

void CameraTrtBranch::debugSaveRoi()
{
  if (!camera_config_.sensor_fusion_) {
    return;
  }

  std::vector<uint8_t> roi_host_data(
    camera_config_.roi_height_ * camera_config_.roi_width_ * BEVFusionConfig::kNumRGBChannels);
  for (std::int64_t camera_id = 0; camera_id < camera_config_.num_cameras_; camera_id++) {
    CHECK_CUDA_ERROR(cudaMemcpy(
      roi_host_data.data(),
      &roi_tensor_d_
        [camera_id * camera_config_.roi_height_ * camera_config_.roi_width_ *
         BEVFusionConfig::kNumRGBChannels],
      camera_config_.roi_height_ * camera_config_.roi_width_ * BEVFusionConfig::kNumRGBChannels,
      cudaMemcpyDeviceToHost));
  }
}

BEVFusionCameraTRT::BEVFusionCameraTRT(
  const TrtBEVFusionConfig & trt_config, const DensificationParam & densification_param,
  const BEVFusionConfig & base_config, const BEVFusionLidarConfig & lidar_config,
  const BEVFusionCameraConfig & camera_config)
: BEVFusionTRT(base_config)
{
  lidar_branch_ = std::make_unique<LidarTrtBranch>(densification_param, base_config, lidar_config);
  camera_branch_ = std::make_unique<CameraTrtBranch>(base_config, lidar_config, camera_config);
  lidar_branch_->initialize(stream_);
  camera_branch_->initialize(stream_);
  initOutputs(lidar_branch_->config());
  initTrt(trt_config);
}

void BEVFusionCameraTRT::initTrt(const TrtBEVFusionConfig & trt_config)
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

bool BEVFusionCameraTRT::preProcess(
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

bool BEVFusionCameraTRT::inference()
{
  // Fusion model: run image backbone first, then the shared main network.
  if (camera_branch_->sensor_fusion()) {
    if (!camera_branch_->runImageBackbone()) {
      return false;
    }
  }

  return runMainNetwork();
}

void BEVFusionCameraTRT::setIntrinsicsExtrinsics(
  std::vector<sensor_msgs::msg::CameraInfo> & camera_info_vector,
  std::vector<Matrix4fRowM> & lidar2camera_vector)
{
  camera_branch_->setIntrinsicsExtrinsics(camera_info_vector, lidar2camera_vector);
}

}  //  namespace autoware::bevfusion
