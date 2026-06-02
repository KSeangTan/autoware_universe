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

#ifndef AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_TRT_HPP_
#define AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_TRT_HPP_

#include "autoware/bevfusion/bevfusion_config.hpp"
#include "autoware/bevfusion/bevfusion_trt.hpp"
#include "autoware/bevfusion/camera/bevfusion_camera_config.hpp"
#include "autoware/bevfusion/camera/camera_data.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_trt.hpp"
#include "autoware/bevfusion/preprocess/pointcloud_densification.hpp"
#include "autoware/bevfusion/preprocess/precomputed_features.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <Eigen/Core>
#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <autoware/cuda_utils/cuda_unique_ptr.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>

#include <sensor_msgs/msg/camera_info.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace autoware::bevfusion
{

using autoware::cuda_utils::CudaUniquePtr;

// Self-contained camera inference branch. It owns the image backbone TensorRT engine, the camera
// device buffers and the precomputed lidar-to-image geometry, and contributes the camera-side
// inputs to the shared main engine. It operates on the shared main engine and CUDA stream owned by
// the BEVFusionTRT base. It is a plain helper (not a detector) so that it can be composed
// independently of the lidar branch. All operations are no-ops when sensor fusion is disabled.
class BEVFUSION_PUBLIC CameraTrtBranch
{
public:
  using Matrix4fRowM = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

  CameraTrtBranch(
    const BEVFusionConfig & base_config, const BEVFusionLidarConfig & lidar_config,
    const BEVFusionCameraConfig & camera_config);

  bool sensor_fusion() const { return camera_config_.sensor_fusion_; }

  // Allocate the device buffers on the given stream (only when sensor fusion is enabled).
  void initialize(cudaStream_t stream);

  // Build the separate image backbone engine (only when sensor fusion is enabled).
  void setupImageBackbone(const TrtBEVFusionConfig & trt_config);

  // Engine setup contributions for the shared main network.
  void addNetworkIO(std::vector<autoware::tensorrt_common::NetworkIO> & network_io) const;
  void addProfileDims(std::vector<autoware::tensorrt_common::ProfileDims> & profile_dims) const;
  // points_d is the lidar raw points buffer reused by the camera "points" input.
  void setTensorAddresses(autoware::tensorrt_common::TrtCommon * network, float * points_d);
  void configureInputShapes(
    autoware::tensorrt_common::TrtCommon * network, std::size_t num_points) const;

  // Precompute and upload the lidar-to-image geometry from the camera intrinsics/extrinsics.
  void setIntrinsicsExtrinsics(
    std::vector<sensor_msgs::msg::CameraInfo> & camera_info_vector,
    std::vector<Matrix4fRowM> & lidar2camera_vector);

  // Per-frame pre-processing / inference steps.
  bool checkImageCameraMatricesReady(
    const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs) const;
  bool processImages(
    const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
    const std::vector<float> & camera_masks);
  bool runImageBackbone();
  void debugSaveRoi();

private:
  BEVFusionConfig base_config_;
  BEVFusionLidarConfig lidar_config_;
  BEVFusionCameraConfig camera_config_;
  cudaStream_t stream_{nullptr};

  std::unique_ptr<autoware::tensorrt_common::TrtCommon> image_backbone_trt_ptr_{nullptr};

  std::vector<int> roi_start_y_vector_;
  std::vector<Matrix4fRowM> img_aug_matrices_;

  // pre computed tensors
  std::int64_t num_geom_feats_{};
  std::int64_t num_kept_{};
  std::int64_t num_ranks_{};
  std::int64_t num_indices_{};
  CudaUniquePtr<float_t[]> lidar2image_d_{};
  CudaUniquePtr<std::int32_t[]> geom_feats_d_{};
  CudaUniquePtr<std::uint8_t[]> kept_d_{};
  CudaUniquePtr<std::int64_t[]> ranks_d_{};
  CudaUniquePtr<std::int64_t[]> indices_d_{};

  // image buffers
  CudaUniquePtr<std::uint8_t[]> roi_tensor_d_{nullptr};
  CudaUniquePtr<float[]> camera_masks_d_{nullptr};

  // image feature buffers for fusion model
  CudaUniquePtr<float[]> image_feats_d_{nullptr};
  CudaUniquePtr<float[]> img_aug_matrix_d_{nullptr};
};

// Camera-only BEVFusion detector. It owns a CameraTrtBranch and a (disabled) LidarTrtBranch on top
// of the shared BEVFusionTRT base. The BEVFusion engine always consumes lidar voxels, so the lidar
// branch is still present (built from a disabled lidar configuration) for engine setup.
class BEVFUSION_PUBLIC BEVFusionCameraTRT : public BEVFusionTRT
{
public:
  BEVFusionCameraTRT(
    const TrtBEVFusionConfig & trt_config, const DensificationParam & densification_param,
    const BEVFusionConfig & base_config, const BEVFusionLidarConfig & lidar_config,
    const BEVFusionCameraConfig & camera_config);

  void setIntrinsicsExtrinsics(
    std::vector<sensor_msgs::msg::CameraInfo> & camera_info_vector,
    std::vector<Matrix4fRowM> & lidar2camera_vector) override;

protected:
  bool preProcess(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
    const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
    const std::vector<float> & camera_masks, const tf2_ros::Buffer & tf_buffer,
    bool & is_num_voxels_within_range) override;

  bool inference() override;

private:
  void initTrt(const TrtBEVFusionConfig & trt_config);

  std::unique_ptr<LidarTrtBranch> lidar_branch_{nullptr};
  std::unique_ptr<CameraTrtBranch> camera_branch_{nullptr};
};

}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_TRT_HPP_
