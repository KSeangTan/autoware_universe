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

#ifndef AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_TRT_HPP_
#define AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_TRT_HPP_

#include "autoware/bevfusion/bevfusion_config.hpp"
#include "autoware/bevfusion/bevfusion_trt.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/preprocess/pointcloud_densification.hpp"
#include "autoware/bevfusion/preprocess/preprocess_kernel.hpp"
#include "autoware/bevfusion/preprocess/voxel_generator.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <autoware/cuda_utils/cuda_unique_ptr.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>

#include <tf2_ros/buffer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace autoware::bevfusion
{

using autoware::cuda_utils::CudaUniquePtr;

// Self-contained lidar inference branch. It owns the point cloud voxelization (voxel generator and
// pre-processing), the lidar device buffers and the lidar-side TensorRT engine inputs. It operates
// on the shared main engine and CUDA stream owned by the BEVFusionTRT base. It is a plain helper
// (not a detector) so that it can be composed independently of the camera branch.
class BEVFUSION_PUBLIC LidarTrtBranch
{
public:
  LidarTrtBranch(
    const DensificationParam & densification_param, const BEVFusionConfig & base_config,
    const BEVFusionLidarConfig & lidar_config);

  // Allocate the device buffers and build the voxel generator / pre-processor on the given stream.
  void initialize(cudaStream_t stream);

  // Engine setup contributions for the shared main network.
  void addNetworkIO(std::vector<autoware::tensorrt_common::NetworkIO> & network_io) const;
  void addProfileDims(std::vector<autoware::tensorrt_common::ProfileDims> & profile_dims) const;
  void setTensorAddresses(autoware::tensorrt_common::TrtCommon * network);
  void configureInputShapes(
    autoware::tensorrt_common::TrtCommon * network, std::int64_t num_voxels) const;

  // Per-frame pre-processing steps (called in order by the base pipeline).
  bool validatePointCloud(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr) const;
  bool enqueuePointCloud(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
    const tf2_ros::Buffer & tf_buffer);
  void clearDeviceMemory(cudaStream_t stream);
  std::size_t generateSweepPoints();
  std::int64_t voxelize(std::size_t num_points, bool & is_num_voxels_within_range);

  const BEVFusionLidarConfig & config() const { return lidar_config_; }
  // The camera fusion "points" input reuses this raw points buffer.
  float * points_d() const { return points_d_.get(); }

private:
  BEVFusionConfig base_config_;
  BEVFusionLidarConfig lidar_config_;
  DensificationParam densification_param_;
  cudaStream_t stream_{nullptr};

  std::unique_ptr<VoxelGenerator> vg_ptr_{nullptr};
  std::unique_ptr<PreprocessCuda> pre_ptr_{nullptr};

  unsigned int voxel_features_size_{0};
  unsigned int voxel_coords_size_{0};

  CudaUniquePtr<float[]> points_d_{nullptr};
  CudaUniquePtr<float[]> voxel_features_d_{nullptr};
  CudaUniquePtr<std::int32_t[]> voxel_coords_d_{nullptr};
  CudaUniquePtr<std::int32_t[]> num_points_per_voxel_d_{nullptr};
};

// Lidar-only BEVFusion detector. It owns a LidarTrtBranch on top of the shared BEVFusionTRT base
// and implements its own engine setup and pre-processing using only the lidar branch.
class BEVFUSION_PUBLIC BEVFusionLidarTRT : public BEVFusionTRT
{
public:
  BEVFusionLidarTRT(
    const TrtBEVFusionConfig & trt_config, const DensificationParam & densification_param,
    const BEVFusionConfig & base_config, const BEVFusionLidarConfig & lidar_config);

protected:
  bool preProcess(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
    const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
    const std::vector<float> & camera_masks, const tf2_ros::Buffer & tf_buffer,
    bool & is_num_voxels_within_range) override;

private:
  void initTrt(const TrtBEVFusionConfig & trt_config);

  std::unique_ptr<LidarTrtBranch> lidar_branch_{nullptr};
};

}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_TRT_HPP_
