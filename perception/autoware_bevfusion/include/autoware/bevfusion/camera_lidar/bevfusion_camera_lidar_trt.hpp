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

#ifndef AUTOWARE__BEVFUSION__CAMERA_LIDAR__BEVFUSION_CAMERA_LIDAR_TRT_HPP_
#define AUTOWARE__BEVFUSION__CAMERA_LIDAR__BEVFUSION_CAMERA_LIDAR_TRT_HPP_

#include "autoware/bevfusion/bevfusion_config.hpp"
#include "autoware/bevfusion/bevfusion_trt.hpp"
#include "autoware/bevfusion/camera/bevfusion_camera_config.hpp"
#include "autoware/bevfusion/camera/bevfusion_camera_trt.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_trt.hpp"
#include "autoware/bevfusion/preprocess/pointcloud_densification.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <memory>

namespace autoware::bevfusion
{

// Camera-lidar fusion BEVFusion detector. Instead of inheriting from the camera detector, it owns a
// LidarTrtBranch and a CameraTrtBranch as independent objects on top of the shared BEVFusionTRT
// base, initializes both on the shared CUDA stream, and feeds both into the shared detection
// pipeline.
class BEVFUSION_PUBLIC BEVFusionCameraLidarTRT : public BEVFusionTRT
{
public:
  BEVFusionCameraLidarTRT(
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

#endif  // AUTOWARE__BEVFUSION__CAMERA_LIDAR__BEVFUSION_CAMERA_LIDAR_TRT_HPP_
