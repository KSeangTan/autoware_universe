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

#include "autoware/bevfusion/camera_lidar/bevfusion_camera_lidar_node.hpp"

#include "autoware/bevfusion/camera/bevfusion_camera_config.hpp"
#include "autoware/bevfusion/camera_lidar/bevfusion_camera_lidar_trt.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/utils.hpp"

#include <memory>

// Contains the camera-lidar fusion BEVFusion node, which composes a camera branch and a lidar
// branch as independent objects.
namespace autoware::bevfusion
{

BEVFusionCameraLidarNode::BEVFusionCameraLidarNode(const rclcpp::NodeOptions & options)
: BEVFusionNode("bevfusion_camera_lidar", options)
{
  const BEVFusionLidarConfig lidar_config = declareLidarConfig(this);
  const CameraSetup camera_setup = declareCameraConfig(this);

  detector_ptr_ = std::make_unique<BEVFusionCameraLidarTRT>(
    makeTrtConfig(camera_setup.image_backbone_trt_config), *densification_param_, *base_config_,
    lidar_config, camera_setup.camera_config);
  shutdownIfBuildOnly();

  camera_branch_.emplace(
    this, detector_ptr_.get(), tf_buffer_, camera_setup.camera_config.sensor_fusion_,
    camera_setup.camera_config.num_cameras_, camera_setup.use_compressed_images,
    camera_setup.max_camera_lidar_delay, camera_setup.image_pre_processing_params);

  lidar_branch_.emplace(
    this, "~/input/pointcloud",
    [this](const LidarBranch::PointCloudConstPtr & pc_msg_ptr) { this->onPointCloud(pc_msg_ptr); });
}

void BEVFusionCameraLidarNode::onPointCloud(const LidarBranch::PointCloudConstPtr & pc_msg_ptr)
{
  camera_branch_->set_lidar_frame(pc_msg_ptr->header.frame_id);

  if (!hasOutputSubscribers() || !camera_branch_->checkReadiness()) {
    return;
  }

  camera_branch_->precomputeIntrinsicsExtrinsics();

  const double lidar_stamp = rclcpp::Time(pc_msg_ptr->header.stamp).seconds();
  camera_branch_->computeCameraMasks(lidar_stamp);

  detectAndPublish(pc_msg_ptr, camera_branch_->camera_data_ptrs(), camera_branch_->camera_masks());
}

}  // namespace autoware::bevfusion

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::bevfusion::BEVFusionCameraLidarNode)
