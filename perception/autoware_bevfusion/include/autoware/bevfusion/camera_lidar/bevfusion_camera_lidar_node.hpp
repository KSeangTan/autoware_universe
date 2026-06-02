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

#ifndef AUTOWARE__BEVFUSION__CAMERA_LIDAR__BEVFUSION_CAMERA_LIDAR_NODE_HPP_
#define AUTOWARE__BEVFUSION__CAMERA_LIDAR__BEVFUSION_CAMERA_LIDAR_NODE_HPP_

#include "autoware/bevfusion/bevfusion_node.hpp"
#include "autoware/bevfusion/camera/bevfusion_camera_node.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_node.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>

#include <optional>

namespace autoware::bevfusion
{

// Camera-lidar fusion BEVFusion node. Instead of inheriting from the camera node, it composes a
// CameraBranch and a LidarBranch as independent objects on top of the shared BEVFusionNode base,
// and feeds both into the shared detection pipeline.
class BEVFUSION_PUBLIC BEVFusionCameraLidarNode : public BEVFusionNode
{
public:
  explicit BEVFusionCameraLidarNode(const rclcpp::NodeOptions & options);

private:
  void onPointCloud(const LidarBranch::PointCloudConstPtr & pc_msg_ptr);

  // Constructed once the detector is ready: the camera branch needs a valid detector and the lidar
  // branch must not deliver a point cloud before the detector exists.
  std::optional<CameraBranch> camera_branch_;
  std::optional<LidarBranch> lidar_branch_;
};
}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__CAMERA_LIDAR__BEVFUSION_CAMERA_LIDAR_NODE_HPP_
