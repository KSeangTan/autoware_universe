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

#ifndef AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_NODE_HPP_
#define AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_NODE_HPP_

#include "autoware/bevfusion/bevfusion_node.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <cuda_blackboard/cuda_blackboard_subscriber.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <rclcpp/rclcpp.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace autoware::bevfusion
{

// Declare the lidar voxelization parameters on node and build the lidar configuration from them.
// Shared by the lidar-only node and the camera-lidar node so that the lidar parameter handling
// lives in a single place.
BEVFUSION_PUBLIC BEVFusionLidarConfig declareLidarConfig(rclcpp::Node * node);

// Self-contained lidar branch. It owns the point cloud subscription and forwards every message to
// the callback provided by the owning node. It is a plain helper (not a ROS node) so that it can be
// composed independently of the camera branch.
class BEVFUSION_PUBLIC LidarBranch
{
public:
  using PointCloudConstPtr = std::shared_ptr<const cuda_blackboard::CudaPointCloud2>;
  using PointCloudCallback = std::function<void(PointCloudConstPtr)>;

  LidarBranch(rclcpp::Node * node, const std::string & topic, PointCloudCallback callback);

private:
  std::unique_ptr<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>
    cloud_sub_;
};

// Lidar-only BEVFusion node. It composes a LidarBranch and runs the shared detection pipeline
// without any camera data.
class BEVFUSION_PUBLIC BEVFusionLidarNode : public BEVFusionNode
{
public:
  explicit BEVFusionLidarNode(const rclcpp::NodeOptions & options);

private:
  void onPointCloud(const LidarBranch::PointCloudConstPtr & pc_msg_ptr);

  // Constructed once the detector is ready so that no point cloud message is processed before the
  // detector exists.
  std::optional<LidarBranch> lidar_branch_;
};
}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_NODE_HPP_
