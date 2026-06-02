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

#include "autoware/bevfusion/lidar/bevfusion_lidar_node.hpp"

#include "autoware/bevfusion/camera/camera_data.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_trt.hpp"
#include "autoware/bevfusion/utils.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Contains the lidar branch and the lidar-only BEVFusion node.
namespace autoware::bevfusion
{

BEVFusionLidarConfig declareLidarConfig(rclcpp::Node * node)
{
  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{}.set__read_only(true);

  auto to_float_vector = [](const auto & v) -> std::vector<float> {
    return std::vector<float>(v.begin(), v.end());
  };

  const auto cloud_capacity = node->declare_parameter<std::int64_t>("cloud_capacity", descriptor);
  const auto max_points_per_voxel =
    node->declare_parameter<std::int64_t>("max_points_per_voxel", descriptor);
  const auto voxels_num =
    node->declare_parameter<std::vector<std::int64_t>>("voxels_num", descriptor);
  const auto point_cloud_range =
    to_float_vector(node->declare_parameter<std::vector<double>>("point_cloud_range", descriptor));
  const auto voxel_size =
    to_float_vector(node->declare_parameter<std::vector<double>>("voxel_size", descriptor));
  const auto use_intensity = node->declare_parameter<bool>("use_intensity", descriptor);

  if (point_cloud_range.size() != 6) {
    throw std::runtime_error("The size of point_cloud_range != 6");
  }
  if (voxel_size.size() != 3) {
    throw std::runtime_error("The size of voxel_size != 3");
  }

  return BEVFusionLidarConfig(
    cloud_capacity, max_points_per_voxel, voxels_num, point_cloud_range, voxel_size, use_intensity);
}

LidarBranch::LidarBranch(
  rclcpp::Node * node, const std::string & topic, PointCloudCallback callback)
{
  cloud_sub_ =
    std::make_unique<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>(
      *node, topic, std::move(callback));
}

BEVFusionLidarNode::BEVFusionLidarNode(const rclcpp::NodeOptions & options)
: BEVFusionNode("bevfusion_lidar", options)
{
  const BEVFusionLidarConfig lidar_config = declareLidarConfig(this);

  detector_ptr_ = std::make_unique<BEVFusionLidarTRT>(
    makeTrtConfig(std::nullopt), *densification_param_, *base_config_, lidar_config);
  shutdownIfBuildOnly();

  lidar_branch_.emplace(
    this, "~/input/pointcloud",
    [this](const LidarBranch::PointCloudConstPtr & pc_msg_ptr) { this->onPointCloud(pc_msg_ptr); });
}

void BEVFusionLidarNode::onPointCloud(const LidarBranch::PointCloudConstPtr & pc_msg_ptr)
{
  if (!hasOutputSubscribers()) {
    return;
  }

  // Lidar-only mode does not use any camera data
  static const std::vector<std::unique_ptr<CameraData>> empty_camera_data_ptrs;
  static const std::vector<float> empty_camera_masks;

  detectAndPublish(pc_msg_ptr, empty_camera_data_ptrs, empty_camera_masks);
}

}  // namespace autoware::bevfusion

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::bevfusion::BEVFusionLidarNode)
