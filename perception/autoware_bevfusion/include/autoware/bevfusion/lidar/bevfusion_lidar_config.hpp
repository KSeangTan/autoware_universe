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

#ifndef AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_CONFIG_HPP_
#define AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_CONFIG_HPP_

#include <array>
#include <cstdint>
#include <vector>

namespace autoware::bevfusion
{

// Lidar branch configuration. It is an independent configuration that holds the point cloud
// voxelization parameters (point cloud range, voxel size, grid size and voxel counts).
class BEVFusionLidarConfig
{
public:
  BEVFusionLidarConfig(
    const std::int64_t cloud_capacity, const std::int64_t max_points_per_voxel,
    const std::vector<std::int64_t> & voxels_num, const std::vector<float> & point_cloud_range,
    const std::vector<float> & voxel_size, const bool use_intensity)
  {
    use_intensity_ = use_intensity;
    if (use_intensity_) {
      // x, y, z, intensity, timestamp_lag
      num_point_feature_size_ = 5;
    } else {
      // x, y, z, timestamp_lag
      num_point_feature_size_ = 4;
    }

    cloud_capacity_ = cloud_capacity;
    max_points_per_voxel_ = max_points_per_voxel;

    if (voxels_num.size() == 3) {
      min_num_voxels_ = voxels_num[0];
      max_num_voxels_ = voxels_num[2];

      voxels_num_[0] = voxels_num[0];
      voxels_num_[1] = voxels_num[1];
      voxels_num_[2] = voxels_num[2];
    }
    if (point_cloud_range.size() == 6) {
      min_x_range_ = point_cloud_range[0];
      min_y_range_ = point_cloud_range[1];
      min_z_range_ = point_cloud_range[2];
      max_x_range_ = point_cloud_range[3];
      max_y_range_ = point_cloud_range[4];
      max_z_range_ = point_cloud_range[5];
    }
    if (voxel_size.size() == 3) {
      voxel_x_size_ = voxel_size[0];
      voxel_y_size_ = voxel_size[1];
      voxel_z_size_ = voxel_size[2];
    }

    grid_x_size_ = static_cast<std::int64_t>((max_x_range_ - min_x_range_) / voxel_x_size_);
    grid_y_size_ = static_cast<std::int64_t>((max_y_range_ - min_y_range_) / voxel_y_size_);
    grid_z_size_ = static_cast<std::int64_t>((max_z_range_ - min_z_range_) / voxel_z_size_);
  }

  bool use_intensity_{false};

  std::int64_t cloud_capacity_{};
  std::int64_t min_num_voxels_{};
  std::int64_t max_num_voxels_{};
  std::int64_t max_points_per_voxel_{};

  std::int64_t num_point_feature_size_{4};  // x, y, z, timestamp_lag

  // Pointcloud range in meters
  float min_x_range_{};
  float max_x_range_{};
  float min_y_range_{};
  float max_y_range_{};
  float min_z_range_{};
  float max_z_range_{};

  // Voxel size in meters
  float voxel_x_size_{};
  float voxel_y_size_{};
  float voxel_z_size_{};

  // Grid size
  std::int64_t grid_x_size_{};
  std::int64_t grid_y_size_{};
  std::int64_t grid_z_size_{};

  ///// RUNTIME DIMENSIONS /////
  std::array<std::int64_t, 3> voxels_num_{};
};

}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__LIDAR__BEVFUSION_LIDAR_CONFIG_HPP_
