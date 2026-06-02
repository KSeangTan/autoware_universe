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

#ifndef AUTOWARE__BEVFUSION__BEVFUSION_CONFIG_HPP_
#define AUTOWARE__BEVFUSION__BEVFUSION_CONFIG_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::bevfusion
{

// Common BEVFusion configuration shared by every modality. It holds the TensorRT plugin path, the
// head and post-processing parameters and the shared constants. The lidar- and camera-specific
// parameters live in the independent BEVFusionLidarConfig and BEVFusionCameraConfig.
class BEVFusionConfig
{
public:
  BEVFusionConfig(
    const std::size_t class_size, const std::string & plugins_path,
    const std::int64_t out_size_factor, const std::int64_t num_proposals,
    const float circle_nms_dist_threshold, const std::vector<double> & yaw_norm_thresholds,
    const std::vector<float> & score_thresholds,
    const std::vector<float> & distance_bin_upper_limits)
  {
    plugins_path_ = plugins_path;

    out_size_factor_ = out_size_factor;

    num_classes_ = class_size;

    if (num_proposals > 0) {
      num_proposals_ = num_proposals;
    }
    // score_upper_bounds must be sorted in ascending order, raise an error if not
    if (!std::is_sorted(distance_bin_upper_limits.begin(), distance_bin_upper_limits.end())) {
      throw std::invalid_argument("distance_bin_upper_limits must be sorted in ascending order");
    }
    distance_bin_upper_limits_ = distance_bin_upper_limits;
    for (auto & distance_bin_upper_limit : distance_bin_upper_limits_) {
      // Note: Square the distance bin upper limit to get the radial distance to skip the sqrtf
      // operation
      distance_bin_upper_limit = distance_bin_upper_limit * distance_bin_upper_limit;
    }

    // score_thresholds must have the size of score_upper_bounds * class_size
    if (score_thresholds.size() != distance_bin_upper_limits_.size() * num_classes_) {
      throw std::invalid_argument(
        "score_thresholds must have the size of distance_bin_upper_limits * class_size");
    }
    score_thresholds_ = score_thresholds;
    for (auto & score_threshold : score_thresholds_) {
      score_threshold = (score_threshold >= 0.f && score_threshold < 1.f) ? score_threshold : 0.f;
    }

    if (circle_nms_dist_threshold > 0.0) {
      circle_nms_dist_threshold_ = circle_nms_dist_threshold;
    }
    yaw_norm_thresholds_ =
      std::vector<float>(yaw_norm_thresholds.begin(), yaw_norm_thresholds.end());
    for (auto & yaw_norm_threshold : yaw_norm_thresholds_) {
      yaw_norm_threshold =
        (yaw_norm_threshold >= 0.0 && yaw_norm_threshold < 1.0) ? yaw_norm_threshold : 0.0;
    }
  }

  // CUDA parameters
  const std::uint32_t threads_per_block_{256};  // threads number for a block

  // TensorRT parameters
  std::string plugins_path_{};

  // Constants
  static constexpr std::int64_t kTransformMatrixDim = 4;  // 4x4 transformation matrix dimension
  static constexpr std::int64_t kNumRGBChannels = 3;      // RGB color channels
  static constexpr std::int64_t kNum3DCoords = 3;         // 3D coordinates (x, y, z)

  ///// NETWORK PARAMETERS /////

  // Common network parameters
  std::int64_t out_size_factor_{};

  // Head parameters
  std::int64_t num_proposals_{};
  std::size_t num_classes_{5};

  // Post processing parameters

  // the score threshold for classification
  std::vector<float> distance_bin_upper_limits_{};
  std::vector<float> score_thresholds_{};

  float circle_nms_dist_threshold_{};
  std::vector<float> yaw_norm_thresholds_{};
  // the detected boxes result decode by (x, y, z, w, l, h, yaw, vx, vy)
  const std::int64_t num_box_values_{10};
};

}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__BEVFUSION_CONFIG_HPP_
