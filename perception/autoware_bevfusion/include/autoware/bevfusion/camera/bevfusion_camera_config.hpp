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

#ifndef AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_CONFIG_HPP_
#define AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_CONFIG_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::bevfusion
{

// Camera branch configuration. It is an independent configuration that holds the image backbone
// network paths and the image and BEV-pooling parameters required for the camera-lidar fusion
// model (frustum bounds, camera geometry, image dimensions and feature dimensions). It also derives
// the sensor_fusion flag from the image backbone parameters.
class BEVFusionCameraConfig
{
public:
  BEVFusionCameraConfig(
    const std::string & image_backbone_onnx_path, const std::string & image_backbone_engine_path,
    const std::string & image_backbone_trt_precision, const std::vector<float> & d_bound,
    const std::vector<float> & x_bound, const std::vector<float> & y_bound,
    const std::vector<float> & z_bound, const std::int64_t num_cameras,
    const std::int64_t raw_image_height, const std::int64_t raw_image_width,
    const float img_aug_scale_x, const float img_aug_scale_y, const std::int64_t roi_height,
    const std::int64_t roi_width, const std::int64_t features_height,
    const std::int64_t features_width, const std::int64_t num_depth_features,
    const std::int64_t image_feature_channel)
  {
    // Derive sensor_fusion from image backbone parameters
    // All three must be empty OR all three must be non-empty
    const bool all_empty = image_backbone_onnx_path.empty() && image_backbone_engine_path.empty() &&
                           image_backbone_trt_precision.empty();
    const bool all_non_empty = !image_backbone_onnx_path.empty() &&
                               !image_backbone_engine_path.empty() &&
                               !image_backbone_trt_precision.empty();

    if (!all_empty && !all_non_empty) {
      throw std::invalid_argument(
        "Image backbone parameters must be either all empty (lidar-only mode) or all non-empty "
        "(fusion mode). Got: image_backbone_onnx_path='" +
        image_backbone_onnx_path + "', image_backbone_engine_path='" + image_backbone_engine_path +
        "', image_backbone_trt_precision='" + image_backbone_trt_precision + "'");
    }

    sensor_fusion_ = all_non_empty;

    if (d_bound.size() == 3 && x_bound.size() == 3 && y_bound.size() == 3 && z_bound.size() == 3) {
      d_bound_ = d_bound;
      x_bound_ = x_bound;
      y_bound_ = y_bound;
      z_bound_ = z_bound;
    }

    num_cameras_ = num_cameras;
    raw_image_height_ = raw_image_height;
    raw_image_width_ = raw_image_width;
    img_aug_scale_x_ = img_aug_scale_x;
    img_aug_scale_y_ = img_aug_scale_y;
    roi_height_ = roi_height;
    roi_width_ = roi_width;
    features_height_ = features_height;
    features_width_ = features_width;
    num_depth_features_ = num_depth_features;
    image_feature_channel_ = image_feature_channel;
    resized_height_ = raw_image_height_ * img_aug_scale_y_;
    resized_width_ = raw_image_width_ * img_aug_scale_x_;
  }

  ///// MODALITY /////
  bool sensor_fusion_{};

  // Camera branch parameters
  std::vector<float> d_bound_{};
  std::vector<float> x_bound_{};
  std::vector<float> y_bound_{};
  std::vector<float> z_bound_{};

  std::int64_t num_cameras_{};
  std::int64_t raw_image_height_{};
  std::int64_t raw_image_width_{};

  float img_aug_scale_x_{};
  float img_aug_scale_y_{};

  std::int64_t roi_height_{};
  std::int64_t roi_width_{};

  std::int64_t resized_height_{};
  std::int64_t resized_width_{};

  std::int64_t features_height_{};
  std::int64_t features_width_{};
  std::int64_t num_depth_features_{};
  std::int64_t image_feature_channel_{256};  // Image feature dimension
};

}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_CONFIG_HPP_
