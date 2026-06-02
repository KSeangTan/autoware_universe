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

#ifndef AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_NODE_HPP_
#define AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_NODE_HPP_

#include "autoware/bevfusion/bevfusion_node.hpp"
#include "autoware/bevfusion/bevfusion_trt.hpp"
#include "autoware/bevfusion/camera/bevfusion_camera_config.hpp"
#include "autoware/bevfusion/camera/camera_data.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <Eigen/Core>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <tf2_ros/buffer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::bevfusion
{

// Bundle of everything that is parsed from the camera parameters: the camera configuration, the
// (optional) image backbone TensorRT configuration and the auxiliary settings consumed by the
// camera branch.
struct CameraSetup
{
  BEVFusionCameraConfig camera_config;
  std::optional<tensorrt_common::TrtCommonConfig> image_backbone_trt_config;
  bool use_compressed_images;
  float max_camera_lidar_delay;
  ImagePreProcessingParams image_pre_processing_params;
};

// Declare the camera parameters on node and build the camera setup from them. Shared by the
// camera-only node and the camera-lidar node so that the camera parameter handling lives in a
// single place.
BEVFUSION_PUBLIC CameraSetup declareCameraConfig(rclcpp::Node * node);

// Self-contained camera branch. It owns the image / camera-info subscriptions, the per-camera data,
// the camera masks and the lidar-to-camera intrinsics/extrinsics. It is a plain helper (not a ROS
// node) so that it can be composed independently of the lidar branch.
class BEVFUSION_PUBLIC CameraBranch
{
public:
  using Matrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

  CameraBranch(
    rclcpp::Node * node, BEVFusionTRT * detector, const tf2_ros::Buffer & tf_buffer,
    bool sensor_fusion, std::int64_t num_cameras, bool use_compressed_images,
    float max_camera_lidar_delay, const ImagePreProcessingParams & image_pre_processing_params);

  // The lidar frame is required to look up the lidar-to-camera extrinsics. The owning node provides
  // it from the latest point cloud message.
  void set_lidar_frame(const std::string & frame_id);

  bool checkReadiness();
  void precomputeIntrinsicsExtrinsics();
  void computeCameraMasks(double lidar_stamp);

  const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs() const
  {
    return camera_data_ptrs_;
  }
  const std::vector<float> & camera_masks() const { return camera_masks_; }

private:
  void initializeSubscribers();
  bool areAllSensorDataAvailable() const;
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg, std::size_t camera_id);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo & msg, std::size_t camera_id);

  rclcpp::Node * node_;
  BEVFusionTRT * detector_;
  const tf2_ros::Buffer & tf_buffer_;

  bool sensor_fusion_{false};
  std::int64_t num_cameras_{0};
  bool use_compressed_images_{false};
  float max_camera_lidar_delay_{0.f};
  ImagePreProcessingParams image_pre_processing_params_;

  std::optional<std::string> lidar_frame_;

  std::vector<image_transport::Subscriber> image_subs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::ConstSharedPtr> camera_info_subs_;
  // unique_ptr to avoid copying the actual camera data in memory since there's gpu buffer in the
  // camera data
  std::vector<std::unique_ptr<CameraData>> camera_data_ptrs_;
  // One CameraMatrices object can be shared by several CameraData
  std::vector<std::shared_ptr<CameraMatrices>> camera_matrices_ptrs_;

  std::vector<float> camera_masks_;
  std::vector<std::optional<Matrix4f>> lidar2camera_extrinsics_;

  bool images_available_{false};
  bool intrinsics_available_{false};
  bool extrinsics_available_{false};
  bool intrinsics_extrinsics_precomputed_{false};
};

// Camera-only BEVFusion node. It composes a CameraBranch on top of the shared BEVFusionNode base.
class BEVFUSION_PUBLIC BEVFusionCameraNode : public BEVFusionNode
{
public:
  explicit BEVFusionCameraNode(const rclcpp::NodeOptions & options);

private:
  // Constructed once the detector is ready so that the camera branch always has a valid detector.
  std::optional<CameraBranch> camera_branch_;
};
}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__CAMERA__BEVFUSION_CAMERA_NODE_HPP_
