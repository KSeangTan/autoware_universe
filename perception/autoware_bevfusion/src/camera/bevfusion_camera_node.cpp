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

#include "autoware/bevfusion/camera/bevfusion_camera_node.hpp"

#include "autoware/bevfusion/camera/bevfusion_camera_config.hpp"
#include "autoware/bevfusion/camera/bevfusion_camera_trt.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/utils.hpp"

#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Contains the camera branch (image/camera-info subscriptions and callbacks, camera masks and the
// lidar-to-camera intrinsics/extrinsics) and the camera-only BEVFusion node.
namespace autoware::bevfusion
{

CameraSetup declareCameraConfig(rclcpp::Node * node)
{
  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{}.set__read_only(true);

  auto to_float_vector = [](const auto & v) -> std::vector<float> {
    return std::vector<float>(v.begin(), v.end());
  };

  const std::string image_backbone_onnx_path =
    node->declare_parameter<std::string>("image_backbone_onnx_path", descriptor);
  const std::string image_backbone_engine_path =
    node->declare_parameter<std::string>("image_backbone_engine_path", descriptor);
  const std::string image_backbone_trt_precision =
    node->declare_parameter<std::string>("image_backbone_trt_precision", descriptor);

  const auto d_bound =
    to_float_vector(node->declare_parameter<std::vector<double>>("d_bound", descriptor));
  const auto x_bound =
    to_float_vector(node->declare_parameter<std::vector<double>>("x_bound", descriptor));
  const auto y_bound =
    to_float_vector(node->declare_parameter<std::vector<double>>("y_bound", descriptor));
  const auto z_bound =
    to_float_vector(node->declare_parameter<std::vector<double>>("z_bound", descriptor));
  const auto num_cameras = node->declare_parameter<std::int64_t>("num_cameras", descriptor);
  const auto raw_image_height =
    node->declare_parameter<std::int64_t>("raw_image_height", descriptor);
  const auto raw_image_width = node->declare_parameter<std::int64_t>("raw_image_width", descriptor);
  const auto img_aug_scale_x = node->declare_parameter<float>("img_aug_scale_x", descriptor);
  const auto img_aug_scale_y = node->declare_parameter<float>("img_aug_scale_y", descriptor);
  const auto roi_height = node->declare_parameter<std::int64_t>("roi_height", descriptor);
  const auto roi_width = node->declare_parameter<std::int64_t>("roi_width", descriptor);
  const auto features_height = node->declare_parameter<std::int64_t>("features_height", descriptor);
  const auto features_width = node->declare_parameter<int>("features_width", descriptor);
  const auto num_depth_features = node->declare_parameter<int>("num_depth_features", descriptor);
  const auto image_feature_channel =
    node->declare_parameter<std::int64_t>("image_feature_channel", descriptor);

  const auto use_compressed_images =
    node->declare_parameter<bool>("use_compressed_images", false, descriptor);
  const auto run_image_undistortion =
    node->declare_parameter<bool>("run_image_undistortion", descriptor);
  const auto max_camera_lidar_delay =
    node->declare_parameter<float>("max_camera_lidar_delay", descriptor);

  const BEVFusionCameraConfig camera_config(
    image_backbone_onnx_path, image_backbone_engine_path, image_backbone_trt_precision, d_bound,
    x_bound, y_bound, z_bound, num_cameras, raw_image_height, raw_image_width, img_aug_scale_x,
    img_aug_scale_y, roi_height, roi_width, features_height, features_width, num_depth_features,
    image_feature_channel);

  std::optional<tensorrt_common::TrtCommonConfig> image_backbone_trt_config;
  if (camera_config.sensor_fusion_) {
    image_backbone_trt_config = tensorrt_common::TrtCommonConfig(
      image_backbone_onnx_path, image_backbone_trt_precision, image_backbone_engine_path,
      1ULL << 32U);
  }

  const ImagePreProcessingParams image_pre_processing_params(
    raw_image_height, raw_image_width, roi_height, roi_width, img_aug_scale_y, img_aug_scale_x,
    run_image_undistortion);

  return CameraSetup{
    camera_config, image_backbone_trt_config, use_compressed_images, max_camera_lidar_delay,
    image_pre_processing_params};
}

CameraBranch::CameraBranch(
  rclcpp::Node * node, BEVFusionTRT * detector, const tf2_ros::Buffer & tf_buffer,
  bool sensor_fusion, std::int64_t num_cameras, bool use_compressed_images,
  float max_camera_lidar_delay, const ImagePreProcessingParams & image_pre_processing_params)
: node_(node),
  detector_(detector),
  tf_buffer_(tf_buffer),
  sensor_fusion_(sensor_fusion),
  num_cameras_(num_cameras),
  use_compressed_images_(use_compressed_images),
  max_camera_lidar_delay_(max_camera_lidar_delay),
  image_pre_processing_params_(image_pre_processing_params)
{
  initializeSubscribers();
}

void CameraBranch::set_lidar_frame(const std::string & frame_id)
{
  lidar_frame_ = frame_id;
}

void CameraBranch::initializeSubscribers()
{
  if (!sensor_fusion_) {
    return;
  }

  image_subs_.resize(num_cameras_);
  camera_info_subs_.resize(num_cameras_);
  lidar2camera_extrinsics_.resize(num_cameras_);
  camera_data_ptrs_.resize(num_cameras_);
  camera_matrices_ptrs_.resize(num_cameras_);

  auto resolve_topic_name = [this](const std::string & query) {
    return node_->get_node_topics_interface()->resolve_topic_name(query);
  };
  const std::string transport = use_compressed_images_ ? "compressed" : "raw";

  for (std::int64_t camera_id = 0; camera_id < num_cameras_; ++camera_id) {
    // First construct CameraMatrices, CameraMatrices is shared by multiple CameraData for the same
    // camera_id
    camera_matrices_ptrs_[camera_id] = std::make_shared<CameraMatrices>();

    // Then construct CameraData
    camera_data_ptrs_[camera_id] = std::make_unique<CameraData>(
      node_, static_cast<int>(camera_id), image_pre_processing_params_,
      camera_matrices_ptrs_[camera_id]);

    // Explicitly resolve the topic name using the node name and namespace, please check
    // https://github.com/ros-perception/image_transport_plugins/issues/155
    const std::string base_topic = resolve_topic_name("~/input/image" + std::to_string(camera_id));
    image_subs_[camera_id] = image_transport::create_subscription(
      node_, base_topic,
      std::bind(
        &CameraBranch::imageCallback, this, std::placeholders::_1,
        static_cast<std::size_t>(camera_id)),
      transport, rmw_qos_profile_sensor_data);

    camera_info_subs_[camera_id] = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
      "~/input/camera_info" + std::to_string(camera_id), rclcpp::SensorDataQoS{}.keep_last(1),
      [this, camera_id](const sensor_msgs::msg::CameraInfo & msg) {
        this->cameraInfoCallback(msg, camera_id);
      });
  }
}

bool CameraBranch::areAllSensorDataAvailable() const
{
  return extrinsics_available_ && images_available_ && intrinsics_available_;
}

bool CameraBranch::checkReadiness()
{
  if (!sensor_fusion_) {
    return true;
  }

  if (!areAllSensorDataAvailable()) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "Sensor fusion mode enabled but missing required data: extrinsics_available=%s, "
      "images_available=%s, intrinsics_available=%s. Skipping detection",
      extrinsics_available_ ? "true" : "false", images_available_ ? "true" : "false",
      intrinsics_available_ ? "true" : "false");
    return false;
  }

  return true;
}

void CameraBranch::precomputeIntrinsicsExtrinsics()
{
  if (!sensor_fusion_ || intrinsics_extrinsics_precomputed_) {
    return;
  }

  std::vector<sensor_msgs::msg::CameraInfo> camera_info_msgs;
  std::vector<Matrix4f> lidar2camera_extrinsics;

  try {
    std::transform(
      camera_data_ptrs_.begin(), camera_data_ptrs_.end(), std::back_inserter(camera_info_msgs),
      [](const auto & camera_data) { return camera_data->camera_info_value(); });
  } catch (const std::runtime_error & e) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("bevfusion"), "Camera info is not available for some cameras!");
    return;
  }

  std::transform(
    lidar2camera_extrinsics_.begin(), lidar2camera_extrinsics_.end(),
    std::back_inserter(lidar2camera_extrinsics), [](const auto & opt) { return *opt; });

  detector_->setIntrinsicsExtrinsics(camera_info_msgs, lidar2camera_extrinsics);
  intrinsics_extrinsics_precomputed_ = true;
}

void CameraBranch::computeCameraMasks(double lidar_stamp)
{
  camera_masks_.resize(camera_data_ptrs_.size());
  for (std::size_t i = 0; i < camera_masks_.size(); ++i) {
    auto camera_mask_timestamp = rclcpp::Time(camera_data_ptrs_[i]->image_msg()->header.stamp);
    camera_masks_[i] =
      (lidar_stamp - camera_mask_timestamp.seconds()) < max_camera_lidar_delay_ ? 1.0 : 0.f;
  }
}

void CameraBranch::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr msg, std::size_t camera_id)
{
  camera_data_ptrs_[camera_id]->update_image_msg(msg);

  std::size_t num_valid_images = std::count_if(
    camera_data_ptrs_.begin(), camera_data_ptrs_.end(),
    [](const auto & camera_data) { return camera_data->is_image_msg_available(); });

  images_available_ = num_valid_images == camera_data_ptrs_.size();
}

void CameraBranch::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo & msg, std::size_t camera_id)
{
  camera_data_ptrs_[camera_id]->update_camera_info(msg);
  std::size_t num_valid_intrinsics = std::count_if(
    camera_data_ptrs_.begin(), camera_data_ptrs_.end(),
    [](const auto & camera_data) { return camera_data->is_camera_info_available(); });

  intrinsics_available_ = num_valid_intrinsics == camera_data_ptrs_.size();

  if (
    lidar2camera_extrinsics_[camera_id].has_value() || !lidar_frame_.has_value() ||
    extrinsics_available_) {
    return;
  }

  try {
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped =
      tf_buffer_.lookupTransform(msg.header.frame_id, *lidar_frame_, msg.header.stamp);

    Eigen::Matrix4f lidar2camera_transform =
      tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();

    Matrix4f lidar2camera_rowmajor_transform = lidar2camera_transform.eval();
    lidar2camera_extrinsics_[camera_id] = lidar2camera_rowmajor_transform;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("bevfusion"), ex.what());
    return;
  }

  std::size_t num_valid_extrinsics = std::count_if(
    lidar2camera_extrinsics_.begin(), lidar2camera_extrinsics_.end(),
    [](const auto & opt) { return opt.has_value(); });

  extrinsics_available_ = num_valid_extrinsics == lidar2camera_extrinsics_.size();
}

BEVFusionCameraNode::BEVFusionCameraNode(const rclcpp::NodeOptions & options)
: BEVFusionNode("bevfusion_camera", options)
{
  const CameraSetup camera_setup = declareCameraConfig(this);

  // Camera-only mode does not voxelize a point cloud, so the lidar configuration is disabled.
  const BEVFusionLidarConfig lidar_config(0, 0, {}, {}, {}, false);

  detector_ptr_ = std::make_unique<BEVFusionCameraTRT>(
    makeTrtConfig(camera_setup.image_backbone_trt_config), *densification_param_, *base_config_,
    lidar_config, camera_setup.camera_config);
  shutdownIfBuildOnly();

  camera_branch_.emplace(
    this, detector_ptr_.get(), tf_buffer_, camera_setup.camera_config.sensor_fusion_,
    camera_setup.camera_config.num_cameras_, camera_setup.use_compressed_images,
    camera_setup.max_camera_lidar_delay, camera_setup.image_pre_processing_params);
}

}  // namespace autoware::bevfusion

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::bevfusion::BEVFusionCameraNode)
