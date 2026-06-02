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

#ifndef AUTOWARE__BEVFUSION__BEVFUSION_NODE_HPP_
#define AUTOWARE__BEVFUSION__BEVFUSION_NODE_HPP_

#include "autoware/bevfusion/bevfusion_trt.hpp"
#include "autoware/bevfusion/camera/camera_data.hpp"
#include "autoware/bevfusion/detection_class_remapper.hpp"
#include "autoware/bevfusion/postprocess/non_maximum_suppression.hpp"
#include "autoware/bevfusion/preprocess/pointcloud_densification.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <Eigen/Core>
#include <autoware_utils_debug/debug_publisher.hpp>
#include <autoware_utils_debug/published_time_publisher.hpp>
#include <autoware_utils_diagnostics/diagnostics_interface.hpp>
#include <autoware_utils_system/stop_watch.hpp>
#include <cuda_blackboard/cuda_adaptation.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <cuda_blackboard/negotiated_types.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::bevfusion
{

// Common base class for all BEVFusion nodes. It owns the detector, the output publisher and the
// shared detection/diagnostics/debug logic, so that the lidar-only, camera and camera-lidar nodes
// only have to compose the camera/lidar branches and provide their specific callbacks.
class BEVFUSION_PUBLIC BEVFusionNode : public rclcpp::Node
{
public:
  using Matrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

protected:
  BEVFusionNode(const std::string & node_name, const rclcpp::NodeOptions & options);

  // Shared detection pipeline: run the detector, apply NMS, remap classes, publish results and
  // debug info. The branches feed the (optional) camera data and masks.
  void detectAndPublish(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
    const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
    const std::vector<float> & camera_masks);

  bool hasOutputSubscribers() const;

  void diagnoseProcessingTime(diagnostic_updater::DiagnosticStatusWrapper & stat);

  // Helper methods for the constructor

  // Assemble the TensorRT configuration from the shared main-network engine settings declared in
  // this base constructor. image_backbone_trt_config is only provided by the camera/camera-lidar
  // fusion nodes.
  TrtBEVFusionConfig makeTrtConfig(
    const std::optional<tensorrt_common::TrtCommonConfig> & image_backbone_trt_config) const;

  // Shut the node down once the TensorRT engine has been built, when the build_only parameter is
  // set. Derived nodes call it right after they construct the detector.
  void shutdownIfBuildOnly();

  // Helper methods for the detection pipeline
  void publishDetectionResults(
    const autoware_perception_msgs::msg::DetectedObjects & output_msg,
    const std_msgs::msg::Header & header);
  void publishDebugInfo(
    const std::unordered_map<std::string, double> & proc_timing,
    const std_msgs::msg::Header & header);

  // Helper methods for diagnoseProcessingTime
  void addNoInferenceDiagnostics(
    diagnostic_updater::DiagnosticStatusWrapper & stat, std::stringstream & message);
  diagnostic_msgs::msg::DiagnosticStatus::_level_type checkProcessingTimeStatus(
    diagnostic_updater::DiagnosticStatusWrapper & stat, std::stringstream & message,
    const rclcpp::Time & timestamp_now);
  diagnostic_msgs::msg::DiagnosticStatus::_level_type checkConsecutiveDelays(
    diagnostic_updater::DiagnosticStatusWrapper & stat, std::stringstream & message,
    const rclcpp::Time & timestamp_now,
    diagnostic_msgs::msg::DiagnosticStatus::_level_type current_level);

  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_{
    nullptr};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_{tf_buffer_};

  DetectionClassRemapper detection_class_remapper_;
  std::vector<std::string> class_names_;

  // Shared configuration declared/parsed in the base constructor. The derived nodes combine these
  // with their own (lidar/camera) configuration to build the detector.
  std::optional<BEVFusionConfig> base_config_;
  std::optional<DensificationParam> densification_param_;
  std::string onnx_path_;
  std::string engine_path_;
  std::string trt_precision_;
  bool build_only_{false};

  // for diagnostics
  double max_allowed_processing_time_ms_;
  double max_acceptable_consecutive_delay_ms_;
  // set as optional to avoid sending error diagnostics before the node starts processing
  std::optional<double> last_processing_time_ms_;
  std::optional<rclcpp::Time> last_in_time_processing_timestamp_;
  diagnostic_updater::Updater diagnostic_processing_time_updater_{this};

  NonMaximumSuppression iou_bev_nms_;

  std::unique_ptr<BEVFusionTRT> detector_ptr_{nullptr};
  std::unique_ptr<autoware_utils_diagnostics::DiagnosticsInterface> diagnostics_detector_trt_;

  // debugger
  std::unique_ptr<autoware_utils_system::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_{
    nullptr};
  std::unique_ptr<autoware_utils_debug::DebugPublisher> debug_publisher_ptr_{nullptr};
  std::unique_ptr<autoware_utils_debug::PublishedTimePublisher> published_time_pub_{nullptr};
};
}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__BEVFUSION_NODE_HPP_
