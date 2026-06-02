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

#include "autoware/bevfusion/bevfusion_node.hpp"

#include "autoware/bevfusion/ros_utils.hpp"
#include "autoware/bevfusion/utils.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Contains the common constructor and the shared detection pipeline of the BEVFusion node family.
namespace autoware::bevfusion
{

BEVFusionNode::BEVFusionNode(const std::string & node_name, const rclcpp::NodeOptions & options)
: Node(node_name, options), tf_buffer_(this->get_clock())
{
  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{}.set__read_only(true);

  const std::string plugins_path = this->declare_parameter<std::string>("plugins_path", descriptor);
  onnx_path_ = this->declare_parameter<std::string>("onnx_path", descriptor);
  engine_path_ = this->declare_parameter<std::string>("engine_path", descriptor);
  trt_precision_ = this->declare_parameter<std::string>("trt_precision", descriptor);

  class_names_ = this->declare_parameter<std::vector<std::string>>("class_names", descriptor);

  const std::string densification_world_frame_id =
    this->declare_parameter<std::string>("densification_world_frame_id", descriptor);
  const int densification_num_past_frames =
    this->declare_parameter<std::int64_t>("densification_num_past_frames", descriptor);

  {  // IoU NMS
    NMSParams p;
    p.search_distance_2d_ =
      this->declare_parameter<double>("iou_nms_search_distance_2d", descriptor);
    p.iou_threshold_ = this->declare_parameter<double>("iou_nms_threshold", descriptor);
    iou_bev_nms_.setParameters(p);
  }

  const auto allow_remapping_by_area_matrix = this->declare_parameter<std::vector<std::int64_t>>(
    "allow_remapping_by_area_matrix", descriptor);
  const auto min_area_matrix =
    this->declare_parameter<std::vector<double>>("min_area_matrix", descriptor);
  const auto max_area_matrix =
    this->declare_parameter<std::vector<double>>("max_area_matrix", descriptor);
  detection_class_remapper_.setParameters(
    allow_remapping_by_area_matrix, min_area_matrix, max_area_matrix);

  const auto out_size_factor = this->declare_parameter<std::int64_t>("out_size_factor", descriptor);
  const auto num_proposals = this->declare_parameter<std::int64_t>("num_proposals", descriptor);

  const float circle_nms_dist_threshold =
    static_cast<float>(this->declare_parameter<double>("circle_nms_dist_threshold", descriptor));
  const auto yaw_norm_thresholds =
    this->declare_parameter<std::vector<double>>("yaw_norm_thresholds", descriptor);

  // Distance-based score thresholds
  const std::vector<double> distance_bin_upper_limits_double =
    this->declare_parameter<std::vector<double>>(
      "detection_score_thresholds.distance_bin_upper_limits", std::vector<double>{});
  // Must set at least one upper bound
  if (distance_bin_upper_limits_double.empty()) {
    throw std::invalid_argument(
      "The number of upper bounds: detection_score_thresholds.distance_bin_upper_limits must be at "
      "least one");
  }
  const std::vector<float> distance_bin_upper_limits(
    distance_bin_upper_limits_double.begin(), distance_bin_upper_limits_double.end());

  // Create empty vector of thresholds for each class * number of upper bounds
  std::vector<float> score_thresholds =
    std::vector<float>(class_names_.size() * distance_bin_upper_limits.size(), 0.0);
  int current_class_index = 0;
  for (const auto & class_name : class_names_) {
    // Construct the parameter path (e.g., "detection_score_thresholds.min_confidence_scores.CAR")
    std::string param_path = "detection_score_thresholds.min_confidence_scores." + class_name;

    // The same class name may appear multiple times in class_names_, so only declare the parameter
    // on the first occurrence and reuse the already-declared value afterwards.
    std::vector<double> class_score_thresholds =
      this->has_parameter(param_path)
        ? this->get_parameter(param_path).as_double_array()
        : this->declare_parameter<std::vector<double>>(param_path, std::vector<double>{});
    if (class_score_thresholds.size() != distance_bin_upper_limits.size()) {
      throw std::invalid_argument(
        "The number of thresholds for " + class_name +
        " is not equal to the number of upper bounds");
    }

    // Move it to the correct position in the 1d-vector score_thresholds, where the order is number
    // of classes * number of upper bounds
    int current_upper_bound_index = 0;
    for (auto class_score_threshold : class_score_thresholds) {
      // The index is the current class index + the current upper bound index * the number of
      // classes since score thresholds for the same class are in the same column For example, #
      // CAR, TRUCK, BUS, BICYCLE, PEDESTRIAN
      // [
      //  0.35, 0.35, 0.35, 0.35, 0.35,   # 0-50m
      //  0.35, 0.35, 0.35, 0.35, 0.35,   # 50.0-90m
      //  0.35, 0.35, 0.35, 0.35, 0.35,   # 90.0-121.0m
      //  0.35, 0.35, 0.35, 0.35, 0.35    # 121.0-200.0m
      // ]
      auto score_threshold_index =
        current_class_index + current_upper_bound_index * class_names_.size();
      score_thresholds[score_threshold_index] = class_score_threshold;
      current_upper_bound_index++;
    }
    current_class_index++;
  }

  base_config_.emplace(
    class_names_.size(), plugins_path, out_size_factor, num_proposals, circle_nms_dist_threshold,
    yaw_norm_thresholds, score_thresholds, distance_bin_upper_limits);

  densification_param_.emplace(densification_world_frame_id, densification_num_past_frames);

  diagnostics_detector_trt_ =
    std::make_unique<autoware_utils_diagnostics::DiagnosticsInterface>(this, "bevfusion_trt");

  objects_pub_ = this->create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
    "~/output/objects", rclcpp::QoS(1));

  published_time_pub_ = std::make_unique<autoware_utils_debug::PublishedTimePublisher>(this);

  {
    using autoware_utils_debug::DebugPublisher;
    using autoware_utils_system::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, this->get_name());
    stop_watch_ptr_->tic("cyclic");
    stop_watch_ptr_->tic("processing/total");
  }

  build_only_ = this->declare_parameter<bool>("build_only", false, descriptor);
}

TrtBEVFusionConfig BEVFusionNode::makeTrtConfig(
  const std::optional<tensorrt_common::TrtCommonConfig> & image_backbone_trt_config) const
{
  auto trt_main_config =
    tensorrt_common::TrtCommonConfig(onnx_path_, trt_precision_, engine_path_, 1ULL << 32U);
  return TrtBEVFusionConfig{trt_main_config, image_backbone_trt_config};
}

void BEVFusionNode::shutdownIfBuildOnly()
{
  if (build_only_) {
    RCLCPP_INFO(this->get_logger(), "TensorRT engine was built. Shutting down the node.");
    rclcpp::shutdown();
  }
}

bool BEVFusionNode::hasOutputSubscribers() const
{
  const auto objects_sub_count =
    objects_pub_->get_subscription_count() + objects_pub_->get_intra_process_subscription_count();
  return objects_sub_count > 0;
}

void BEVFusionNode::detectAndPublish(
  const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
  const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
  const std::vector<float> & camera_masks)
{
  if (stop_watch_ptr_) {
    stop_watch_ptr_->toc("processing/total", true);
  }

  diagnostics_detector_trt_->clear();

  std::vector<Box3D> det_boxes3d;
  std::unordered_map<std::string, double> proc_timing;
  bool is_num_voxels_within_range = true;
  const bool is_success = detector_ptr_->detect(
    pc_msg_ptr, camera_data_ptrs, camera_masks, tf_buffer_, det_boxes3d, proc_timing,
    is_num_voxels_within_range);

  if (!is_success) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "BEVFusion detection failed. No detection results will be published.");
    return;
  }

  diagnostics_detector_trt_->add_key_value(
    "is_num_voxels_within_range", is_num_voxels_within_range);
  if (!is_num_voxels_within_range) {
    std::stringstream message;
    message << "BEVFusionTRT::detect: The actual number of voxels exceeds its maximum value, "
            << "which may limit the detection performance.";
    diagnostics_detector_trt_->update_level_and_message(
      diagnostic_msgs::msg::DiagnosticStatus::WARN, message.str());
  }

  std::vector<autoware_perception_msgs::msg::DetectedObject> raw_objects;
  raw_objects.reserve(det_boxes3d.size());
  for (const auto & box3d : det_boxes3d) {
    autoware_perception_msgs::msg::DetectedObject obj;
    box3DToDetectedObject(box3d, class_names_, obj);
    raw_objects.emplace_back(obj);
  }

  autoware_perception_msgs::msg::DetectedObjects output_msg;
  output_msg.header = pc_msg_ptr->header;
  output_msg.objects = iou_bev_nms_.apply(raw_objects);

  detection_class_remapper_.mapClasses(output_msg);

  publishDetectionResults(output_msg, pc_msg_ptr->header);
  publishDebugInfo(proc_timing, output_msg.header);
}

}  // namespace autoware::bevfusion
