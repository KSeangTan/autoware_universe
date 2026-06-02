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
#include "autoware/bevfusion/utils.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Contains the shared helper implementations of the BEVFusionNode base class (parameter validation,
// result/debug publishing and processing-time diagnostics). Separated to reduce the code complexity
// of bevfusion_node.cpp.
namespace autoware::bevfusion
{

void BEVFusionNode::publishDetectionResults(
  const autoware_perception_msgs::msg::DetectedObjects & output_msg,
  const std_msgs::msg::Header & header)
{
  const auto objects_sub_count =
    objects_pub_->get_subscription_count() + objects_pub_->get_intra_process_subscription_count();

  if (objects_sub_count > 0) {
    objects_pub_->publish(output_msg);
    published_time_pub_->publish_if_subscribed(objects_pub_, output_msg.header.stamp);
  }

  diagnostics_detector_trt_->publish(header.stamp);
}

void BEVFusionNode::publishDebugInfo(
  const std::unordered_map<std::string, double> & proc_timing, const std_msgs::msg::Header & header)
{
  if (!debug_publisher_ptr_ || !stop_watch_ptr_) {
    return;
  }

  const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic", true);
  const double processing_time_ms = stop_watch_ptr_->toc("processing/total", true);
  const double pipeline_latency_ms =
    std::chrono::duration<double, std::milli>(
      std::chrono::nanoseconds((this->get_clock()->now() - header.stamp).nanoseconds()))
      .count();

  debug_publisher_ptr_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
    "debug/cyclic_time_ms", cyclic_time_ms);
  debug_publisher_ptr_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
    "debug/pipeline_latency_ms", pipeline_latency_ms);
  debug_publisher_ptr_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
    "debug/processing_time/total_ms", processing_time_ms);

  for (const auto & [topic, time_ms] : proc_timing) {
    debug_publisher_ptr_->publish<autoware_internal_debug_msgs::msg::Float64Stamped>(
      topic, time_ms);
  }
}

void BEVFusionNode::addNoInferenceDiagnostics(
  diagnostic_updater::DiagnosticStatusWrapper & stat, std::stringstream & message)
{
  stat.add("is_processing_time_ms_in_expected_range", true);
  stat.add("processing_time_ms", 0.0);
  stat.add("is_consecutive_processing_delay_in_range", true);
  stat.add("consecutive_processing_delay_ms", 0.0);
  message << "Waiting for the node to perform inference.";
}

diagnostic_msgs::msg::DiagnosticStatus::_level_type BEVFusionNode::checkProcessingTimeStatus(
  diagnostic_updater::DiagnosticStatusWrapper & stat, std::stringstream & message,
  const rclcpp::Time & timestamp_now)
{
  if (last_processing_time_ms_ > max_allowed_processing_time_ms_) {
    stat.add("is_processing_time_ms_in_expected_range", false);

    message.clear();
    message << "Processing time exceeds the acceptable limit of " << max_allowed_processing_time_ms_
            << " ms by "
            << (last_processing_time_ms_.value_or(0.0) - max_allowed_processing_time_ms_) << " ms.";

    if (!last_in_time_processing_timestamp_) {
      last_in_time_processing_timestamp_ = timestamp_now;
    }

    return diagnostic_msgs::msg::DiagnosticStatus::WARN;
  }

  stat.add("is_processing_time_ms_in_expected_range", true);
  last_in_time_processing_timestamp_ = timestamp_now;
  return diagnostic_msgs::msg::DiagnosticStatus::OK;
}

diagnostic_msgs::msg::DiagnosticStatus::_level_type BEVFusionNode::checkConsecutiveDelays(
  diagnostic_updater::DiagnosticStatusWrapper & stat, std::stringstream & message,
  const rclcpp::Time & timestamp_now,
  diagnostic_msgs::msg::DiagnosticStatus::_level_type current_level)
{
  // if the last in time processing timestamp is not set, return the current level
  if (!last_in_time_processing_timestamp_) {
    return current_level;
  }
  const double delayed_state_duration =
    std::chrono::duration<double, std::milli>(
      std::chrono::nanoseconds(
        (timestamp_now - last_in_time_processing_timestamp_.value()).nanoseconds()))
      .count();

  if (delayed_state_duration > max_acceptable_consecutive_delay_ms_) {
    stat.add("is_consecutive_processing_delay_in_range", false);
    message << " Processing delay has consecutively exceeded the acceptable limit continuously.";
    stat.add("consecutive_processing_delay_ms", delayed_state_duration);
    return diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  }

  stat.add("is_consecutive_processing_delay_in_range", true);
  stat.add("consecutive_processing_delay_ms", delayed_state_duration);
  return current_level;
}

// Check the processing time and delayed timestamp
// If the node is consistently delayed, publish an error diagnostic message
void BEVFusionNode::diagnoseProcessingTime(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  const rclcpp::Time timestamp_now = this->get_clock()->now();
  diagnostic_msgs::msg::DiagnosticStatus::_level_type diag_level =
    diagnostic_msgs::msg::DiagnosticStatus::OK;
  std::stringstream message{"OK"};

  if (!last_processing_time_ms_) {
    addNoInferenceDiagnostics(stat, message);
  } else {
    diag_level = checkProcessingTimeStatus(stat, message, timestamp_now);
    stat.add("processing_time_ms", last_processing_time_ms_.value_or(0.0));
    diag_level = checkConsecutiveDelays(stat, message, timestamp_now, diag_level);
  }

  stat.summary(diag_level, message.str());
}

}  // namespace autoware::bevfusion
