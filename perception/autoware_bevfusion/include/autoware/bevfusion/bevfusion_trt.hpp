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

#ifndef AUTOWARE__BEVFUSION__BEVFUSION_TRT_HPP_
#define AUTOWARE__BEVFUSION__BEVFUSION_TRT_HPP_

#include "autoware/bevfusion/bevfusion_config.hpp"
#include "autoware/bevfusion/camera/camera_data.hpp"
#include "autoware/bevfusion/lidar/bevfusion_lidar_config.hpp"
#include "autoware/bevfusion/postprocess/postprocess_kernel.hpp"
#include "autoware/bevfusion/utils.hpp"
#include "autoware/bevfusion/visibility_control.hpp"

#include <Eigen/Core>
#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <autoware/cuda_utils/cuda_unique_ptr.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <autoware_utils_system/stop_watch.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2_ros/buffer.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::bevfusion
{

using autoware::cuda_utils::CudaUniquePtr;

struct TrtBEVFusionConfig
{
  tensorrt_common::TrtCommonConfig common;
  std::optional<tensorrt_common::TrtCommonConfig> image_backbone;
};

class NetworkParam
{
public:
  NetworkParam(std::string onnx_path, std::string engine_path, std::string trt_precision)
  : onnx_path_(std::move(onnx_path)),
    engine_path_(std::move(engine_path)),
    trt_precision_(std::move(trt_precision))
  {
  }

  std::string onnx_path() const { return onnx_path_; }
  std::string engine_path() const { return engine_path_; }
  std::string trt_precision() const { return trt_precision_; }

private:
  std::string onnx_path_;
  std::string engine_path_;
  std::string trt_precision_;
};

// Common base of the BEVFusion detector family. It owns the shared main TensorRT engine, the output
// buffers, the post-processing and the detection skeleton (detect -> preProcess / inference /
// postProcess). The modality-specific engine setup (initTrt) and per-frame pre-processing
// (preProcess) are implemented individually by each detector (BEVFusionLidarTRT /
// BEVFusionCameraTRT / BEVFusionCameraLidarTRT), which own their lidar/camera branches. The base
// only provides the shared output/engine boilerplate as reusable helpers and never reaches into the
// branches itself.
class BEVFUSION_PUBLIC BEVFusionTRT
{
public:
  using Matrix4fRowM = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

  virtual ~BEVFusionTRT();

  bool detect(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & input_pointcloud_msg_ptr,
    const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
    const std::vector<float> & camera_masks, const tf2_ros::Buffer & tf_buffer,
    std::vector<Box3D> & det_boxes3d, std::unordered_map<std::string, double> & proc_timing,
    bool & is_num_voxels_within_range);

  // Only the camera / camera-lidar detectors upload camera intrinsics/extrinsics; the base default
  // is a no-op (lidar-only mode has no cameras).
  virtual void setIntrinsicsExtrinsics(
    std::vector<sensor_msgs::msg::CameraInfo> & camera_info_vector,
    std::vector<Matrix4fRowM> & lidar2camera_vector);

protected:
  explicit BEVFusionTRT(const BEVFusionConfig & base_config);

  // Allocate the shared output buffers and build the post-processor. The lidar configuration is
  // required to decode the BEV grid, so the derived detector passes its lidar branch configuration.
  void initOutputs(const BEVFusionLidarConfig & lidar_config);

  // Per-frame pre-processing. Implemented individually by each modality detector using its own
  // branch(es).
  virtual bool preProcess(
    const std::shared_ptr<const cuda_blackboard::CudaPointCloud2> & pc_msg_ptr,
    const std::vector<std::unique_ptr<CameraData>> & camera_data_ptrs,
    const std::vector<float> & camera_masks, const tf2_ros::Buffer & tf_buffer,
    bool & is_num_voxels_within_range) = 0;

  // Run inference. The base default runs only the shared main network; the camera / camera-lidar
  // detectors override it to run the image backbone first.
  virtual bool inference();

  bool postProcess(std::vector<Box3D> & det_boxes3d);

  // Shared engine / buffer helpers used by the per-modality initTrt implementations.
  void clearOutputMemory();
  void addOutputNetworkIO(std::vector<autoware::tensorrt_common::NetworkIO> & network_io) const;
  void createNetwork(
    const TrtBEVFusionConfig & trt_config,
    std::vector<autoware::tensorrt_common::NetworkIO> network_io,
    std::vector<autoware::tensorrt_common::ProfileDims> profile_dims);
  void setOutputTensorAddresses();
  bool runMainNetwork();

  cudaStream_t stream_{nullptr};
  std::unique_ptr<autoware_utils_system::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_{
    nullptr};

  BEVFusionConfig base_config_;

  std::unique_ptr<autoware::tensorrt_common::TrtCommon> network_trt_ptr_{nullptr};
  std::unique_ptr<PostprocessCuda> post_ptr_{nullptr};

  // output buffers
  unsigned int bbox_pred_size_{0};
  CudaUniquePtr<std::int64_t[]> label_pred_output_d_{nullptr};
  CudaUniquePtr<float[]> bbox_pred_output_d_{nullptr};
  CudaUniquePtr<float[]> score_output_d_{nullptr};
};

}  // namespace autoware::bevfusion

#endif  // AUTOWARE__BEVFUSION__BEVFUSION_TRT_HPP_
