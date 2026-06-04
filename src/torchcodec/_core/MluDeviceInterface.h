// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "DeviceInterface.h"

namespace facebook::torchcodec {

// MluDeviceInterface provides hardware-accelerated video decoding on
// Cambricon MLU devices via FFmpeg's AV_HWDEVICE_TYPE_MLU backend.
//
// It uses FFmpeg's hwframe transfer mechanism to move decoded frames from
// MLU device memory (AV_PIX_FMT_MLU) to CPU, where they are converted to
// RGB tensors. The design mirrors CudaDeviceInterface where possible.
class MluDeviceInterface : public DeviceInterface {
 public:
  MluDeviceInterface(const StableDevice& device);

  ~MluDeviceInterface() override;

  std::optional<const AVCodec*> findCodec(
      const AVCodecID& codecId,
      bool isDecoder = true) override;

  void initialize(const SharedAVCodecContext& codecContext) override;

  void initializeVideo(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const VideoStreamOptions& videoStreamOptions,
      const std::vector<std::unique_ptr<Transform>>& transforms,
      const std::optional<FrameDims>& resizedOutputDims) override;

  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::stable::Tensor> preAllocatedOutputTensor) override;

  std::string getDetails() override;

 private:
  // CPU fallback interface used when MLU cannot decode a particular stream
  // (unsupported codec, MLU unavailable, etc.)
  std::unique_ptr<DeviceInterface> cpuInterface_;

  VideoStreamOptions videoStreamOptions_;

  UniqueAVBufferRef hardwareDeviceCtx_;

  bool usingCPUFallback_ = false;
  bool hasDecodedFrame_ = false;
};

} // namespace facebook::torchcodec
