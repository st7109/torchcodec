// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "MluDeviceInterface.h"
#include "CpuDeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_mlu.h>
}

namespace facebook::torchcodec {

namespace {

// Static registration - register MLU with "default" variant
static bool g_mlu = registerDeviceInterface(
    DeviceInterfaceKey(kStableMLU, "default"),
    [](const StableDevice& device) {
      return new MluDeviceInterface(device);
    });

} // namespace

MluDeviceInterface::MluDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  // Create hardware device context for MLU
  AVBufferRef* hwDeviceCtx = nullptr;
  int err = av_hwdevice_ctx_create(
      &hwDeviceCtx, AV_HWDEVICE_TYPE_MLU, nullptr, nullptr, 0);

  if (err < 0) {
    usingCPUFallback_ = true;
    hardwareDeviceCtx_ = nullptr;
  } else {
    hardwareDeviceCtx_ = UniqueAVBufferRef(hwDeviceCtx);
    usingCPUFallback_ = false;
  }
}

MluDeviceInterface::~MluDeviceInterface() = default;

std::optional<const AVCodec*> MluDeviceInterface::findCodec(
    const AVCodecID& codecId,
    bool isDecoder) {
  if (usingCPUFallback_ || !isDecoder) {
    return std::nullopt;
  }

  // Try to find MLU hardware decoder
  const char* codecName = nullptr;
  switch (codecId) {
    case AV_CODEC_ID_H264:
      codecName = "h264_mludec";
      break;
    case AV_CODEC_ID_HEVC:
      codecName = "hevc_mludec";
      break;
    case AV_CODEC_ID_MJPEG:
      codecName = "mjpeg_mludec";
      break;
    default:
      return std::nullopt;
  }

  const AVCodec* codec = avcodec_find_decoder_by_name(codecName);
  if (codec != nullptr) {
    return codec;
  }

  return std::nullopt;
}

void MluDeviceInterface::initialize(const SharedAVCodecContext& codecContext) {
  codecContext_ = codecContext;
}

void MluDeviceInterface::initializeVideo(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx,
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {
  videoStreamOptions_ = videoStreamOptions;

  // Create CPU fallback interface - fully initialized so it can handle
  // frame-to-tensor conversion for frames transferred from MLU to CPU
  if (!cpuInterface_) {
    cpuInterface_ = createDeviceInterface(StableDevice(kStableCPU), "default");
    STD_TORCH_CHECK(
        cpuInterface_ != nullptr, "Failed to create CPU device interface");
    cpuInterface_->initialize(codecContext_);
  }

  // Always propagate to cpuInterface_ - it handles frame-to-tensor
  // conversion even when MLU hardware acceleration is active
  cpuInterface_->initializeVideo(avStream, avFormatCtx, videoStreamOptions, transforms, resizedOutputDims);
}

void MluDeviceInterface::registerHardwareDeviceWithCodec(
    AVCodecContext* codecContext) {
  if (usingCPUFallback_ || !hardwareDeviceCtx_) {
    return;
  }

  // Attach hardware device context to codec context
  codecContext->hw_device_ctx = av_buffer_ref(hardwareDeviceCtx_.get());
  codecContext->hwaccel_flags |= AV_HWACCEL_FLAG_ALLOW_HIGH_DEPTH;
}

void MluDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::stable::Tensor> preAllocatedOutputTensor) {
  hasDecodedFrame_ = true;

  // When a pre-allocated tensor is provided (e.g. from batch decoding), it is
  // allocated on the MLU device. The CPU interface expects CPU tensors, so we
  // must never pass it through. Instead we let the CPU interface create its own
  // CPU tensor and copy the result back to the MLU tensor afterwards.
  //
  // Two scenarios reach this function:
  //  1. MLU hardware decoding: frame is AV_PIX_FMT_MLU, we transfer to CPU,
  //     then convert via cpuInterface_.
  //  2. CPU fallback (MLU hw not available or unsupported codec): frame is
  //     already on CPU, we convert via cpuInterface_ directly.

  bool frameOnMLU = (avFrame->format == AV_PIX_FMT_MLU);
  UniqueAVFrame cpuFrame;

  if (frameOnMLU && !usingCPUFallback_) {
    // Transfer frame from MLU to CPU
    cpuFrame.reset(av_frame_alloc());
    int err = av_hwframe_transfer_data(cpuFrame.get(), avFrame.get(), 0);
    if (err < 0) {
      // Transfer failed, try CPU fallback
      usingCPUFallback_ = true;
      // Fall through to CPU path below with original avFrame
      frameOnMLU = false;
    } else {
      // Copy metadata from original frame
      cpuFrame->pts = avFrame->pts;
      cpuFrame->pkt_dts = avFrame->pkt_dts;
      cpuFrame->best_effort_timestamp = avFrame->best_effort_timestamp;
      setDuration(cpuFrame, getDuration(avFrame));
    }
  }

  // Convert frame to tensor using CPU interface (never pass pre-allocated
  // tensor since it may be on a different device).
  UniqueAVFrame& frameForCPU = (frameOnMLU && cpuFrame) ? cpuFrame : avFrame;
  cpuInterface_->convertAVFrameToFrameOutput(
      frameForCPU, frameOutput, std::nullopt);

  // If a pre-allocated tensor was provided, copy the result to it
  if (preAllocatedOutputTensor.has_value()) {
    auto& targetTensor = preAllocatedOutputTensor.value();
    auto& sourceTensor = frameOutput.data;

    // Validate shapes before copying
    auto targetSizes = targetTensor.sizes();
    auto sourceSizes = sourceTensor.sizes();
    STD_TORCH_CHECK(
        targetSizes.size() == sourceSizes.size(),
        "Rank mismatch: target ",
        targetSizes.size(),
        " vs source ",
        sourceSizes.size());
    for (size_t i = 0; i < targetSizes.size(); ++i) {
      STD_TORCH_CHECK(
          targetSizes[i] == sourceSizes[i],
          "Shape mismatch at dim ", i, ": target ",
          targetSizes[i], " vs source ", sourceSizes[i]);
    }

    // Copy data from CPU to MLU
    torch::stable::copy_(targetTensor, sourceTensor);
  }
}

std::string MluDeviceInterface::getDetails() {
  if (usingCPUFallback_) {
    return "MLU device interface (CPU fallback active)";
  }
  return "MLU device interface (hardware acceleration active)";
}

} // namespace facebook::torchcodec
