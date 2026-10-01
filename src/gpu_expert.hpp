#pragma once
#include "quant.hpp"
#include <memory>
#include <span>
namespace qwen {
struct ExpertTiming { double upload_ms, resident_ms, completed_wall_ms; };
class GpuExpert {
public:
    GpuExpert(QMatrix gate,QMatrix up,QMatrix down,int device,int max_columns,bool planar);
    ~GpuExpert();
    GpuExpert(const GpuExpert&)=delete;
    GpuExpert& operator=(const GpuExpert&)=delete;
    // CPU/GPU packer parity (identical quantized activation bytes), then resident
    // float -> quantization -> gate/up -> SiLU*up -> quantization -> down.
    void check_quantization(std::span<const float> input,std::span<const Q8_1> expected);
    void linear(bool down,bool up,std::span<const Q8_1> input,int columns,std::span<float> output);
    ExpertTiming run(std::span<const float> input,int columns,std::span<float> output,int repeats);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
