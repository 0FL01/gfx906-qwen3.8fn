#pragma once
#include <cstddef>
namespace qwen {
void gpu_probe();
double cpu_read_fma(const float *data, std::size_t begin, std::size_t end);
}
