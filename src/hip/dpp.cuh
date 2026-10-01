#pragma once
#include <hip/hip_runtime.h>
// PORT: mx ggml-cuda/common.cuh, dcd685463d597d31f5ca759d32c94592a2740fa4.
// Only gfx906 DPP primitives; see third_party/mx-LICENSE. No ggml dispatch.
namespace qwen {
__device__ __forceinline__ float add_xor1(float x) {
    float y; asm volatile("s_nop 4\n v_add_f32_dpp %0, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                         : "=v"(y):"v"(x):"memory"); return y;
}
__device__ __forceinline__ float add_xor2(float x) {
    float y; asm volatile("s_nop 1\n v_add_f32_dpp %0, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
                         : "=v"(y):"v"(x):"memory"); return y;
}
__device__ __forceinline__ float add_xor8(float x) {
    float y; asm volatile("s_nop 1\n v_add_f32_dpp %0, %1, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
                         : "=v"(y):"v"(x):"memory"); return y;
}
template<int Offset> __device__ __forceinline__ int xfer_xor(int x) {
    int y;
    if constexpr(Offset==4) {
        asm volatile("v_mov_b32 %0, %1\n s_nop 1\n"
                     "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                     "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa"
                     :"=v"(y):"v"(x):"memory");
    } else {
        static_assert(Offset==16);
        asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n s_waitcnt lgkmcnt(0)"
                     :"=v"(y):"v"(x):"memory");
    }
    return y;
}
__device__ __forceinline__ float sum32(float x) {
    x=add_xor1(x); x=add_xor2(x);
    x+=__int_as_float(xfer_xor<4>(__float_as_int(x))); x=add_xor8(x);
    return x+__int_as_float(xfer_xor<16>(__float_as_int(x)));
}
__device__ __forceinline__ float sum64(float x) {
    x=sum32(x); return x+__shfl_xor(x,32,64);
}
}
