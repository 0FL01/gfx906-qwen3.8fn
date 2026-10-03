#include "hip/dense.cuh"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void check(rocblas_status s) { if (s != rocblas_status_success) throw std::runtime_error("rocBLAS failure"); }
struct Buffer {
    float* p = nullptr;
    explicit Buffer(std::size_t n) { check(hipMalloc(&p, n * sizeof(float))); }
    ~Buffer() { if (p && hipFree(p) != hipSuccess) std::abort(); }
};
struct Fixture {
    hipStream_t stream{};
    rocblas_handle handle{};
    Fixture() {
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        try { check(rocblas_create_handle(&handle)); }
        catch (...) { if (hipStreamDestroy(stream) != hipSuccess) std::abort(); throw; }
    }
    ~Fixture() {
        if (hipStreamSynchronize(stream) != hipSuccess ||
            rocblas_destroy_handle(handle) != rocblas_status_success ||
            hipStreamDestroy(stream) != hipSuccess) std::abort();
    }
    void run(int k, int n, int alignment) {
        constexpr int m = 3;
        const int wo = alignment == 1, xo = alignment == 2;
        std::vector<float> w(k * m + wo), x(k * n + xo), y(m * n + 2, -12345.0f);
        for (int i = 0; i < k * m; ++i) w[wo + i] = float((i * 7 + 3) % 15 - 7) / 64.0f;
        for (int i = 0; i < k * n; ++i) x[xo + i] = float((i * 11 + 5) % 15 - 7) / 64.0f;
        Buffer dw(w.size()), dx(x.size()), dy(y.size());
        check(hipMemcpyAsync(dw.p, w.data(), w.size() * 4, hipMemcpyHostToDevice, stream));
        check(hipMemcpyAsync(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice, stream));
        check(hipMemcpyAsync(dy.p, y.data(), y.size() * 4, hipMemcpyHostToDevice, stream));
        check(qwen::launch_dense_linear(handle, {dw.p + wo, k, m}, dx.p + xo, n, dy.p + 1, stream));
        check(hipMemcpyAsync(y.data(), dy.p, y.size() * 4, hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        if (y.front() != -12345.0f || y.back() != -12345.0f) throw std::runtime_error("output canary");
        for (int c = 0; c < n; ++c) for (int r = 0; r < m; ++r) {
            float reference = 0.0f;
            for (int i = 0; i < k; ++i) reference += w[wo + r * k + i] * x[xo + c * k + i];
            // Products/sums of these dyadic integers are exact in FP32.
            if (!std::isfinite(y[1 + c * m + r]) || y[1 + c * m + r] != reference)
                throw std::runtime_error("MMVF/fallback boundary parity");
        }
    }
};
}
int main() {
    try {
        int count = 0, cases = 0; check(hipGetDeviceCount(&count));
        if (count != 2) throw std::runtime_error("requires two GPU devices");
        for (int d = 0; d < 2; ++d) {
            check(hipSetDevice(d));
            {
                Fixture f;
                // MMVF N1..8; N9 and N128 retain the SGEMM column seam/limit.
                // K128/130 straddle the one/two-wave seam; K7 and offsets
                // independently exercise odd-K and weight/input alignment fallback.
                for (int k : {2, 7, 64, 128, 130, 320, 10240, 16384})
                    for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 128})
                        for (int a : {0, 1, 2}) { f.run(k, n, a); ++cases; }
            }
        }
        if (cases != 480) throw std::runtime_error("coverage");
        std::printf("{\"kind\":\"dense_mmvf_boundaries\",\"devices\":2,\"cases\":480,\"exact_numeric\":true,\"canaries\":true,\"passed\":true}\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "dense-mmvf: %s\n", e.what()); return 1; }
}
