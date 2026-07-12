#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// One block per variable: reduce the row to its mean, then write 1[x > mean].
__global__ void mean_binarize_kernel(
    const float* __restrict__ x,   // (n, T) row-major
    int8_t*      __restrict__ out, // (n, T)
    int n, long T)
{
    const int g = blockIdx.x;
    if (g >= n) return;
    const float* row  = x   + (long)g * T;
    int8_t*      orow = out + (long)g * T;
    const int tid = threadIdx.x, B = blockDim.x, W = B >> 5;
    extern __shared__ float sbuf[];             // W floats

    float s = 0.f;
    for (long t = tid; t < T; t += B) s += row[t];
    s = warp_reduce_sum(s);
    if ((tid & 31) == 0) sbuf[tid >> 5] = s;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < W) ? sbuf[tid] : 0.f;
        v = warp_reduce_sum(v);
        if (tid == 0) sbuf[0] = v / (float)T;   // mean
    }
    __syncthreads();
    const float thr = sbuf[0];
    for (long t = tid; t < T; t += B)
        orow[t] = (row[t] > thr) ? (int8_t)1 : (int8_t)0;
}

// Elementwise 1[x > thr[row]] with a supplied per-variable threshold.
__global__ void threshold_binarize_kernel(
    const float* __restrict__ x,   // (n, T)
    const float* __restrict__ thr, // (n,)
    int8_t*      __restrict__ out, // (n, T)
    int n, long T)
{
    const long total = (long)n * T;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long)gridDim.x * blockDim.x) {
        const int g = (int)(i / T);
        out[i] = (x[i] > thr[g]) ? (int8_t)1 : (int8_t)0;
    }
}

torch::Tensor mean_binarize(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kFloat32,
                "mean_binarize expects a 2-D float32 CUDA tensor");
    const c10::cuda::CUDAGuard guard(x.device());
    x = x.contiguous();
    const int  n = (int)x.size(0);
    const long T = (long)x.size(1);
    auto out = torch::empty({x.size(0), x.size(1)},
                            torch::dtype(torch::kInt8).device(x.device()));
    const int B = 256, W = B / 32;
    mean_binarize_kernel<<<n, B, W * sizeof(float),
                           at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<int8_t>(), n, T);
    return out;
}

torch::Tensor threshold_binarize(torch::Tensor x, torch::Tensor thr) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kFloat32,
                "threshold_binarize expects a 2-D float32 CUDA tensor");
    const c10::cuda::CUDAGuard guard(x.device());
    x = x.contiguous();
    thr = thr.to(torch::kFloat32).contiguous();
    const int  n = (int)x.size(0);
    const long T = (long)x.size(1);
    auto out = torch::empty({x.size(0), x.size(1)},
                            torch::dtype(torch::kInt8).device(x.device()));
    const long total = (long)n * T;
    const int  B = 256;
    const int  grid = (int)std::min<long>((total + B - 1) / B, 65535L);
    threshold_binarize_kernel<<<grid, B, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), thr.data_ptr<float>(), out.data_ptr<int8_t>(), n, T);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_binarize", &mean_binarize, "fused per-variable mean binarize");
    m.def("threshold_binarize", &threshold_binarize, "per-variable threshold binarize");
}
