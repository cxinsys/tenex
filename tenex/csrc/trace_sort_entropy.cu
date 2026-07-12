/*
 * POINT Sort-Entropy CUDA kernels v5.
 *
 * v5: Adaptive dtype — int16 data when max_bin < 32768 (2× less BW).
 *     Hash kernels are templated on data type (int16_t or int32_t).
 *     Launchers auto-dispatch based on data_flat.dtype().
 *
 * v3 (retained): 4-way hash fusion, bit-for-bit identical results.
 *
 * Hash collision note
 * -------------------
 * The polynomial hash uses int64 wrap-around with a fixed prime
 * HASH_P = 1_000_000_007.  Two distinct multivariate states could in
 * principle alias under modulo-2^64 wrapping when d (network in/out
 * degree) is large.  In practice this has been validated as bit-for-bit
 * identical against the gold-standard `np.unique(structured_dtype)`
 * reference on 4 datasets (mESC, Skin, Zebrafish, CeNGEN; max_bin up to
 * 318) across RTX 2080 Ti / A5000 / Blackwell GPUs.  Cross-GPU result
 * `np.array_equal == True`.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <math.h>

static constexpr int64_t HASH_P = 1000000007LL;

// ── Templated 4-way hash kernels ────────────────────────────────────

template <typename T>
__global__ void outte_4way_hash_kernel(
    const T*       __restrict__ data,
    const int64_t* __restrict__ yf_starts,
    const int64_t* __restrict__ yp_starts,
    const int64_t* __restrict__ xp_starts,
    const int32_t* __restrict__ d_vals,
    const int32_t* __restrict__ d_offsets,
    int64_t*       __restrict__ hashes,
    int start_gene, int n_batch, int L
) {
    int local = blockIdx.x;
    if (local >= n_batch) return;

    int g     = start_gene + local;
    int d     = d_vals[g];
    int cbase = d_offsets[g];

    int64_t P_d = 1;
    for (int j = 0; j < d; j++) P_d *= HASH_P;

    int64_t out_base = (int64_t)local * 4 * L;

    for (int t = threadIdx.x; t < L; t += blockDim.x) {
        int64_t h_yf = 0, h_yp = 0;
        for (int j = 0; j < d; j++) {
            h_yf = h_yf * HASH_P + (int64_t)data[yf_starts[cbase + j] + t];
            h_yp = h_yp * HASH_P + (int64_t)data[yp_starts[cbase + j] + t];
        }
        int64_t h_xp = (int64_t)data[xp_starts[g] + t];

        hashes[out_base + 0*L + t] = h_yf * P_d + h_yp;
        hashes[out_base + 1*L + t] = h_yp;
        hashes[out_base + 2*L + t] = (h_yf * HASH_P + h_xp) * P_d + h_yp;
        hashes[out_base + 3*L + t] = h_xp * P_d + h_yp;
    }
}

template <typename T>
__global__ void inte_4way_hash_kernel(
    const T*       __restrict__ data,
    const int64_t* __restrict__ yp_starts,
    const int64_t* __restrict__ xf_starts,
    const int64_t* __restrict__ xp_starts,
    const int32_t* __restrict__ d_vals,
    const int32_t* __restrict__ d_offsets,
    int64_t*       __restrict__ hashes,
    int start_gene, int n_batch, int L
) {
    int local = blockIdx.x;
    if (local >= n_batch) return;

    int g     = start_gene + local;
    int d     = d_vals[g];
    int cbase = d_offsets[g];

    int64_t P_d = 1;
    for (int j = 0; j < d; j++) P_d *= HASH_P;

    int64_t out_base = (int64_t)local * 4 * L;

    for (int t = threadIdx.x; t < L; t += blockDim.x) {
        int64_t h_yp = 0;
        for (int j = 0; j < d; j++)
            h_yp = h_yp * HASH_P + (int64_t)data[yp_starts[cbase + j] + t];

        int64_t h_xf = (int64_t)data[xf_starts[g] + t];
        int64_t h_xp = (int64_t)data[xp_starts[g] + t];
        int64_t h_xfxp = h_xf * HASH_P + h_xp;

        hashes[out_base + 0*L + t] = h_xfxp;
        hashes[out_base + 1*L + t] = h_xp;
        hashes[out_base + 2*L + t] = h_xfxp * P_d + h_yp;
        hashes[out_base + 3*L + t] = h_xp * P_d + h_yp;
    }
}

// Legacy (non-templated, int32 only)
__global__ void poly_hash_kernel(
    const int32_t* __restrict__ data,
    const int64_t* __restrict__ col_starts,
    const int32_t* __restrict__ job_offsets,
    int64_t*       __restrict__ hashes,
    int start_job, int n_batch, int L
) {
    int local = blockIdx.x;
    if (local >= n_batch) return;
    int gj    = start_job + local;
    int d     = job_offsets[gj + 1] - job_offsets[gj];
    int cbase = job_offsets[gj];
    int64_t obase = (int64_t)local * L;
    for (int t = threadIdx.x; t < L; t += blockDim.x) {
        int64_t h = 0;
        for (int j = 0; j < d; j++)
            h = h * HASH_P + (int64_t)data[col_starts[cbase + j] + t];
        hashes[obase + t] = h;
    }
}


// ── Entropy from sorted hashes ──────────────────────────────────────

__global__ void sorted_entropy_kernel(
    const int64_t* __restrict__ sorted,
    float*         __restrict__ ent,
    int n_batch, int L
) {
    int job = blockIdx.x;
    if (job >= n_batch || threadIdx.x != 0) return;

    int64_t base = (int64_t)job * L;
    float inv_L  = 1.0f / (float)L;
    float e      = 0.0f;
    int   count  = 1;

    for (int t = 1; t < L; t++) {
        if (sorted[base + t] != sorted[base + t - 1]) {
            float p = (float)count * inv_L;
            e -= p * log2f(p);
            count = 1;
        } else {
            count++;
        }
    }
    float p = (float)count * inv_L;
    e -= p * log2f(p);
    ent[job] = e;
}


// ── CUB segmented sort helper ───────────────────────────────────────

static void cub_segmented_sort(
    int64_t* d_keys_in, int64_t* d_keys_out,
    int n_segs, int L, cudaStream_t stream,
    torch::Device dev
) {
    int total = n_segs * L;
    auto seg_offsets = torch::arange(
        0, (int64_t)(n_segs + 1) * L, (int64_t)L,
        torch::TensorOptions().dtype(torch::kInt32).device(dev));

    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, temp_bytes,
        d_keys_in, d_keys_out, total, n_segs,
        seg_offsets.data_ptr<int32_t>(),
        seg_offsets.data_ptr<int32_t>() + 1,
        0, 64, stream);

    auto temp = torch::empty({(int64_t)temp_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(dev));

    cub::DeviceSegmentedRadixSort::SortKeys(
        temp.data_ptr<uint8_t>(), temp_bytes,
        d_keys_in, d_keys_out, total, n_segs,
        seg_offsets.data_ptr<int32_t>(),
        seg_offsets.data_ptr<int32_t>() + 1,
        0, 64, stream);
}


// ── Dtype-dispatching macro ─────────────────────────────────────────

#define DISPATCH_DATA_DTYPE(data_flat, KERNEL, ...)                       \
    do {                                                                  \
        if (data_flat.scalar_type() == torch::kInt16) {                   \
            KERNEL<int16_t><<<__VA_ARGS__>>>;                             \
        } else {                                                          \
            KERNEL<int32_t><<<__VA_ARGS__>>>;                             \
        }                                                                 \
    } while (0)


// ── Fused OutTE 4-way launch (int16/int32 adaptive) ─────────────────

torch::Tensor fused_outte_4way_launch(
    torch::Tensor data_flat,
    torch::Tensor yf_starts,
    torch::Tensor yp_starts,
    torch::Tensor xp_starts,
    torch::Tensor d_vals,
    torch::Tensor d_offsets,
    int start_gene, int n_batch, int L
) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto dev = data_flat.device();
    int n_segs = n_batch * 4;

    auto hashes = torch::empty({n_segs, L},
        torch::TensorOptions().dtype(torch::kInt64).device(dev));

    if (data_flat.scalar_type() == torch::kInt16) {
        outte_4way_hash_kernel<int16_t><<<n_batch, 256, 0, stream>>>(
            data_flat.data_ptr<int16_t>(),
            yf_starts.data_ptr<int64_t>(),
            yp_starts.data_ptr<int64_t>(),
            xp_starts.data_ptr<int64_t>(),
            d_vals.data_ptr<int32_t>(),
            d_offsets.data_ptr<int32_t>(),
            hashes.data_ptr<int64_t>(),
            start_gene, n_batch, L);
    } else {
        outte_4way_hash_kernel<int32_t><<<n_batch, 256, 0, stream>>>(
            data_flat.data_ptr<int32_t>(),
            yf_starts.data_ptr<int64_t>(),
            yp_starts.data_ptr<int64_t>(),
            xp_starts.data_ptr<int64_t>(),
            d_vals.data_ptr<int32_t>(),
            d_offsets.data_ptr<int32_t>(),
            hashes.data_ptr<int64_t>(),
            start_gene, n_batch, L);
    }

    auto sorted = torch::empty_like(hashes);
    cub_segmented_sort(
        hashes.data_ptr<int64_t>(), sorted.data_ptr<int64_t>(),
        n_segs, L, stream, dev);

    auto ent = torch::zeros({n_segs},
        torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    sorted_entropy_kernel<<<n_segs, 1, 0, stream>>>(
        sorted.data_ptr<int64_t>(), ent.data_ptr<float>(), n_segs, L);

    return ent;
}


// ── Fused InTE 4-way launch (int16/int32 adaptive) ──────────────────

torch::Tensor fused_inte_4way_launch(
    torch::Tensor data_flat,
    torch::Tensor yp_starts,
    torch::Tensor xf_starts,
    torch::Tensor xp_starts,
    torch::Tensor d_vals,
    torch::Tensor d_offsets,
    int start_gene, int n_batch, int L
) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto dev = data_flat.device();
    int n_segs = n_batch * 4;

    auto hashes = torch::empty({n_segs, L},
        torch::TensorOptions().dtype(torch::kInt64).device(dev));

    if (data_flat.scalar_type() == torch::kInt16) {
        inte_4way_hash_kernel<int16_t><<<n_batch, 256, 0, stream>>>(
            data_flat.data_ptr<int16_t>(),
            yp_starts.data_ptr<int64_t>(),
            xf_starts.data_ptr<int64_t>(),
            xp_starts.data_ptr<int64_t>(),
            d_vals.data_ptr<int32_t>(),
            d_offsets.data_ptr<int32_t>(),
            hashes.data_ptr<int64_t>(),
            start_gene, n_batch, L);
    } else {
        inte_4way_hash_kernel<int32_t><<<n_batch, 256, 0, stream>>>(
            data_flat.data_ptr<int32_t>(),
            yp_starts.data_ptr<int64_t>(),
            xf_starts.data_ptr<int64_t>(),
            xp_starts.data_ptr<int64_t>(),
            d_vals.data_ptr<int32_t>(),
            d_offsets.data_ptr<int32_t>(),
            hashes.data_ptr<int64_t>(),
            start_gene, n_batch, L);
    }

    auto sorted = torch::empty_like(hashes);
    cub_segmented_sort(
        hashes.data_ptr<int64_t>(), sorted.data_ptr<int64_t>(),
        n_segs, L, stream, dev);

    auto ent = torch::zeros({n_segs},
        torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    sorted_entropy_kernel<<<n_segs, 1, 0, stream>>>(
        sorted.data_ptr<int64_t>(), ent.data_ptr<float>(), n_segs, L);

    return ent;
}


// ── Legacy launchers (int32 only, kept for compatibility) ───────────

torch::Tensor fused_hash_sort_entropy_launch(
    torch::Tensor data_flat,
    torch::Tensor col_starts,
    torch::Tensor job_offsets,
    int start_job, int n_batch, int L
) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto dev = data_flat.device();

    auto hashes = torch::empty({n_batch, L},
        data_flat.options().dtype(torch::kInt64));
    poly_hash_kernel<<<n_batch, 256, 0, stream>>>(
        data_flat.data_ptr<int32_t>(),
        col_starts.data_ptr<int64_t>(),
        job_offsets.data_ptr<int32_t>(),
        hashes.data_ptr<int64_t>(),
        start_job, n_batch, L);

    auto sorted = torch::empty_like(hashes);
    cub_segmented_sort(
        hashes.data_ptr<int64_t>(), sorted.data_ptr<int64_t>(),
        n_batch, L, stream, dev);

    auto ent = torch::zeros({n_batch},
        data_flat.options().dtype(torch::kFloat32));
    sorted_entropy_kernel<<<n_batch, 1, 0, stream>>>(
        sorted.data_ptr<int64_t>(), ent.data_ptr<float>(), n_batch, L);
    return ent;
}

torch::Tensor poly_hash_launch(
    torch::Tensor data_flat, torch::Tensor col_starts,
    torch::Tensor job_offsets, int start_job, int n_batch, int L
) {
    auto hashes = torch::empty({n_batch, L},
        data_flat.options().dtype(torch::kInt64));
    poly_hash_kernel<<<n_batch, 256>>>(
        data_flat.data_ptr<int32_t>(),
        col_starts.data_ptr<int64_t>(),
        job_offsets.data_ptr<int32_t>(),
        hashes.data_ptr<int64_t>(),
        start_job, n_batch, L);
    return hashes;
}

torch::Tensor sorted_entropy_launch(
    torch::Tensor sorted_hashes, int n_batch, int L
) {
    auto ent = torch::zeros({n_batch},
        sorted_hashes.options().dtype(torch::kFloat32));
    sorted_entropy_kernel<<<n_batch, 1>>>(
        sorted_hashes.data_ptr<int64_t>(),
        ent.data_ptr<float>(), n_batch, L);
    return ent;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("poly_hash_launch",               &poly_hash_launch);
    m.def("sorted_entropy_launch",          &sorted_entropy_launch);
    m.def("fused_hash_sort_entropy_launch", &fused_hash_sort_entropy_launch);
    m.def("fused_outte_4way_launch",        &fused_outte_4way_launch);
    m.def("fused_inte_4way_launch",         &fused_inte_4way_launch);
}
