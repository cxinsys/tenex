/*
 * POINT Fused Hash-Entropy CUDA kernel.
 *
 * Replaces the 3-step pipeline (poly_hash → torch.sort → sorted_entropy)
 * with a single fused kernel per batch:
 *   1. Polynomial hash: d columns → 1 int64 per time point
 *   2. Open-addressing hash table: count occurrences (no sort!)
 *   3. Entropy from counts: −Σ (c/L) log₂(c/L)
 *
 * Why this is faster than sort:
 *   - Sort is O(L log L) with large constant (CUB radix sort moves all L elements)
 *   - Hash table insert is O(L) with atomic adds
 *   - Entropy scan is O(U) where U = unique states << L
 *   - Total: O(L + U) vs O(L log L)
 *
 * Hash table size: 2 * L (load factor ~0.5) — fits easily in global memory.
 * Uses open addressing with linear probing on int64 keys.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// ── Fused hash-count-entropy kernel ─────────────────────────────────
//
// One block per job, 256 threads.
// Phase 1: polynomial hash + insert into per-job hash table
// Phase 2: scan hash table, compute entropy via warp reduction
//
// Hash table layout: keys[HT_SIZE], counts[HT_SIZE] per job
// Total memory per job: HT_SIZE * (8 + 4) = 12 * HT_SIZE bytes
//

__device__ __forceinline__ float warp_reduce_sum_ent(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

static constexpr int64_t EMPTY_KEY = 0x7FFFFFFFFFFFFFFFLL;  // sentinel
static constexpr int64_t HASH_P    = 1000000007LL;

__global__ void fused_hash_entropy_kernel(
    const int32_t* __restrict__ data,        // (n_vars * T,) flat
    const int64_t* __restrict__ col_starts,  // (total_cols,)
    const int32_t* __restrict__ job_offsets, // (n_jobs_total + 1,)
    int64_t*       __restrict__ ht_keys,     // (n_batch, HT_SIZE)
    int32_t*       __restrict__ ht_counts,   // (n_batch, HT_SIZE)
    float*         __restrict__ ent,         // (n_batch,)
    int start_job, int n_batch, int L, int HT_SIZE
) {
    const int local = blockIdx.x;
    if (local >= n_batch) return;

    const int tid   = threadIdx.x;
    const int BLOCK = blockDim.x;
    const int WARPS = BLOCK / 32;

    extern __shared__ char smem[];
    float* warp_buf = (float*)smem;

    const int gj    = start_job + local;
    const int d     = job_offsets[gj + 1] - job_offsets[gj];
    const int cbase = job_offsets[gj];

    // Hash table base for this job
    int64_t* keys   = ht_keys   + (int64_t)local * HT_SIZE;
    int32_t* counts = ht_counts + (int64_t)local * HT_SIZE;

    // ── Phase 0: initialize hash table ──────────────────────
    for (int i = tid; i < HT_SIZE; i += BLOCK) {
        keys[i]   = EMPTY_KEY;
        counts[i] = 0;
    }
    __syncthreads();

    // ── Phase 1: hash + insert ──────────────────────────────
    for (int t = tid; t < L; t += BLOCK) {
        // Polynomial hash
        int64_t h = 0;
        for (int j = 0; j < d; j++)
            h = h * HASH_P + (int64_t)data[col_starts[cbase + j] + t];

        // Open addressing insert
        uint32_t slot = (uint32_t)(h * 0x9E3779B97F4A7C15ULL >> 32) % HT_SIZE;
        while (true) {
            int64_t old = atomicCAS((unsigned long long*)&keys[slot],
                                    (unsigned long long)EMPTY_KEY,
                                    (unsigned long long)h);
            if (old == EMPTY_KEY || old == h) {
                atomicAdd(&counts[slot], 1);
                break;
            }
            slot = (slot + 1) % HT_SIZE;
        }
    }
    __syncthreads();

    // ── Phase 2: entropy from counts ────────────────────────
    float inv_L = 1.0f / (float)L;
    float local_e = 0.0f;

    for (int i = tid; i < HT_SIZE; i += BLOCK) {
        int c = counts[i];
        if (c > 0) {
            float p = (float)c * inv_L;
            local_e -= p * log2f(p);
        }
    }

    // Warp reduction
    local_e = warp_reduce_sum_ent(local_e);
    if (tid % 32 == 0)
        warp_buf[tid / 32] = local_e;
    __syncthreads();

    local_e = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    local_e = warp_reduce_sum_ent(local_e);

    if (tid == 0)
        ent[local] = local_e;
}


// ── Launcher ────────────────────────────────────────────────────────

torch::Tensor fused_hash_entropy_launch(
    torch::Tensor data_flat,     // (n_vars * T,) int32 CUDA
    torch::Tensor col_starts,    // (total_cols,) int64 CUDA
    torch::Tensor job_offsets,   // (n_jobs_total + 1,) int32 CUDA
    torch::Tensor ht_keys,       // (n_batch, HT_SIZE) int64 CUDA
    torch::Tensor ht_counts,     // (n_batch, HT_SIZE) int32 CUDA
    int start_job, int n_batch, int L, int HT_SIZE
) {
    auto ent = torch::empty({n_batch},
        data_flat.options().dtype(torch::kFloat32));

    int smem = (256 / 32) * sizeof(float);  // warp_buf only

    fused_hash_entropy_kernel<<<n_batch, 256, smem>>>(
        data_flat.data_ptr<int32_t>(),
        col_starts.data_ptr<int64_t>(),
        job_offsets.data_ptr<int32_t>(),
        ht_keys.data_ptr<int64_t>(),
        ht_counts.data_ptr<int32_t>(),
        ent.data_ptr<float>(),
        start_job, n_batch, L, HT_SIZE);
    return ent;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_hash_entropy_launch", &fused_hash_entropy_launch);
}
