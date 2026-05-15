/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>
#include <new>

#if __has_include(<cuda/__cmath/fast_modulo_division.h>)
#include <cuda/__cmath/fast_modulo_division.h>
#define FLASHINFER_HAS_CUDA_FAST_MOD_DIV 1
#elif __has_include(<cub/detail/fast_modulo_division.cuh>)
#include <cub/detail/fast_modulo_division.cuh>
#define FLASHINFER_HAS_CUDA_FAST_MOD_DIV 1
#else
#define FLASHINFER_HAS_CUDA_FAST_MOD_DIV 0
#endif

namespace flashinfer {

#if __has_include(<cuda/__cmath/fast_modulo_division.h>)
using uint_fastdiv_impl = cuda::fast_mod_div<uint32_t>;
#elif __has_include(<cub/detail/fast_modulo_division.cuh>)
using uint_fastdiv_impl = cub::detail::fast_div_mod<uint32_t>;
#endif

// API-compatible wrapper around the fastest modulo/division helper available in
// the active CUDA/CCCL headers. Preserves the default constructor, implicit
// conversions, and divmod() method expected by existing call sites throughout
// the attention kernels.
struct uint_fastdiv {
#if FLASHINFER_HAS_CUDA_FAST_MOD_DIV
  __host__ __device__ uint_fastdiv() : impl_(1), d_(0) {}

  __host__ __device__ explicit uint_fastdiv(uint32_t d) : impl_(d ? d : 1), d_(d) {}

  __host__ __device__ uint_fastdiv(const uint_fastdiv& other)
      : impl_(other.d_ ? other.d_ : 1), d_(other.d_) {}

  __host__ __device__ uint_fastdiv& operator=(const uint_fastdiv& other) {
    d_ = other.d_;
    new (&impl_) uint_fastdiv_impl(d_ ? d_ : 1);
    return *this;
  }

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / impl_;
    r = n - q * d_;
  }

 private:
  uint_fastdiv_impl impl_;
#else
  __host__ __device__ uint_fastdiv() : d_(0) {}

  __host__ __device__ explicit uint_fastdiv(uint32_t d) : d_(d) {}

  __host__ __device__ uint_fastdiv(const uint_fastdiv& other) = default;
  __host__ __device__ uint_fastdiv& operator=(const uint_fastdiv& other) = default;

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / d_;
    r = n - q * d_;
  }

 private:
#endif
  uint32_t d_;
};

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}  // namespace flashinfer

#endif  // FLASHINFER_FASTDIV_CUH_
