#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <mma.h>

#ifdef ENBALE_TENSOR

//                             N     
//                          ©°©Ð©´        
//           K              ©À©à©È       1         N
//  ©°©Ð©Ð©Ð©Ð©Ð©Ð©Ð©´      ©À©à©È     ©°©´     ©°©Ð©´
//  ©À©à©à©à©à©à©à©à©È      ©À©à©È     ©À©È     ©À©à©È
// M©À©à©à©à©à©à©à©à©È ¡Á  K©À©à©È +  M©À©È =  M©À©à©È    
//  ©À©à©à©à©à©à©à©à©È      ©À©à©È     ©À©È     ©À©à©È
//  ©¸©Ø©Ø©Ø©Ø©Ø©Ø©Ø©¼      ©À©à©È     ©¸©¼     ©¸©Ø©¼
//                          ©À©à©È        
//                          ©¸©Ø©¼
template<int M, int K, int N = 32, int BaseDim = 16, typename Major = nvcuda::wmma::col_major, bool C_Is_Vector = true>
__device__ void WMMA_Relu(const half* a, const half* b, const half* c, half* res) {
    using namespace nvcuda;
    static_assert((M % BaseDim == 0) && (K % BaseDim == 0) && (N % BaseDim == 0), "m, k, n must be integral multiple of 'BaseDim'.");
    constexpr int lda = std::is_same<Major, wmma::col_major>::value ? M : K;
    constexpr int ldb = std::is_same<Major, wmma::col_major>::value ? K : N;
    constexpr int ldc = std::is_same<Major, wmma::col_major>::value ? M : N;
    constexpr wmma::layout_t MajorEnum = std::is_same<Major, wmma::col_major>::value ? wmma::mem_col_major : wmma::mem_row_major;
    wmma::fragment<wmma::matrix_a, BaseDim, BaseDim, BaseDim, half, Major> a_frag;
    wmma::fragment<wmma::matrix_b, BaseDim, BaseDim, BaseDim, half, Major> b_frag;
    wmma::fragment<wmma::accumulator, BaseDim, BaseDim, BaseDim, half> acc_frag;
    if constexpr (C_Is_Vector) {
        unsigned lane;
        asm volatile ("mov.u32 %0, %laneid;" : "=r"(lane));
        if constexpr (std::is_same<Major, wmma::col_major>::value) {
            constexpr int loop_num = (M + 31) / 32;
            for (int i = 0; i < loop_num; i++) {
                int k = i * 32 + lane;
                float value = k < M ? c[k] : 0;
                for (int j = 0; j < N; j++) {
                    if (k < M)
                        res[k + j * M] = value;
                }
            }
        }
        else {
            constexpr int loop_num = (M + 31) / 32;
            constexpr int loop_num2 = (N + 31) / 32;
            for (int i = 0; i < loop_num; i++) {
                int k = i * 32 + lane;
                float value = k < M ? c[k] : 0;
                for (int j = 0; j < 32; j++) {
                    int row = loop_num * 32 + j;
                    float sfl_value = __shfl_sync(0xFFFFFFFFU, value, j);
                    if (row < N) {
                        for (int k = 0; k < loop_num2; k++) {
                            int col = k * 32 + lane;
                            if (col < M)
                                res[row * N + col] = sfl_value;
                        }
                    }
                }
            }
        }
        __syncwarp();
    }
    for (int i = 0; i < M; i += BaseDim) {
        for (int j = 0; j < N; j += BaseDim) {
            int aRow = i;
            int bCol = j;
            if constexpr (C_Is_Vector) {
                if constexpr (std::is_same<Major, wmma::col_major>::value)
                    wmma::load_matrix_sync(acc_frag, res + aRow + bCol * ldc, ldc, MajorEnum);
                else
                    wmma::load_matrix_sync(acc_frag, res + aRow + bCol * ldc, ldc, MajorEnum);
            }
            else {
                if constexpr (std::is_same<Major, wmma::col_major>::value)
                    wmma::load_matrix_sync(acc_frag, c + bCol + aRow * ldc, ldc, MajorEnum);
                else
                    wmma::load_matrix_sync(acc_frag, c + bCol + aRow * ldc, ldc, MajorEnum);
            }
            // Loop over the K-dimension
            for (int k = 0; k < K; k += BaseDim) {
                int aCol = k;
                int bRow = k;

                if constexpr (std::is_same<Major, wmma::col_major>::value) {
                    wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
                    wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
                }
                else {
                    wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
                    wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);
                }
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            for (int i = 0; i < acc_frag.num_elements; i++)
                acc_frag.x[i] = max(0.0f, acc_frag.x[i]);
            if constexpr (std::is_same<Major, wmma::col_major>::value)
                wmma::store_matrix_sync(res + aRow + bCol * ldc, acc_frag, ldc, wmma::mem_col_major);
            else
                wmma::store_matrix_sync(res + bCol + aRow * ldc, acc_frag, ldc, wmma::mem_row_major);
        }
    }
}



template<int M, int K>
__device__ void WMMA_Relu(const float* a, const float* b, const float* c, float* res) {
    using namespace nvcuda;
    const int N = 32, BaseDim = 16;
    static_assert((M % BaseDim == 0) && (K % BaseDim == 0) && (N % BaseDim == 0), "m, k, n must be integral multiple of 'BaseDim'.");
    constexpr int lda = K;
    __shared__ half A[BaseDim * BaseDim];
    __shared__ half B[N * BaseDim];
    __shared__ float C[N * BaseDim];
    wmma::fragment<wmma::matrix_a, BaseDim, BaseDim, BaseDim, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BaseDim, BaseDim, BaseDim, half, wmma::row_major> b_frag1;
    wmma::fragment<wmma::matrix_b, BaseDim, BaseDim, BaseDim, half, wmma::row_major> b_frag2;
    wmma::fragment<wmma::accumulator, BaseDim, BaseDim, BaseDim, float> acc_frag1;
    wmma::fragment<wmma::accumulator, BaseDim, BaseDim, BaseDim, float> acc_frag2;

    unsigned lane; { asm volatile ("mov.u32 %0, %laneid;" : "=r"(lane)); }
    constexpr int loop_num = (M + N - 1) / N;
    for (int i = 0; i < loop_num; i++) {
        int row = i * N;
        int k = row + lane;
        float value = k < M ? c[k] : 0;
        for (int j = 0; j < N; j++) {
            res[row + j] = __shfl_sync(0xFFFFFFFFU, value, j);
        }
    }

    for (int i = 0; i < M; i += BaseDim) {
        int aRow = i;
        int bCol = 0;

        for (int j = 0; j < BaseDim; j++) C[lane + j * N] = res[aRow + j];
        wmma::load_matrix_sync(acc_frag1, C, N, wmma::mem_row_major);
        wmma::load_matrix_sync(acc_frag2, C + BaseDim, N, wmma::mem_row_major);

        // Loop over the K-dimension
        for (int k = 0; k < K; k += BaseDim) {
            int aCol = k;
            int bRow = k;
            for (int loopNum = 0; loopNum < BaseDim * BaseDim; loopNum += N)
            {
                unsigned int task_id = lane + loopNum;
                unsigned int col = aCol + (task_id % BaseDim), row = aRow + (task_id / BaseDim);
                A[task_id] = a[row * K + col];
            }
            __syncwarp();
            wmma::load_matrix_sync(a_frag, A, BaseDim);
            for (int j = 0; j < BaseDim; j++) B[lane + j * N] = b[bRow + j];
            __syncwarp();
            wmma::load_matrix_sync(b_frag1, B, N);
            wmma::load_matrix_sync(b_frag2, B + BaseDim, N);
            wmma::mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
            wmma::mma_sync(acc_frag2, a_frag, b_frag2, acc_frag2);
        }

        wmma::store_matrix_sync(C, acc_frag1, N, wmma::mem_row_major);
        wmma::store_matrix_sync(C + BaseDim, acc_frag2, N, wmma::mem_row_major);
        for (int j = 0; j < BaseDim; j++) res[aRow + j] = max(0.0f, C[lane + j * N]);
        __syncwarp();
    }
}

#endif


//
//const int mm = 64;
//const int mk = 128;
//const int mn = 32;
//
//__global__ void Foo(half* a, half* b, half* c, half* res) {
//
//    //__shared__ half a[mm * mk];
//    //__shared__ half b[mk * mn];
//    //__shared__ half c[mm];
//    //__shared__ half res[mm * mn];
//
//    WMMA_Relu<mm, mk, mn>(a, b, c, res);
//}
//
//void main() {
//
//    half* a, *b, *c, *res;
//
//    cudaMalloc((void**)&a, sizeof(half) * mm * mk);
//    cudaMalloc((void**)&b, sizeof(half) * mk * mn);
//    cudaMalloc((void**)&c, sizeof(half) * mm);
//    cudaMalloc((void**)&res, sizeof(half) * mm * mn);
//
//    half a_[mm * mk];
//    half b_[mk * mn];
//    half c_[mm];
//    half res_[mm * mn];
//
//    for (int i = 0; i < mm; i++)
//    {
//        for (int j = 0; j < mk; j++)
//        {
//            a_[i + j * mm] = (rand() % 1000) / 999.0f - 0.5f;
//        }
//    }
//
//    for (int i = 0; i < mk; i++)
//    {
//        for (int j = 0; j < mn; j++)
//        {
//            b_[i + j * mk] = (rand() % 1000) / 999.0f - 0.5f;
//        }
//    }
//
//    for (int i = 0; i < mm; i++)
//    {
//        c_[i] = (rand() % 1000) / 999.0f - 0.5f;
//    }
//
//    cudaMemcpy(a, a_, sizeof(half) * mm * mk, cudaMemcpyHostToDevice);
//    cudaMemcpy(b, b_, sizeof(half) * mk * mn, cudaMemcpyHostToDevice);
//    cudaMemcpy(c, c_, sizeof(half) * mm, cudaMemcpyHostToDevice);
//
//    Foo << <1, 32 >> > (a, b, c, res);
//
//    cudaMemcpy(res_, res, sizeof(half) * mm * mn, cudaMemcpyDeviceToHost);
//
//    half res__[mm * mn];
//    for (int i = 0; i < mm; i++)
//    {
//        for (int j = 0; j < mn; j++)
//        {
//            float t = c_[i];
//            for (int k = 0; k < mk; k++)
//            {
//                t = t + float(a_[i + k * mm]) * float(b_[j * mk + k]);
//            }
//            res__[i + j * mm] = max(0.0f, t);
//        }
//    }
//
//    for (int i = 0; i < mm * mn; i++)
//    {
//        float cpu = float(res__[i]);
//        float gpu = float(res_[i]);
//        if (abs(cpu - gpu) > 0.003)
//            std::cout << cpu << ", " << gpu << std::endl;
//    }
//
//}
//