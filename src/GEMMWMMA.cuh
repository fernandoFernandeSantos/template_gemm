/*
 * GEMMWMMA.h
 *
 *  Created on: 12/08/2018
 *      Author: fernando
 */

#ifndef GEMMWMMA_H_
#define GEMMWMMA_H_

#include "GEMM.cuh"
#include <mma.h>

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
template<class T>
__global__ void wmma_matrix_mul(T *a, T *b, float *c, int M, int N, int K) {
	// Leading dimensions. Packed with no transpositions.
	int lda = M;
	int ldb = K;
	int ldc = M;

	// Tile using a 2D grid
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	nvcuda::wmma::fragment < nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major
			> a_frag;
	nvcuda::wmma::fragment < nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major
			> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			float> acc_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			float> c_frag;

	nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

	// Loop over k
	for (int i = 0; i < K; i += WMMA_K) {
		int aRow = warpM * WMMA_M;
		int aCol = i;

		int bRow = i;
		int bCol = warpN * WMMA_N;

		// Bounds checking
		if (aRow < M && aCol < K && bRow < K && bCol < N) {
			// Load the inputs
			nvcuda::wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
			nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

			// Perform the matrix multiplication
			nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

		}
	}

	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
	int cRow = warpM * WMMA_M;
	int cCol = warpN * WMMA_N;

	if (cRow < M && cCol < N) {
		nvcuda::wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc,
				wmma::mem_col_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
		}

		// Store the output
		nvcuda::wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc,
				wmma::mem_col_major);
	}
}

namespace radiation {

class GEMMWMMA: public GEMM {
public:
	GEMMWMMA::GEMMWMMA() {
		// TODO Auto-generated constructor stub

	}

	virtual GEMMWMMA::~GEMMWMMA() {
		// TODO Auto-generated destructor stub
	}
};

} /* namespace radiation */

#endif /* GEMMWMMA_H_ */
