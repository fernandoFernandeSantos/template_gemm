/*
 * GEMM.h
 *
 *  Created on: 10/08/2018
 *      Author: fernando
 */

#ifndef GEMM_H_
#define GEMM_H_

#include <stdlib.h>
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// For half precision computation
//#include "half.hpp"
#include <cuda_fp16.h>

//KERNELS DEFINITION
/**
 * matrix_mul for half2
 */
//template<>
__global__ void matrix_mul(half* C, half* A, half* B, size_t w_a, size_t w_b) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = w_a * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + w_a - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * w_b;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	half2 Csub = __float2half2_rn(0.0);

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ half2 As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ half2 Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = __half2half2(A[a + w_a * ty + tx]);
		Bs[ty][tx] = __half2half2(B[b + w_b * ty + tx]);

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			//__hfma2( __half2half2( d_A0[ty * n + k] ), __half2half2( d_B0[k * (n / 2) + tx] ), acc);
			__half2 a = __half2half2(((half*)As)[ty * w_a + k]);
			__half2 b = __half2half2(((half*)Bs)[k * (w_b / 2) + tx]);
			Csub = __hfma2(a, b, Csub);
//			Csub = __hfma2(__half2half2(As[ty * w_a + k]),
//					__half2half2(Bs[k * (w_b / 2) + tx]), Csub);
			//Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = w_b * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	((half2*)C)[c + (w_b / 2) * ty + tx] = Csub;
}

/**
 * matrix_mul for Double and Single
 */
template<class T>
__global__ void matrix_mul(T* C, T* A, T* B, size_t w_a, size_t w_b) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = w_a * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + w_a - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * w_b;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	T Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ T As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + w_a * ty + tx];
		Bs[ty][tx] = B[b + w_b * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = w_b * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + w_b * ty + tx] = Csub;
}

namespace radiation {

//ERROR functions definitions
#define check_framework_errors(error) __check_framework_errors(error, __LINE__, __FILE__)
#define error(error) __error(error, __LINE__, __FILE__)

void __check_framework_errors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

void __error(const char* error, int line, const char* file) {
	printf("%s - Line: %d at %s\n", error, line, file);
	exit (EXIT_FAILURE);
}

template<class T>
class GEMM {
public:

	// Memory pointers to device and host data
	T* device_ptr_a = nullptr;
	T* device_ptr_b = nullptr;
	T* device_ptr_c = nullptr;

	// Size of the matrix
	size_t cols_a, rows_a;
	size_t cols_b, rows_b;
	size_t cols_c, rows_c;

	/**
	 * Class constructor
	 */
	GEMM(const T* host_ptr_a, const T* host_ptr_b, const T* host_ptr_c,
			size_t rows_a, size_t cols_a, size_t cols_b) {

		this->rows_a = rows_a;
		this->cols_a = cols_a;
		this->rows_b = this->cols_a;
		this->cols_b = cols_b;
		this->cols_c = this->rows_a;
		this->rows_c = this->cols_b;

		if (rows_a > 0 && cols_a > 0 && cols_b > 0) {
			this->debug("device memory allocation");
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_a),
							this->rows_a * this->cols_a * sizeof(T)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_b),
							this->rows_b * this->cols_b * sizeof(T)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_c),
							this->rows_c * this->cols_c * sizeof(T)));

			this->debug("push memory to device");
			//set 0 to C matrix
			this->push_arrays(host_ptr_a, host_ptr_b);
		} else {
			error("columns or rows equal to zero, or less than zero");
		}
	}

	/**
	 * Destructor for the GEMM class
	 */
	virtual ~GEMM() {

		this->debug("destructor");
		if (this->device_ptr_a != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_a));

		if (this->device_ptr_b != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_b));

		if (this->device_ptr_c != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_c));
	}

	/**
	 * PUSH arrays to gpu and set 0x0 to C matrix
	 */
	void push_arrays(const T* host_ptr_a, const T* host_ptr_b) {

		this->debug("memset array C");
		//set 0 to C matrix
		check_framework_errors(
				cudaMemset(this->device_ptr_c, 0x00,
						this->rows_c * this->cols_c * sizeof(T)));

		this->debug("memcpy array A");
		//PUSH A
		check_framework_errors(
				cudaMemcpy(this->device_ptr_a, host_ptr_a,
						this->rows_a * this->cols_a * sizeof(T),
						cudaMemcpyHostToDevice));

		this->debug("memcpy array B");
		//PUSH B
		check_framework_errors(
				cudaMemcpy(this->device_ptr_b, host_ptr_b,
						this->rows_b * this->cols_b * sizeof(T),
						cudaMemcpyHostToDevice));
	}

	/**
	 * PULL C array to host
	 */

	void pull_array(T* host_ptr_c) {

		this->debug("memcpy array C to host");
		// PULL C
		check_framework_errors(
				cudaMemcpy(host_ptr_c, this->device_ptr_c,
						this->rows_c * this->cols_c * sizeof(T),
						cudaMemcpyDeviceToHost));
	}

	/**
	 * Template multiplication
	 */
	void mul() {

		this->debug("thread dim allocation");
		// Setup execution parameters
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(this->cols_b / threads.x, this->rows_a / threads.y);

		this->debug("matrix multiplication");
		matrix_mul<T> <<<grid, threads>>>(this->device_ptr_c,
				this->device_ptr_a, this->device_ptr_b, this->cols_a,
				this->cols_b);

		this->debug("device synchronize");
		check_framework_errors(cudaDeviceSynchronize());

	}

private:
	void debug(const char* message) {
#ifdef DEBUG
		std::cout << "DEBUG: " << message << std::endl;
#endif
	}

};

} /* namespace radiation */

#endif /* GEMM_H_ */
