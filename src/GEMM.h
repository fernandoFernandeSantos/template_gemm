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
	exit(EXIT_FAILURE);
}

bool __error(const char* error, int line, std::string file) {
	printf("%s - Line: %d at %s\n", error, line, file);
	exit(EXIT_FAILURE);
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
	GEMM(char* trans_a, char* trans_b, size_t rows_a, size_t cols_a,
			size_t cols_b, const T alpha = 0, const T* host_ptr_a, int lda,
			const T* host_ptr_b, int ldb, T beta = 0, const T* host_ptr_c,
			int ldc) {

		this->rows_a = rows_a;
		this->cols_a = cols_a;
		this->rows_b = this->cols_a;
		this->cols_b = cols_b;
		this->cols_c = this->rows_a;
		this->rows_c = this->cols_b;

		if (cols != 0 && rows != 0) {
			check_framework_errors(
					cudaMalloc(this->device_ptr_a,
							this->rows_a * this->cols_a * sizeof(T)));
			check_framework_errors(
					cudaMalloc(this->device_ptr_b,
							this->rows_b * this->cols_b * sizeof(T)));
			check_framework_errors(
					cudaMalloc(this->device_ptr_c,
							this->rows_c * this->cols_c * sizeof(T)));

			//set 0 to C matrix
			check_framework_errors(
					cudaMemset(this->device_ptr_c, 0x00,
							this->rows_c * this->cols_c * sizeof(T)));

			//PUSH A
			check_framework_errors(
					cudaMemcpy(this->device_ptr_a, host_ptr_a,
							this->rows_a * this->cols_a * sizeof(T),
							cudaMemcpyHostToDevice));

			//PUSH B
			check_framework_errors(
					cudaMemcpy(this->device_ptr_b, host_ptr_b,
							this->rows_b * this->cols_b * sizeof(T),
							cudaMemcpyHostToDevice));

		} else {
			error("columns or rows equal to zero, or less than zero");
		}
	}

	virtual ~GEMM() {
		if (this->device_ptr_a != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_a));

		if (this->device_ptr_b != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_b));

		if (this->device_ptr_c != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_c));
	}



};

} /* namespace radiation */

#endif /* GEMM_H_ */
