#include "GEMM.cuh"
#include "GEMMWMMA.cuh"

#include <iostream>
#include <cuda_fp16.h>

template<class T> void print(T* c, size_t n) {
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			std::cout << __half2float(c[i * n + j]) << " ";
		}
		std::cout << std::endl;
	}
}

template<class T> void matrix_mul(size_t siz, size_t n, bool wmma = false) {
	T* a = (T*) (calloc(siz, sizeof(T)));
	T* b = (T*) (calloc(siz, sizeof(T)));
	T* c = (T*) (calloc(siz, sizeof(T)));

	for (size_t i = 0; i < siz; i++)
		a[i] = b[i] = 1.0;

	if (wmma){
		radiation::GEMMWMMA<T> test1(a, b, c, n, n, n);
		test1.mul();
		test1.pull_array(c);
		print(c, n);
	}else{
		radiation::GEMM<T> test(a, b, c, n, n, n);
		test.mul();
		test.pull_array(c);
		print(c, n);
	}

	free(a);
	free(b);
	free(c);
}

int main(int argc, char** argv) {
	if (argc == 2) {
		size_t n = atoi(argv[1]);
		size_t siz = n * n;

//		std::cout << "Multiplying for double" << std::endl;
//		matrix_mul<double>(siz, n);
//
//		std::cout << "Multiplying for float" << std::endl;
//		matrix_mul<float>(siz, n);
//
//		std::cout << "Multiplying for half" << std::endl;
//		matrix_mul<half>(siz, n);



//		std::cout << "Multiplying for single for WMMA" << std::endl;
//		matrix_mul<float>(siz, n, true);

		std::cout << "Multiplying for half for WMMA" << std::endl;
		matrix_mul<half>(siz, n, true);



	}

	return 0;
}
