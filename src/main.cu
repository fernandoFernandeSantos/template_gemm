

#include "GEMM.cuh"



int main(){
	size_t n = 100;
	double a[n], b[n], c[n];
	for(auto i = 0; i < n; i++)
		a[i] = b[i] = c[i] = 1.0;
//	(char* trans_a, char* trans_b, size_t rows_a, size_t cols_a,
//				size_t cols_b, const T alpha, const T* host_ptr_a, int lda,
//				const T* host_ptr_b, int ldb, const T beta, const T* host_ptr_c,
//				int ldc, size_t block_size = 32)

	radiation::GEMM<double> test(a, b, c, n, n, n);
	test.mul();

	return 0;
}
