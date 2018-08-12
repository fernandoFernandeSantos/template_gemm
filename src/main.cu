
#include "GEMM.cuh"

int main() {
	size_t n = 2048;
	size_t siz = n * n;
	double* a = (double*)calloc(siz, sizeof(double));
	double* b = (double*)calloc(siz, sizeof(double));
	double* c = (double*)calloc(siz, sizeof(double));

	for (auto i = 0; i < siz; i++)
		a[i] = b[i] = 1.0;
//	(char* trans_a, char* trans_b, size_t rows_a, size_t cols_a,
//				size_t cols_b, const T alpha, const T* host_ptr_a, int lda,
//				const T* host_ptr_b, int ldb, const T beta, const T* host_ptr_c,
//				int ldc, size_t block_size = 32)

	radiation::GEMM<double> test(a, b, c, n, n, n);
	test.mul();
	test.pull_array(c);

//	for(auto i = 0; i < n; i++){
//		for(auto j = 0; j < n; j++){
//			printf("%lf ", c[i * n + j]);
//		}
//		printf("\n");
//	}

	free(a);
	free(b);
	free(c);

	return 0;
}
