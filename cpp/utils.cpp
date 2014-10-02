#include <cstdlib>
#include <iostream>
#include <math.h>
#include "mkl.h"
using namespace std;

double uniform(double min, double max) {
	return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

int binomial(int n, double p) {
	if (p < 0 || p > 1) return 0;

	int c = 0;
	double r;

	for (int i = 0; i<n; i++) {
		r = rand() / (RAND_MAX + 1.0);
		if (r < p) c++;
	}

	return c;
}

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}


/*
* mkl functions 
*/

//initialize array with random data
void init_arr(int N, int M, double* a)
{
	int i, j;
	for (i = 0; i< N; i++) {
		for (j = 0; j< M; j++) {
			a[i*M + j] = (i + j + 1) % 10; //keep all entries less than 10. pleasing to the eye!
		}
	}
}

//print array to std out
void print_arr(int N, int M,  double* array)
{
	int i, j;
	//printf("\n%s\n", name);
	for (i = 0; i<N; i++){
		for (j = 0; j<M; j++) {
			printf("%g ", *(array + i*M + j));
			//printf("%g ", array[i*M + j]);
		}
		printf("\n");
	}
	printf("\n");

}

//print array to std out
void copy_arrfrom1to2(int N, int M, double* a1, double* a2)
{
	int i, j;
	for (i = 0; i<N; i++){
		for (j = 0; j<M; j++) {
			a2[i*M + j] = *(a1 + i*M + j);
		}
	}
}

// multiply two matrix: (n * k) x (k * m) = (n * m)
void mkl_matrix_multiplication(int n, int k, int m, double* a, double*  b, double*  c, bool transA, bool transB, double bt)
{
	//MKL_INT64 AllocatedBytes;
	//int N_AllocatedBuffers;

	//copy
	double *aa = (double*)malloc(sizeof(double)*n*k);
	double *bb = (double*)malloc(sizeof(double)*k*m);
	double *cc = (double*)malloc(sizeof(double)*n*m);
	copy_arrfrom1to2(n, k, a, aa);
	copy_arrfrom1to2(k, m, b, bb);
	copy_arrfrom1to2(n, m, c, cc);

	//print_arr(n, k,a);
	//print_arr(m, k, b);
	//print_arr(N, M, "c", c);




	//print_arr(n, k, a);
	//print_arr(m, k, b);

	double alpha = 1.0, beta = bt;

	//cblas_dgemm(CblasRowMajor, 
	//	(transA) ? CblasTrans : CblasNoTrans,
	//	(transB) ? CblasTrans : CblasNoTrans,
	//	n, m, k, alpha,
	//	a, (transA) ? n : k, b, (transB) ? k : m,
	//	beta, c, m);
	//print_arr(n, m,  c);

	cblas_dgemm(CblasRowMajor,
		(transA) ? CblasTrans : CblasNoTrans,
		(transB) ? CblasTrans : CblasNoTrans,
		n, m, k, alpha,
		aa, (transA) ? n : k, bb, (transB) ? k : m,
		beta, cc, m);
	

	copy_arrfrom1to2(n, m, cc, c);
	free(aa);
	free(bb);
	free(cc);


	//AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
	//printf("\nDGEMM uses %ld bytes in %d buffers", (long)AllocatedBytes, N_AllocatedBuffers);

	//mkl_free_buffers();

	//AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
	//if (AllocatedBytes > 0) {
	//	printf("\nMKL memory leak!");
	//	printf("\nAfter mkl_free_buffers there are %ld bytes in %d buffers",
	//		(long)AllocatedBytes, N_AllocatedBuffers);
	//}




}
