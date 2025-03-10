#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "cholesky.h"

void cholesky_openmp(int n) {
	int i, j, k, l;
	double** A;
	double** L;
	double** U;
	double** B;
	double tmp;
	double start, end;
	int cnt;
	
	/**
	 * 1. Matrix initialization for A, L, U and B
	 */
	start = omp_get_wtime();
	A = (double **)malloc(n * sizeof(double *)); 
	L = (double **)malloc(n * sizeof(double *)); 
	U = (double **)malloc(n * sizeof(double *)); 
	B = (double **)malloc(n * sizeof(double *)); 
	
	for(i=0; i<n; i++) {
		 A[i] = (double *)malloc(n * sizeof(double)); 
		 L[i] = (double *)malloc(n * sizeof(double)); 
		 U[i] = (double *)malloc(n * sizeof(double)); 
		 B[i] = (double *)malloc(n * sizeof(double)); 
	}
	
	srand(time(NULL));
	// Generate random values for the matrix
	for(i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			A[i][j] = ((double)rand() / RAND_MAX)*2.0 - 1.0;  // Generate values between -1 and 1
		}
	}

	// Make the matrix positive definite
	for(i=0; i<n; i++) {
		for (j=i; j<n; j++) {
			if (i==j) {
				A[i][j] += n;
			} else {
				A[i][j] += ((double)rand() / RAND_MAX)*sqrt(n);
				A[j][i] = A[i][j];
			}
		}
	}

	for(i=0; i < n; i++) {
		for(j=0; j < n; j++) {
			L[i][j] = 0.0;
			U[i][j] = 0.0;
		}
	}
	end = omp_get_wtime();
	printf("Initialization: %f\n", end-start);
	
	/**
	 * 2. Compute Cholesky factorization for U
	 */
	start = omp_get_wtime();
	for(i=0; i<n; i++) {
		// Calculate diagonal elements
		tmp = 0.0;
		#pragma omp parallel for schedule(dynamic) reduction(+:tmp)
		for(k=0;k<i;k++) {
			tmp += U[k][i]*U[k][i];
		}
		U[i][i] = sqrt(A[i][i]-tmp);
		// Calculate non-diagonal elements
		#pragma omp parallel for schedule(dynamic)
		for(j=i+1;j<n;j++) {
			tmp = 0.0;
			for(k=0; k<i; k++) {
				tmp += U[k][j]*U[k][i];	
			}
			U[i][j] = (A[j][i] - tmp) / U[i][i];
		}
	}
	end = omp_get_wtime();
	printf("Cholesky: %f\n", end-start);
		
	/**
	 * 3. Calculate L from U'
	 */
	start = omp_get_wtime();
	// TODO L=U'
	// Use stripmining to exploit spatial locality
	#pragma omp parallel for private(i, j, k, l) shared(L, U) schedule(dynamic, 4)
	for(i=0; i<n; i+=STRIPSIZE) {
		for(j=i; j<n; j+=STRIPSIZE) { // Starting from i and not 0 skips zero elements
			for(k=i; k< i+STRIPSIZE && k<n; k++) {
				for(l=j; l < j+STRIPSIZE && l<n; l++) {
					L[l][k] = U[k][l];
				}	
			}
		}	
	}
	end = omp_get_wtime();
	printf("L=U': %f\n", end-start);
	
	/**
	 * 4. Compute B=LU
	 */
	start = omp_get_wtime();
	// TODO B=LU
	#pragma omp parallel for private(i, j, k) shared(B, L, U) schedule(guided) // Not sure about the guided scheduling, but it was the one with best results in the testing and the difference was not that impressive
	for(i=0; i<n; i++) {
		for(k=0; k<=i; k++) { // Swapping the inner loop provides spatial locality --> speed
			for(j=k; j<n; j++) { // The boundaries used in the inner loops make it so that zero elements are skipped
				B[i][j] += L[i][k] * U[k][j];
			}
		}
	}
	// Other options that were not the most efficient
	// Worked fast taking advantage of spatial locality but does not skip zeros
	/*
	for(i=0; i<n; i++) {
		for(k=0; k<=i; k++) {
			for(j=0; j<n; j++) {
				B[i][j] += L[i][k] * U[k][j];
			}
		}
	}
	*/
	// Works nice but the condition is weird and not fast
	/*
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) {
			for(k=0; k<=min(i, j); k++) {
				B[i][j] += L[i][k] * U[k][j];
			}
		}
	}
	*/
	end = omp_get_wtime();
	printf("B=LU: %f\n", end-start);

	/**
	 * 5. Check if all elements of A and B have a difference smaller than 0.001%
	 */
	start = omp_get_wtime();
	cnt=0;
	// TODO check if matrices are equal
	#pragma omp parallel for collapse(2) private(i, j) shared(B, A) reduction(+:cnt)
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) {
			if (abs(B[i][j] - A[i][j]) / A[i][j] > 0.001) cnt++;
		}
	}
	// Not needed and not that fast when implemented tbh
	/*
	for(i=0; i<n; i+=STRIPSIZE) {
		for(j=0; j<n; j+=STRIPSIZE) {
			for(k=i; k<i+STRIPSIZE && k<n; k++) {
				for(l=j; l<k+STRIPSIZE && l<n; l++) {
					if (abs(B[k][l] - A[k][l]) / A[k][l] > 0.001) cnt++;
				}	
			}
		}
	}
	*/
	if(cnt != 0) {
		printf("Matrices are not equal\n");
	} else {
		printf("Matrices are equal\n");
	}
	end = omp_get_wtime();
	printf("A==B?: %f\n", end-start);
}

void cholesky(int n) {
	int i, j, k, l;
	double** A;
	double** L;
	double** U;
	double** B;
	double tmp;
	double start, end;
	int cnt;
	
	/**
	 * 1. Matrix initialization for A, L, U and B
	 */
	start = omp_get_wtime();
	A = (double **)malloc(n * sizeof(double *)); 
	L = (double **)malloc(n * sizeof(double *)); 
	U = (double **)malloc(n * sizeof(double *)); 
	B = (double **)malloc(n * sizeof(double *)); 
	
	for(i=0; i<n; i++) {
		 A[i] = (double *)malloc(n * sizeof(double)); 
		 L[i] = (double *)malloc(n * sizeof(double)); 
		 U[i] = (double *)malloc(n * sizeof(double)); 
		 B[i] = (double *)malloc(n * sizeof(double)); 
	}
	
	srand(time(NULL));
	// Generate random values for the matrix
	for(i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			A[i][j] = ((double)rand() / RAND_MAX)*2.0 - 1.0;  // Generate values between -1 and 1
		}
	}

	// Make the matrix positive definite
	for(i=0; i<n; i++) {
		for (j=i; j<n; j++) {
			if (i==j) {
				A[i][j] += n;
			} else {
				A[i][j] += ((double)rand() / RAND_MAX)*sqrt(n);
				A[j][i] = A[i][j];
			}
		}
	}

	for(i=0; i < n; i++) {
		for(j=0; j < n; j++) {
			L[i][j] = 0.0;
			U[i][j] = 0.0;
		}
	}
	end = omp_get_wtime();
	printf("Initialization: %f\n", end-start);
	
	/**
	 * 2. Compute Cholesky factorization for U
	 */
	start = omp_get_wtime();
	for(i=0; i<n; i++) {
		// Calculate diagonal elements
		tmp = 0.0;
		for(k=0;k<i;k++) {
			tmp += U[k][i]*U[k][i];
		}
		U[i][i] = sqrt(A[i][i]-tmp);
		// Calculate non-diagonal elements
		for(j=i+1;j<n;j++) {
			tmp = 0.0;
			for(k=0; k<i; k++) {
				tmp += U[k][j]*U[k][i];	
			}	
			U[i][j] = (A[j][i] - tmp) / U[i][i];
		}
	}
	end = omp_get_wtime();
	printf("Cholesky: %f\n", end-start);
		
	/**
	 * 3. Calculate L from U'
	 */
	start = omp_get_wtime();
	// TODO L=U'
	for(i=0; i<n; i+=STRIPSIZE) {
		for(j=i; j<n; j+=STRIPSIZE) {
			for(k=i; k< i+STRIPSIZE && k<n; k++) {
				for(l=j; l < j+STRIPSIZE && l<n; l++) {
					L[l][k] = U[k][l];
				}	
			}
		}	
	}
	end = omp_get_wtime();
	printf("L=U': %f\n", end-start);
	
	/**
	 * 4. Compute B=LU
	 */
	start = omp_get_wtime();
	// TODO B=LU
	for(i=0; i<n; i++) {
		for(k=0; k<=i; k++) {
			for(j=k; j<n; j++) {
				B[i][j] += L[i][k] * U[k][j];
			}
		}
	}
	
	end = omp_get_wtime();
	printf("B=LU: %f\n", end-start);

	/**
	 * 5. Check if all elements of A and B have a difference smaller than 0.001%
	 */
	start = omp_get_wtime();
	cnt=0;
	// TODO check if matrices are equal
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) {
			if (abs(B[i][j] - A[i][j]) / A[i][j] > 0.001) cnt++;
		}
	}
	
	if(cnt != 0) {
		printf("Matrices are not equal\n");
	} else {
		printf("Matrices are equal\n");
	}
	end = omp_get_wtime();
	printf("A==B?: %f\n", end-start);
}