#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "cholesky.h"

void cholesky_openmp(int n) {
	int i, j, k;
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
    for(i=0; i < n; i++) {
        A[i][i] = rand()%1000+100.0;
        for(j=i+1; j < n; j++) {
            A[i][j] = rand()%100 + 1.0;
            A[j][i] = A[i][j];
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
	#pragma omp parallel for private(i, j, tmp)
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
	int strip_size = 8; // L1 cache size is 64 bytes (lscpu 32+32), 8 doubles
	#pragma omp parallel for collapse(2) private(i, j)
	for(i=0; i<n; i+=strip_size) {
		for(j=0; j<n; j+=strip_size) {
			for(k=i; k<(i+strip_size) && k<n; k++) {
				for(int l=j; l<(j+strip_size) && l<n; l++) {
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
	#pragma omp parallel for 
    for(i=0; i<n; i++) {
    	for(k=0; k<n; k++) { // swapped two inner loops (works)
			B[i][j] = 0.0;
			for(j=0; j<n; j++) {
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
	#pragma omp parallel for collapse(2)
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) {
			if (abs(B[i][j] - A[i][j]) / A[i][j] > 0.001) {
				#pragma omp atomic
				cnt++;
			}
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

void cholesky(int n) {
    int i, j, k;
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
    for(i=0; i < n; i++) {
        A[i][i] = rand()%1000+100.0;
        for(j=i+1; j < n; j++) {
            A[i][j] = rand()%100 + 1.0;
            A[j][i] = A[i][j];
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
	int strip_size = 8; // L1 cache size is 64 bytes, 8 doubles
	for(i=0; i<n; i+=strip_size) {
		for(j=0; j<n; j+=strip_size) {
			for(k=i; k< i+strip_size && k<n; k++) {
				for(int l=j; l < j+strip_size && l<n; l++) {
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
    	for(j=0; j<n; j++) {
			B[i][j] = 0.0;
			for(k=0; k<n; k++) {
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