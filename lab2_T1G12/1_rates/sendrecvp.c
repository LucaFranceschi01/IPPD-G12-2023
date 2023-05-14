#include <stdio.h>
#include <mpi.h>

#define MIN(a,b) (((a)<(b))?(a):(b)) // define min function (used for outliers)
#define OUTLIER_ITER 10 // we get good results with 10 iterations

int main (int argc, char **argv) {

    int rank, nprocs;

	if(argc > 1) return 1;

    // MPI Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) printf("n\ttime (sec)\tRate (MB/sec)\n");

    // Number of doubles to send
    int n = 1, received;
    MPI_Status status;

	double wtime, local_wtime;
	MPI_Request request = MPI_REQUEST_NULL;
    while(n < 10000000) {
        double buf[n];
		// No need for initialization

        if(rank == 0) {
			if(n < 1000) {
				wtime = 0;
				for(int i=0; i<1000/n; i++) {
					local_wtime = MPI_Wtime();
					MPI_Isend(&buf, n, MPI_DOUBLE, 1, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Irecv(&buf, n, MPI_DOUBLE, 1, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					wtime += MPI_Wtime() - local_wtime; // add up partial values of wait time
				}
				wtime = wtime * n / 1000.f; // average waiting time
			} else {
				wtime = __DBL_MAX__; // init to max double value
				for(int i=0; i<OUTLIER_ITER; i++) { // iterate a couple of times to get rid of outliers
					local_wtime = MPI_Wtime();
					MPI_Isend(&buf, n, MPI_DOUBLE, 1, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Irecv(&buf, n, MPI_DOUBLE, 1, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					wtime = MIN(wtime, MPI_Wtime() - local_wtime);
				}
			}
			printf("%d\t%f\t%f\n", n, wtime, sizeof(buf)/(1048576 * wtime));
        } 
		if(rank == 1) {
            if(n < 1000) {
				wtime = 0;
				for(int i=0; i<1000/n; i++) {
					local_wtime = MPI_Wtime();
					MPI_Irecv(&buf, n, MPI_DOUBLE, 0, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Isend(&buf, n, MPI_DOUBLE, 0, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					wtime += MPI_Wtime() - local_wtime;
				}
				wtime = wtime * n / 1000.f;
			} else {
				wtime = __DBL_MAX__;
				for(int i=0; i<OUTLIER_ITER; i++) {
					local_wtime = MPI_Wtime();
					MPI_Irecv(&buf, n, MPI_DOUBLE, 0, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Isend(&buf, n, MPI_DOUBLE, 0, 1234, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					wtime = MIN(wtime, MPI_Wtime() - local_wtime);
				}
			}
        }
		n = n*2;
    }

    MPI_Finalize();
    return 0;
}