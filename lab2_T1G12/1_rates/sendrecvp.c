#include <stdio.h>
#include <mpi.h>

int main (int argc, char **argv) {

    int rank, nprocs;

    // MPI Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) printf("n\ttime (sec)\tRate (MB/sec)\n");

    // Number of doubles to send
    int n = 1, received;
    MPI_Status status;

	double wtime;
	MPI_Request request = MPI_REQUEST_NULL;
    while(n < 10000000) {
        double buff[n];
		// No need for initialization
		
        if(rank == 0) {
			if(n < 1000) {
				double local_wtime = 0;
				for(int i=0; i<1000/n; i++) {
					local_wtime = MPI_Wtime();
					MPI_Isend(&buff, n, MPI_DOUBLE, 1, 1111, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Irecv(&buff, n, MPI_DOUBLE, 1, 1111, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					wtime += MPI_Wtime() - local_wtime;
				}
				wtime = wtime * n / 1000;
			} else {
				wtime = MPI_Wtime();
				MPI_Isend(&buff, n, MPI_DOUBLE, 1, 1111, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				MPI_Irecv(&buff, n, MPI_DOUBLE, 1, 1111, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				wtime = MPI_Wtime() - wtime;
			}

			printf("%d\t%f\t%f\n", n, wtime, sizeof(buff)/(1000000*wtime));
        }

        if(rank == 1) {
            if(n < 1000) {
				double local_wtime = 0;
				for(int i=0; i<1000/n; i++) {
					local_wtime = MPI_Wtime();
					MPI_Isend(&buff, n, MPI_DOUBLE, 0, 1111, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					MPI_Irecv(&buff, n, MPI_DOUBLE, 0, 1111, MPI_COMM_WORLD, &request);
					MPI_Wait(&request, &status);
					wtime += MPI_Wtime() - local_wtime;
				}
				wtime = wtime * n / 1000;
			} else {
				wtime = MPI_Wtime();
				MPI_Isend(&buff, n, MPI_DOUBLE, 0, 1111, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				MPI_Irecv(&buff, n, MPI_DOUBLE, 0, 1111, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				wtime = MPI_Wtime() - wtime;
			}
        }

		n = n*2;
    }

    MPI_Finalize();
    return 0;
}