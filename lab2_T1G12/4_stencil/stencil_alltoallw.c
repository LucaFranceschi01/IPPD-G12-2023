
/*
 * 2D stencil code using a blocking collective.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls a blocking collective operation with derived data types to
 * exchange a halo with neighbors.
 */

#include "mpi.h"
#include "stencil_par.h"
#include "perf_stat.h"

/* row-major order */
#define ind(i,j, bx) ((j)*(bx+2)+(i))

int ind_f(int i, int j, int bx)
{
	return ind(i, j, bx);
}

void setup(int rank, int proc, int argc, char **argv,
		   int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag);

void init_sources(int bx, int by, int offx, int offy, int n,
				  const int nsources, int sources[][2], int *locnsources_ptr, int locsources[][2]);

void alloc_bufs(int bx, int by, double **aold_ptr, double **anew_ptr);

void free_bufs(double *aold, double *anew);

void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr);

int main(int argc, char **argv)
{
	int rank, size;
	int n, energy, niters, px, py;

	int rx, ry; // QUESTION: P=processes / R=rank cartesian / B=blocksize / OFF=offset
	int north = -1, south = -1, west = -1, east = -1; // ranks of neighbours if any to NULL
	int bx, by, offx, offy;

	/* three heat sources */
	const int nsources = 3;
	int sources[nsources][2];
	int locnsources;            /* number of sources in my area */
	int locsources[nsources][2];        /* sources local to my rank */

	int *send_counts, *recv_counts;
	int *sdispls, *rdispls;
	MPI_Datatype typeNS, typeEW;

	int iter, i;

	double *aold, *anew, *tmp;

	double heat, rheat;

	int final_flag;


	/* initialize MPI envrionment */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* argument checking and setting */
	setup(rank, size, argc, argv, &n, &energy, &niters, &px, &py, &final_flag);

	if (final_flag == 1) {
		MPI_Finalize();
		exit(0);
	}

	/* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
	rx = rank % px;
	ry = rank / py; // Integer division truncates result

	/* determine my four neighbors */
	if ((rank - px) >= 0) north = rank - px;
	if ((rank + px) < (px*py)) south = rank + px;
	if ((rank - 1) / py == ry) west = rank - 1;
	if ((rank + 1) / py == ry) east = rank + 1;

	/* decompose the domain */
	bx = n / px;
	by = n / py;
	offx = rx * bx;
	offy = ry * by;

	printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank, ry, rx, west, east, north, south);

	/* initialize three heat sources */
	init_sources(bx, by, offx, offy, n, nsources, sources, &locnsources, locsources);

	/* allocate working arrays */
	alloc_bufs(bx, by, &aold, &anew);

	/* create north-south datatype */
	MPI_Type_contiguous(bx, MPI_DOUBLE, &typeNS);
	MPI_Type_commit(&typeNS);

	/* create east-west datatype */
	MPI_Type_vector(by, 1, bx+2, MPI_DOUBLE, &typeEW);
	MPI_Type_commit(&typeEW);

	/* prepare arguments of alltoallw */ // DISPLACEMENTS IN BYTES 
	recv_counts = (int*) malloc(size*sizeof(int));
	send_counts = (int*) malloc(size*sizeof(int));
	rdispls = (int*) malloc(size*sizeof(int));
	sdispls = (int*) malloc(size*sizeof(int));
	/*
	for (i = 0; i < size; i++) {
		if(i == north) {
			send_counts[i] = bx;
			sdispls[i] = ind_f(rx*bx, ry*by, bx) * sizeof(double); // (first col, first row) of block
			rdispls[i] = ind_f(rx*bx, ry*by-1, bx) * sizeof(double); // north halo (first row-1, first col)
		}
		else if(i == south) {
			send_counts[i] = bx;
			sdispls[i] = ind_f(rx*bx, ry*by+by, bx) * sizeof(double); // (first col, last row) of block
			rdispls[i] = ind_f(rx*bx, ry*by+by+1, bx) * sizeof(double); // south halo (first col, last row+1)
		}
		else if(i == west) {
			send_counts[i] = by;
			sdispls[i] = ind_f(rx*bx, ry*by, bx) * sizeof(double); // (first col, first row) of block
			rdispls[i] = ind_f(rx*bx-1, ry*by, bx) * sizeof(double); // west halo (first col-1, first row)
		}
		else if(i == east) {
			send_counts[i] = by;
			sdispls[i] = ind_f(rx*bx+bx, ry*by, bx) * sizeof(double); // (last col, first row) of block
			rdispls[i] = ind_f(rx*bx+bx+1, ry*by, bx) * sizeof(double); // east halo (last col+1, first row)
		}
		else {
			send_counts[i] = 0;
			sdispls[i] = 0;
			rdispls[i] = 0;
		}
		// printf("R%d---%i count=%d sdis=%d rdis=%d\n", rank, i, send_counts[i], sdispls[i], rdispls[i]);
	}
	*/

	for (i = 0; i < size; i++) {
		if(i == north) {
			send_counts[i] = 1; // SOLO ENVIAS UNO DE CADA TIPO CREO 
			// CREO QUE CADA PROCESO SOLO TIENE EN SU AOLD Y ANEW SU PARTE DEL ARRAY, ASI QUE ES COMO LOCAL NO SE SI ME EXPLICO
			sdispls[i] = ind_f(1, 1, bx) * sizeof(double); // (first col, first row) of block
			rdispls[i] = ind_f(1, 0, bx) * sizeof(double); // north halo (first col, first row-1) of block
		}
		else if(i == south) {
			send_counts[i] = 1;
			sdispls[i] = ind_f(1, by, bx) * sizeof(double); // (first col, last row) of block
			rdispls[i] = ind_f(1, by+1, bx) * sizeof(double); // south halo (first col, last row+1) of block
		}
		else if(i == west) {
			send_counts[i] = 1;
			sdispls[i] = ind_f(1, 1, bx) * sizeof(double); // (first col, first row) of block
			rdispls[i] = ind_f(0, 1, bx) * sizeof(double); // west halo (first col-1, first row) of block
		}
		else if(i == east) {
			send_counts[i] = 1;
			sdispls[i] = ind_f(bx, 1, bx) * sizeof(double); // (last col, first row) of block
			rdispls[i] = ind_f(bx+1, 1, bx) * sizeof(double); // east halo (last col+1, first row) of block
		}
		else {
			send_counts[i] = 0;
			sdispls[i] = 0;
			rdispls[i] = 0;
		}
		printf("R%d---%i count=%d sdis=%d rdis=%d\n", rank, i, send_counts[i], sdispls[i], rdispls[i]);
	}
	
	/* use different count parameters because some MPI implementations do not consider displacements
	 * in aliasing check */
	memcpy(recv_counts, send_counts, size * sizeof(int));

	for (iter = 0; iter < niters; ++iter) {

		/* refresh heat sources */
		PERF_COMP_BEGIN();
		for (i = 0; i < locnsources; ++i) {
			aold[ind_f(locsources[i][0], locsources[i][1], bx)] += energy;    /* heat source */
		}
		PERF_COMP_END();

		PERF_COMM_BEGIN();
		/* COMMUNICATION */ // MPI PROCESS NULL if neighbor is -1
		// NO SE SI EL ALLTOALL SE HACE ASI
		for(i=0; i<size; i++) {
			if(i == north || i == south) {
				MPI_Alltoallw(aold, send_counts, sdispls, &typeNS, anew, recv_counts, rdispls, &typeNS, MPI_COMM_WORLD);
			}
			if(i == east || i == west) {
				MPI_Alltoallw(aold, send_counts, sdispls, &typeEW, anew, recv_counts, rdispls, &typeEW, MPI_COMM_WORLD);
			}
		}

		PERF_COMM_END();

		/* update grid points */
		PERF_COMP_BEGIN();
		update_grid(bx, by, aold, anew, &heat);

		/* swap working arrays */
		tmp = anew;
		anew = aold;
		aold = tmp;

		/* optional - print image */
		if (iter == niters - 1)
			printarr_par(iter, anew, n, px, py, rx, ry, bx, by, offx, offy, ind_f, MPI_COMM_WORLD);
		PERF_COMP_END();
	}

	MPI_Type_free(&typeNS);
	MPI_Type_free(&typeEW);

	/* free working arrays and communication buffers */
	free_bufs(aold, anew);
	free(send_counts);
	free(recv_counts);
	free(rdispls);
	free(sdispls);

	/* get final heat in the system */

	rheat = heat; // NO LO SE

	if (!rank) {
		printf("[%i] last heat: %f \n", rank, rheat);
		PERF_PRINT();
	}

	MPI_Finalize();
	return 0;
}

void setup(int rank, int proc, int argc, char **argv,
		   int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag)
{
	int n, energy, niters, px, py;

	(*final_flag) = 0;

	if (argc < 6) {
		if (!rank)
			printf("usage: stencil_mpi <n> <energy> <niters> <px> <py>\n");
		(*final_flag) = 1;
		return;
	}

	n = atoi(argv[1]);  /* nxn grid */
	energy = atoi(argv[2]);     /* energy to be injected per iteration */
	niters = atoi(argv[3]);     /* number of iterations */
	px = atoi(argv[4]); /* 1st dim processes */
	py = atoi(argv[5]); /* 2nd dim processes */

	if (px * py != proc) {
		fprintf(stderr, "px * py must equal to the number of processes.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);   /* abort if px or py are wrong */
	}
	if (n % px != 0) {
		fprintf(stderr, "grid size n must be divisible by px.\n");
		MPI_Abort(MPI_COMM_WORLD, 2);   /* abort px needs to divide n */
	}
	if (n % py != 0) {
		fprintf(stderr, "grid size n must be divisible by py.\n");
		MPI_Abort(MPI_COMM_WORLD, 3);   /* abort py needs to divide n */
	}

	(*n_ptr) = n;
	(*energy_ptr) = energy;
	(*niters_ptr) = niters;
	(*px_ptr) = px;
	(*py_ptr) = py;
}

void init_sources(int bx, int by, int offx, int offy, int n,
				  const int nsources, int sources[][2], int *locnsources_ptr, int locsources[][2])
{
	int i, locnsources = 0;

	sources[0][0] = n / 2;
	sources[0][1] = n / 2;
	sources[1][0] = n / 3;
	sources[1][1] = n / 3;
	sources[2][0] = n * 4 / 5;
	sources[2][1] = n * 8 / 9;

	for (i = 0; i < nsources; ++i) {    /* determine which sources are in my patch */
		int locx = sources[i][0] - offx;
		int locy = sources[i][1] - offy;
		if (locx >= 0 && locx < bx && locy >= 0 && locy < by) {
			locsources[locnsources][0] = locx + 1;      /* offset by halo zone */
			locsources[locnsources][1] = locy + 1;      /* offset by halo zone */
			locnsources++;
		}
	}

	(*locnsources_ptr) = locnsources;
}

void alloc_bufs(int bx, int by, double **aold_ptr, double **anew_ptr)
{
	double *aold, *anew;

	/* allocate two working arrays */
	anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
	aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */

	memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
	memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

	(*aold_ptr) = aold;
	(*anew_ptr) = anew;
}

void free_bufs(double *aold, double *anew)
{
	free(aold);
	free(anew);
}

void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr)
{
	int i, j;
	double heat = 0.0;

	for (i = 1; i < bx + 1; ++i) {
		for (j = 1; j < by + 1; ++j) {
			anew[ind_f(i, j, bx)] =
				anew[ind_f(i, j, bx)] / 2.0 + (aold[ind_f(i - 1, j, bx)] + aold[ind_f(i + 1, j, bx)] +
										 aold[ind_f(i, j - 1, bx)] + aold[ind_f(i, j + 1, bx)]) / 4.0 / 2.0;
			heat += anew[ind_f(i, j, bx)];
		}
	}

	(*heat_ptr) = heat;
}
