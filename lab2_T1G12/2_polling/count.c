/* POLLING */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Definitions */
#define MAX_QUEST 5
#define FILENAME "./votations.dat"

typedef struct {
	int idTable;
	int idQuestion;
	int yes;
	int no;
}  tRecord;

/* Main program */
int main (int argc, char **argv)
{
	int rank, nprocs;
	int recordsize, filenumrecords, rem, numrecords;
	int i, quest;
	tRecord *buf;
	long total, yes[MAX_QUEST], totYes[MAX_QUEST];
	long no[MAX_QUEST], totNo[MAX_QUEST];

	MPI_Status   info;
	MPI_Datatype rectype;
	MPI_File     infile;
	MPI_Offset   filesize, poffset;

	/* MPI Initialization */
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

	/* Create datatype */
	int count = 4;
	int blocklens[] = {1, 1, 1, 1};
	MPI_Aint lowerbound, extent;
	MPI_Type_get_extent(MPI_INT, &lowerbound, &extent);
	MPI_Aint offsets[] = {0, extent, 2*extent, 3*extent};
	MPI_Datatype oldtypes[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
	MPI_Type_create_struct(count, blocklens, offsets, oldtypes, &rectype);
	MPI_Type_commit(&rectype);

	/* Each process reads a part of the file */
	MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);

	/* Each process determines number of records to read and initial offset */
	MPI_File_get_size(infile, &filesize);
	filenumrecords = filesize / sizeof(tRecord);
	numrecords = filenumrecords / nprocs; // hope that it is divisible
	MPI_File_seek(infile, numrecords*rank*sizeof(tRecord), MPI_SEEK_SET);

	/* Allocate buffer for records */
	buf = (tRecord*) malloc(sizeof(tRecord)*numrecords);

	/* Process reads numrecords consecutive elements */
	MPI_File_read(infile, buf, numrecords, rectype, &info);
	MPI_File_close(&infile);

	/* Count results by each process */
	for(i=0; i<MAX_QUEST; i++) {
		yes[i] = 0;
		no[i] = 0;
		totYes[i] = 0;
		totNo[i] = 0;
	}

	i=0;
	while(i<numrecords) {
		quest = buf[i].idQuestion;
		yes[quest] += buf[i].yes;
		no[quest] += buf[i].no;
		i++;
	}

	total = 0;
	for(i=0; i<MAX_QUEST; i++) total += yes[i] + no[i];

	/* Print local results */
	printf ("Proc %3d. Counted votes = %d\n", rank, total);

	/* Print global results on process 0 */

	if (rank == 0)
	{
		total = 0;
		printf ("------------------------------------------------------------\n");
		for (i=0; i<MAX_QUEST; i++) 
		{
			total += totYes[i] + totNo[i];
			printf("Question %d: yes: %.1f%% (%d) no: %.1f%% (%d)\n", i, totYes[i]*100.0/total, totYes[i], totNo[i]*100.0/total, totNo[i]);
			fflush (stdout);
		}

		printf ("------------------------------------------------------------\n");
		printf ("Total votes = %d\n", total);
		fflush (stdout);
	}

	/* Free datatype */
	MPI_Type_free (&rectype);

	/* End MPI */
	MPI_Finalize();
	return 0;
}

