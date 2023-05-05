
#ifndef PERF_STAT_H
#define PERF_STAT_H

static double t_comp_total = 0.0;
static double t_comm_total = 0.0;

#ifdef MPI_INCLUDED

static double t_comp_start = 0.0;
static double t_comp_stop = 0.0;

static double t_comm_start = 0.0;
static double t_comm_stop = 0.0;

#define PERF_COMP_BEGIN() \
    do { \
        t_comp_start = MPI_Wtime(); \
    } while (0)

#define PERF_COMP_END() \
    do { \
        t_comp_stop = MPI_Wtime(); \
        t_comp_total += (t_comp_stop - t_comp_start); \
    } while (0)

#define PERF_COMM_BEGIN() \
    do { \
        t_comm_start = MPI_Wtime(); \
    } while (0)

#define PERF_COMM_END() \
    do { \
        t_comm_stop = MPI_Wtime(); \
        t_comm_total += (t_comm_stop - t_comm_start); \
    } while (0)

#else

#include <time.h>

static struct timespec t_comp_start, t_comp_end;
static struct timespec t_comm_start, t_comm_end;

#define TIME_STRUCT_DIFF(end, start) \
    ((double)((end).tv_sec - (start).tv_sec) \
     + 1.0e-9 * (double)((end).tv_nsec - (start).tv_nsec))

#define PERF_COMP_BEGIN() \
    do { \
        clock_gettime(CLOCK_REALTIME, &t_comp_start); \
    } while (0)

#define PERF_COMP_END() \
    do { \
        clock_gettime(CLOCK_REALTIME, &t_comp_end); \
        t_comp_total += TIME_STRUCT_DIFF(t_comp_end, t_comp_start); \
    } while (0)

#define PERF_COMM_BEGIN() \
    do { \
        clock_gettime(CLOCK_REALTIME, &t_comm_start); \
    } while (0)

#define PERF_COMM_END() \
    do { \
        clock_gettime(CLOCK_REALTIME, &t_comm_end); \
        t_comm_total += TIME_STRUCT_DIFF(t_comm_end, t_comm_start); \
    } while (0)

#endif

#define PERF_PRINT() \
    do { \
        printf("Total computation time: %f s\nTotal communication time: %f s\n", t_comp_total, t_comm_total); \
    } while (0)

#endif /* PERF_STAT_H */
