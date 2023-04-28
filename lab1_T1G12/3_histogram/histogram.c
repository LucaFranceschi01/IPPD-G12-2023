
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "random.h"

// uncomment this #define if you want diagnostic output
//#define     DEBUG         0

#define     num_trials    1000000 // number of x values
#define     num_buckets   50         // number of buckets in hitogram
static long xlow        = 0.0;      // low end of x range
static long xhi         = 100.0;    // High end of x range

int main (){

    double x[num_trials];     // array used to assign counters in the historgram
    long   hist[num_buckets]; // the histogram
    double bucket_width;      // the width of each bucket in the histogram
    double time;


    seed(xlow, xhi);  // seed the random number generator over range of x
    bucket_width = (xhi-xlow)/(double)num_buckets;

    // fill the array
    for(int i=0;i<num_trials;i++)
        x[i] = drandom();

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- sequential
    ////////////////////////////////////////////////////////////////

    // initialize the histogram -> this can be turned into a function
    for(int i=0;i<num_buckets;i++)
        hist[i] = 0;

    // Assign x values to the right historgram bucket
    time = omp_get_wtime();
    for(int i=0;i<num_trials;i++){

        long ival = (long) (x[i] - xlow)/bucket_width;

        hist[ival]++;

        #ifdef DEBUG
        printf("i = %d,  xi = %f, ival = %d\n",i,(float)x[i], ival);
        #endif

    }

    time = omp_get_wtime() - time;

    // compute statistics ... ave, std-dev for whole histogram and quartiles
    // -> this can be turned into a function
    double sumh=0.0, sumhsq=0.0, ave, std_dev;
    for(int i=0;i<num_buckets;i++){
        sumh   += (double) hist[i];
        sumhsq += (double) hist[i]*hist[i];
    }

    ave     = sumh/num_buckets;
    std_dev = sqrt(sumhsq - sumh*sumh/(double)num_buckets);


    printf("Sequential histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf("ave = %f, std_dev = %f\n",(float)ave, (float)std_dev);
    printf("in %f seconds\n",(float)time);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- critical
    ////////////////////////////////////////////////////////////////

	// initialize the histogram -> this can be turned into a function
    for(int i=0;i<num_buckets;i++)
        hist[i] = 0;

    // Assign x values to the right historgram bucket
    time = omp_get_wtime();
	long ival;
	#pragma omp parallel for private(ival)
    for(int i=0;i<num_trials;i++){

        ival = (long) (x[i] - xlow)/bucket_width;
		#pragma omp critical
        hist[ival]++;

        #ifdef DEBUG
        printf("i = %d,  xi = %f, ival = %d\n",i,(float)x[i], ival);
        #endif

    }

    time = omp_get_wtime() - time;

    // compute statistics ... ave, std-dev for whole histogram and quartiles
    // -> this can be turned into a function
    sumh=0.0, sumhsq=0.0;
    for(int i=0;i<num_buckets;i++){
        sumh   += (double) hist[i];
        sumhsq += (double) hist[i]*hist[i];
    }

    ave     = sumh/num_buckets;
    std_dev = sqrt(sumhsq - sumh*sumh/(double)num_buckets);


    printf("Par with critical histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf("ave = %f, std_dev = %f\n",(float)ave, (float)std_dev);
    printf("in %f seconds\n",(float)time);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- locks
    ////////////////////////////////////////////////////////////////

	omp_lock_t lock;
	omp_init_lock(&lock);

	for(int i=0;i<num_buckets;i++)
        hist[i] = 0;

    // Assign x values to the right historgram bucket
    time = omp_get_wtime();

	#pragma omp parallel for private(ival)
    for(int i=0;i<num_trials;i++){

        ival = (long) (x[i] - xlow)/bucket_width;
		omp_set_lock(&lock);
        hist[ival]++;
		omp_unset_lock(&lock);

        #ifdef DEBUG
        printf("i = %d,  xi = %f, ival = %d\n",i,(float)x[i], ival);
        #endif

    }

	omp_destroy_lock(&lock);

    time = omp_get_wtime() - time;

    // compute statistics ... ave, std-dev for whole histogram and quartiles
    // -> this can be turned into a function
    sumh=0.0, sumhsq=0.0;
    for(int i=0;i<num_buckets;i++){
        sumh   += (double) hist[i];
        sumhsq += (double) hist[i]*hist[i];
    }

    ave     = sumh/num_buckets;
    std_dev = sqrt(sumhsq - sumh*sumh/(double)num_buckets);


    printf("Par with critical histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf("ave = %f, std_dev = %f\n",(float)ave, (float)std_dev);
    printf("in %f seconds\n",(float)time);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- reduction
    ////////////////////////////////////////////////////////////////

	for(int i=0;i<num_buckets;i++)
        hist[i] = 0;

    // Assign x values to the right historgram bucket
    time = omp_get_wtime();

	#pragma omp parallel for private(ival) reduction(+:hist)
    for(int i=0;i<num_trials;i++){

        ival = (long) (x[i] - xlow)/bucket_width;
        hist[ival]++;

        #ifdef DEBUG
        printf("i = %d,  xi = %f, ival = %d\n",i,(float)x[i], ival);
        #endif

    }

    time = omp_get_wtime() - time;

    // compute statistics ... ave, std-dev for whole histogram and quartiles
    // -> this can be turned into a function
    sumh=0.0, sumhsq=0.0;
    for(int i=0;i<num_buckets;i++){
        sumh   += (double) hist[i];
        sumhsq += (double) hist[i]*hist[i];
    }

    ave     = sumh/num_buckets;
    std_dev = sqrt(sumhsq - sumh*sumh/(double)num_buckets);


    printf("Par with critical histogram for %d buckets of %d values\n",num_buckets, num_trials);
    printf("ave = %f, std_dev = %f\n",(float)ave, (float)std_dev);
    printf("in %f seconds\n",(float)time);

    return 0;
}
	  
