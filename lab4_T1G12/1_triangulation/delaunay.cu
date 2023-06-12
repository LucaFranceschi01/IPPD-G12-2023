#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define pixel(i, j, w)  (((j)*(w)) +(i))
#define TPB 128 // THREADS PER BLOCK needs testing

int max_num_triangles;

/* A point in 2D space */
struct Point {
    double x;
    double y;
    double value;
};

/* A triangle defined by three points */
struct Triangle {
    struct Point p1;
    struct Point p2;
    struct Point p3;
};

/* Helper function to output the triangles in the Delaunay Triangulation */
void print_triangles(struct Triangle * triangles, int num_triangles) {
    for (int i = 0; i < num_triangles; i++) {
        printf("(%lf, %lf) (%lf, %lf) (%lf, %lf)\n", 
            triangles[i].p1.x, triangles[i].p1.y,
            triangles[i].p2.x, triangles[i].p2.y,
            triangles[i].p3.x, triangles[i].p3.y);   
    }
}

/* Helper function to calculate the distance between two points */
__device__ double distance(struct Point * p1, struct Point * p2) {
    double dx = (*p1).x - (*p2).x;
    double dy = (*p1).y - (*p2).y;
    return sqrt(dx*dx + dy*dy);
}

/* Helper function to check if a triangle is clockwise */
__device__ int is_ccw(struct Triangle * t) {
    double ax = (*t).p2.x - (*t).p1.x;
    double ay = (*t).p2.y - (*t).p1.y;
    double bx = (*t).p3.x - (*t).p1.x;
    double by = (*t).p3.y - (*t).p1.y;

    double area = ax * by - ay * bx;
    return area > 0;
}

/* Helper function to check if a point is inside a circle defined by three points */
__device__ int inside_circle(struct Point * p, struct Triangle * t) {
//      | ax-dx, ay-dy, (ax-dx)² + (ay-dy)² |
//det = | bx-dx, by-dy, (bx-dx)² + (by-dy)² |
//      | cx-dx, cy-dy, (cx-dx)² + (cy-dy)² |

    int clockwise = is_ccw(t);
    
    double ax = (*t).p1.x - (*p).x;
    double ay = (*t).p1.y - (*p).y;
    double bx = (*t).p2.x - (*p).x;
    double by = (*t).p2.y - (*p).y;
    double cx = (*t).p3.x - (*p).x;
    double cy = (*t).p3.y - (*p).y;

    double det = ax*by + bx*cy + cx*ay - ay*bx - by*cx - cy*ax;
    det = (ax*ax + ay*ay) * (bx*cy-cx*by) -
            (bx*bx + by*by) * (ax*cy-cx*ay) +
            (cx*cx + cy*cy) * (ax*by-bx*ay);
    
    if(clockwise) {
        return det > 0;
	}
    return det < 0;
}

//* Helper function to compute barycentric coordintaes of a point respect a triangle */
__device__ void barycentric_coordinates(struct Triangle * t, struct Point * p, double * alpha, double * beta, double * gamma) {
    // Compute the barycentric coordinates of the point with respect to the triangle
    (*alpha) = (((*t).p2.y - (*t).p3.y) * ((*p).x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*p).y - (*t).p3.y)) /
                  (((*t).p2.y - (*t).p3.y) * ((*t).p1.x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*t).p1.y - (*t).p3.y));
    (*beta) = (((*t).p3.y - (*t).p1.y) * ((*p).x - (*t).p3.x) + ((*t).p1.x - (*t).p3.x) * ((*p).y - (*t).p3.y)) /
                 (((*t).p2.y - (*t).p3.y) * ((*t).p1.x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*t).p1.y - (*t).p3.y));
    (*alpha) =(*alpha) > 0 ? (*alpha) : 0;
    (*alpha) =(*alpha) < 1 ? (*alpha) : 1;
    (*beta) = (*beta) > 0 ? (*beta) : 0;
    (*beta) = (*beta) < 1 ? (*beta) : 1;
    (*gamma) = 1.0 - (*alpha) - (*beta);
    (*gamma) = (*gamma) > 0 ? (*gamma) : 0;
    (*gamma) = (*gamma) < 1 ? (*gamma) : 1;
}


/* Helper function to check if a point is inside a triangle (IT CAN BE REMOVED)*/
__device__ int inside_triangle(struct Triangle * t, struct Point * p) {
    double alpha, beta, gamma;
    barycentric_coordinates(t, p, &alpha, &beta, &gamma); 
    // Check if the barycentric coordinates are positive and add up to 1
    if (alpha > 0 && beta > 0 && gamma > 0) {
        return 1;
    } else {
        return 0;
    }
}

/* Checks if p2 is in a square of size 5 around p1*/
__device__ int inside_square(struct Point *p1, struct Point *p2) {
	return (abs((p1->x - p2->x)) <= 2.5 && abs((p1->y - p2->y)) <= 2.5);
}

/* Helper function to save an image */   
void save_image(char const * filename, int width, int height, double *image){

   FILE *fp=NULL;
   fp = fopen(filename,"w");
   for(int j=0; j<height; ++j){
      for(int i=0; i<width; ++i){
         fprintf(fp,"%f ", image[pixel(i,j,width)]);      
      }
      fprintf(fp,"\n");
   }
   fclose(fp);

}

/* helper function to initialize the points */
void init_points(struct Point* points, int num_points, int width, int height) {
    for(int i = 0; i < num_points; i++) {
        points[i].x =  ((double) rand() / RAND_MAX)*width;
        points[i].y =  ((double) rand() / RAND_MAX)*height;
        points[i].value = 0.f;//(rand() % 10000) / 100.;
        //printf("Point %d [%f,%f]=%f\n", i, points[i].x, points[i].y, points[i].value);
    }
}

__global__ void count_close_points(struct Point* points, int num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < num_points) {
		for(int i=0; i<idx; i++) {
			if (distance(&points[idx], &points[i]) < 100.f) {
				points[idx].value++;
			}
		}
		for(int i=idx+1; i<num_points; i++) {
			if (distance(&points[idx], &points[i]) < 100.f) {
				points[idx].value++;
			}
		}
	}
}

/*Wraper function to launch the CUDA kernel to count the close points*/
void count_close_points_gpu(struct Point* points, int num_points) {
	struct Point* d_points;
	size_t size = num_points * sizeof(struct Point);

	cudaMalloc((void**) &d_points, size);

	cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice);

	int dimGrid = (num_points + (TPB-1)) / TPB; // amount of blocks of size TPB
	int dimBlock = TPB; // int multiple of 32 (warp size) (1024 maximum) try values 128-512

	count_close_points<<<dimGrid, dimBlock>>>(d_points, num_points);

	cudaDeviceSynchronize();

	cudaMemcpy(points, d_points, size, cudaMemcpyDeviceToHost);
	cudaFree(d_points);
}

__global__ void delaunay_triangulation(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles) {
	/* Iterate over every possible triangle defined by three points */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int i_max = num_points;
	int j_max = num_points-1;
	int k_max = num_points-2;

	int i = (idx/(j_max*k_max));
	int j = (idx/k_max) % (j_max) + 1;
	int k = idx % (k_max) + 2;

	if (i<i_max && j<j_max+1 && k<k_max+2 && i<j && j<k) {
		int flag = 0;
		struct Triangle local;
		local.p1 = points[i];
		local.p2 = points[j];
		local.p3 = points[k];
		
		for(int l=0; l<num_points; l++)	{
			if(inside_circle(&points[l], &local)) {
				flag = 1;
				break;
			}
		}

		if(flag == 0) {
			atomicAdd(num_triangles, 1);
			triangles[*num_triangles] = local;
		}
	}
}

/*Wraper function to launch the CUDA kernel to compute delaunay triangulation*/
void delaunay_triangulation_gpu(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles) {
	struct Point* d_points;
	struct Triangle* d_triangles;
	int* d_num_triangles;

	int size_points = sizeof(struct Point) * num_points;
	int size_triangles = sizeof(struct Triangle) * max_num_triangles;

	cudaMalloc((void**) &d_points, size_points);
	cudaMalloc((void**) &d_triangles, size_triangles);
	cudaMalloc((void**) &d_num_triangles, sizeof(int));

	cudaMemcpy(d_points, points, size_points, cudaMemcpyHostToDevice);
	cudaMemcpy(d_num_triangles, num_triangles, sizeof(int), cudaMemcpyHostToDevice);

	int collapsed_points = num_points * (num_points - 1) * (num_points - 2);
	int dimGrid = (collapsed_points + (TPB-1)) / TPB; // amount of blocks of size TPB
	int dimBlock = TPB; // int multiple of 32 (warp size) (1024 maximum) try values 128-512

    delaunay_triangulation<<<dimGrid, dimBlock>>>(d_points, num_points, d_triangles, d_num_triangles);

	cudaDeviceSynchronize();

	cudaMemcpy(num_triangles, d_num_triangles, sizeof(int), cudaMemcpyDeviceToHost);
	size_triangles = sizeof(struct Triangle) * *num_triangles;
	cudaMemcpy(triangles, d_triangles, size_triangles, cudaMemcpyDeviceToHost);
	
	cudaFree(d_points); cudaFree(d_triangles); cudaFree(d_num_triangles);
}

__global__ void save_triangulation_image(struct Point* points, int num_points, struct Triangle* triangles, int num_triangles, int width, int height, double* image) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int i = idx / width;
	int j = idx % width;

	struct Point pixel;
	double alpha, beta, gamma;

	pixel.x = i;
	pixel.y = j;
	image[pixel(i, j, width)] = -1.0;

	for(int k=0; k<num_triangles; k++) {
		barycentric_coordinates(&triangles[k], &pixel, &alpha, &beta, &gamma);
		if (alpha > 0 && beta > 0 && gamma > 0) {
			image[pixel(i, j, width)] = alpha*(triangles[k].p1.value) + beta*(triangles[k].p2.value) + gamma*(triangles[k].p3.value);
			break;
		}
	}
	
	for(int k=0; k<num_points; k++) {
		if(inside_square(&points[k], &pixel)) {
			image[pixel(i, j, width)] = 101.f;
			break;
		}
	}
}

/*Wraper function to launch the CUDA kernel to compute delaunay triangulation. 
Remember to store an image of int's between 0 and 100, where points store 101, and empty areas -1, and points inside triangle the average of value */
void save_triangulation_image_gpu(struct Point* points, int num_points, struct Triangle* triangles, int num_triangles, int width, int height) {
    //create structures
	int pixels = width * height;
    double* image = (double *) malloc(sizeof(double)*pixels);
	double *d_image;
	struct Point* d_points;
	struct Triangle* d_triangles;

	int size_points = sizeof(struct Point) * num_points;
	int size_triangles = sizeof(struct Triangle) * max_num_triangles;

	cudaMalloc((void**) &d_points, size_points);
	cudaMalloc((void**) &d_triangles, size_triangles);
	cudaMalloc((void**) &d_image, sizeof(double)*pixels);

	int dimGrid = (pixels + (TPB-1)) / TPB; // amount of blocks of size TPB
	int dimBlock = TPB; // int multiple of 32 (warp size) (1024 maximum) try values 128-512
	
	save_triangulation_image<<<dimGrid, dimBlock>>>(d_points, num_points, d_triangles, num_triangles, width, height, d_image);

	cudaDeviceSynchronize();

	cudaMemcpy(image, d_image, pixels * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_points); cudaFree(d_triangles); cudaFree(d_image);
	
    //write image
    save_image("image.txt", width, height, image);

    //free structures
    free(image);
}

void printCudaInfo() {
    int devNo = 0;
    printf("\n------------------------------------------------------------------\n");
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);
    printf("Maximum grid size is: (");
    for (int i = 0; i < 3; i++)
        printf("%d, ", iProp.maxGridSize[i]);
    printf(")\n");
    printf("Maximum block dim is: (");
    for (int i = 0; i < 3; i++)
        printf("%d, ", iProp.maxThreadsDim[i]);
    printf(")\n");
    printf("Max threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("------------------------------------------------------------------\n\n");
}

extern "C" int delaunay(int num_points, int width, int height) {
    printCudaInfo();
    
    float time = 0.f;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

    max_num_triangles = num_points*30;
    struct Point * points = (struct Point *) malloc(sizeof(struct Point)*num_points);
    struct Triangle * triangles = (struct Triangle *) malloc(sizeof(struct Triangle)*max_num_triangles);
    printf("Maximum allowed number of triangles = %d\n", max_num_triangles);
    
    init_points(points, num_points, width, height);

    cudaEventRecord(start);
    count_close_points_gpu(points, num_points);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("Counting close points: %f\n", time/1000.f);

    int num_triangles = 0;
    cudaEventRecord(start);
    delaunay_triangulation_gpu(points, num_points, triangles, &num_triangles);
    cudaEventRecord(end);
	cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("Delaunay triangulation: %f\n", time/1000.f);

    printf("Number of generated triangles = %d\n", num_triangles);

    cudaEventRecord(start);
    save_triangulation_image_gpu(points, num_points, triangles, num_triangles, width, height);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("Generate image: %f\n", time/1000.f);

    //Free memory
    free(points);
    free(triangles);

    return 0;
}
    