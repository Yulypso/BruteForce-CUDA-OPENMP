#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <algorithm>
#include <cstdlib>
using namespace std;
 
// ====================================
// kernel declaration
// ====================================
__global__
void sum(float *x, float *y, float *z, int size) {
    // compute Global Thread InDex
    int gtid = threadIdx.x ;
   
    if (gtid < size) {
        z[gtid] = x[gtid] + y[gtid]; // paralell part
    }
}
 
// ====================================
// main function
// ====================================
int main() {
    printf("First Cuda program :)\n");
    const int SIZE = 512;
 
    // allocate data on computer global memory
    float *x_cpu = new float[ SIZE ];
    float *y_cpu = new float[ SIZE ];
    float *z_cpu = new float[ SIZE ];
 
    // fill x, y
    std::fill(&x_cpu[0], &x_cpu[SIZE], 1);
    std::fill(&y_cpu[0], &y_cpu[SIZE], 2);
 
    // allocate data on GPU global memory
    float *x_gpu, *y_gpu, *z_gpu;
 
    cudaMalloc( (void**) &x_gpu, SIZE * sizeof(float) );
    cudaMalloc( (void**) &y_gpu, SIZE * sizeof(float) );
    cudaMalloc( (void**) &z_gpu, SIZE * sizeof(float) );
 
    // copy x and y resp. to dev_x, dev_y
    cudaMemcpy(x_gpu, x_cpu, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y_cpu, SIZE * sizeof(float), cudaMemcpyHostToDevice);
 
    // define number of threads
    dim3 grid(1, 1, 1);
    dim3 block(SIZE, 1, 1);
    // call kernel
    sum<<< grid, block >>>(x_gpu, y_gpu, z_gpu, SIZE);
 
    // copy result back to computer's global memory
    cudaMemcpy(z_cpu, z_gpu, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
 
    // free memory
    cudaFree( x_gpu );
    cudaFree( y_gpu );
    cudaFree( z_gpu );
 
    delete [] x_cpu;
    delete [] y_cpu;
    delete [] z_cpu;
   
    exit(EXIT_SUCCESS);
}