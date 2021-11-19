#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define checkCudaErrors(val) \
        fprintf(stderr, "CUDA error at %s:%d (%s) \n", __FILE__, __LINE__, cudaGetErrorString(val)) 

__global__ 
void kernel(double *a, double *b, double *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    /*for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            for(k = 0; k < N; ++k) # restera sur le GPU
                c[i][j] += a[i][k] * b[k][j] # restera sur le GPU*/

    /* 
    for(i; i < N; i += (blockDim.x * gridDim.x))
        c[i] = a[i] + b[i];
    */

    for(i = 0; i < N; ++i)
        c[i] += a[i] * b[i + N];
}

void displayMatrix(double *matrix, int N)
{
    for(int i = 0 ; i < N * N ; ++i)
    {
        if(i % N == 0)
            printf("\n");
        printf("%lf ", matrix[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    int N = 10;
    int sz_in_bytes = N * N * sizeof(double);

    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;

    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);

    // Initiate values on h_a and h_b
    for(int i = 0 ; i < N * N ; i++)
    {
	    h_a[i] = 1.;
	    h_b[i] = 2.;
        h_c[i] = 0.;
    }

    displayMatrix(h_a, N);
    displayMatrix(h_b, N);

    // 3-arrays allocation on device 
    checkCudaErrors(cudaMalloc((void**)&d_a, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_b, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_c, sz_in_bytes));

    // copy on device values pointed on host by h_a and h_b
    // (the new values are pointed by d_a et d_b on device)
    checkCudaErrors(cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice));

    dim3  dimBlock(32, 1, 1);
    dim3  dimGrid((N * N + dimBlock.x - 1)/dimBlock.x, 1, 1);
    printf("dim grid: %d", (N + dimBlock.x - 1)/dimBlock.x);
    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N * N);

    // Result is pointed by d_c on device
    // Copy this result on host (result pointed by h_c on host)
    checkCudaErrors(cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost));

    displayMatrix(h_c, N);

    // freeing on device 
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
