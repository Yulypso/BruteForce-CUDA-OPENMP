#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define checkCudaErrors(val) \
        fprintf(stderr, "CUDA check at %s:%d (%s) \n", __FILE__, __LINE__, cudaGetErrorString(val)) 

__global__ void kernel(double *a, double *b, double *c, int N)
{
    /* 
    *  - Ecriture dans la matrice c avec l'indice i allant de 0 à N*N
    *
    *  - parcours de la matrice a par ligne :
    *       - premiere ligne est donnée par : line=0 * N
    *       - parcours des éléments : (line=0) * N + k
    *
    *   - parcours de la matrice b par colonne:
    *       - premiere colonne est donnée par : col=0
    *       - parcours des éléments : (col=0) + k * N
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int line = i / N; 
    int col = i % N; 

    /* Methode 1
    * permet de donner a un meme thread du calcul supplémentaire (décalé de n threads) si le nombre de thread est inférieur à la taille de la matrice: N * N
    * for(i; i < N * N; i += (blockDim.x * gridDim.x)) 
    *   for(int k = 0; k < N; ++k)
    *       c[i] += a[line * N + k] * b[col + k * N];
    */
    
    /* Methode 2
    * En definissant le nombre de bloc de thread necessaire correctement
    */
    for(int k = 0; k < N; ++k)
        c[i] += a[line * N + k] * b[col + k * N];
}

void displayMatrix(double *matrix, int N)
{
    /*
    * Affichage des matrices
    */

    for(int i = 0 ; i < N * N ; ++i)
    {
        if(i % N == 0)
            printf("\n");
        printf("%2.0lf ", matrix[i]);
    }
    printf("\n\n");
}

int main(int argc, char **argv)
{
    srand((unsigned) time(0));

    /* CPU (host)
    * Initialisation des variables et allocation mémoire des trois matrices
    */
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
	    h_a[i] = rand() % 100;
	    h_b[i] = rand() % 100;
        h_c[i] = 0;
    }

    /* CPU (host)
    * Affichage des matrices a et b
    */
    displayMatrix(h_a, N);
    displayMatrix(h_b, N);

    /* GPU (device)
    * Allocation mémoire des trois matrices
    */
    checkCudaErrors(cudaMalloc((void**)&d_a, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_b, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_c, sz_in_bytes));

    /* CPU (host) -> GPU (device)
    * Copie des variables du CPU vers le GPU
    */
    checkCudaErrors(cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice));

    /* GPU (device)
    * Execution sur le GPU (calculs paralleles)
    */
    dim3  dimBlock(32, 1, 1); // 32 car c'est la taille d'un warp, et un SM démarre 4 warps de 32 threads, soit 128 threads
    dim3  dimGrid((N * N + dimBlock.x - 1)/dimBlock.x, 1, 1); // calcul du nombre de bloc nécessaire B tel que B >= N*N 
    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);

    /* GPU (device) -> CPU (host)
    * Copie des variables résultantes du GPU vers le CPU
    */
    checkCudaErrors(cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost));

    /* CPU (host)
    * Affichage de la matrice résultante c
    */
    displayMatrix(h_c, N);

    /* GPU (device)
    * Libération de la mémoire sur le GPU
    */
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    /* CPU (host)
    * Libération de la mémoire sur le CPU
    */
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
