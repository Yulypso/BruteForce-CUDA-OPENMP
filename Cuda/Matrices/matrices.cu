#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define checkCudaErrors(val) \
        fprintf(stderr, "CUDA check at %s:%d (%s) \n", __FILE__, __LINE__, cudaGetErrorString(val)) 

__global__ void kernel(double *a, double *b, double *c, unsigned long long int N)
{
    /* 
    * En definissant le nombre de bloc de thread necessaire correctement
    * Chaque thread va effectuer seulement un calcul de somme ligne [i] matrice A * colonne [j] matrice B
    * et sauvegarder le résultat dans la case [i][j] en question de la matrice C
    */

    unsigned long long int i = (blockIdx.x * blockDim.x + threadIdx.x); // i (= lines) prend des valeurs entre 0 à N-1
    unsigned long long int j = (blockIdx.y * blockDim.y + threadIdx.y); // j (= columns) prends des valeurs entre 0 à N-1

    /* 
    *  - Ecriture dans la matrice c avec l'indice i allant de 0 à N*N
    *
    *  - parcours de la matrice a par ligne :
    *       - premiere ligne est donnée par : i=0 * N
    *       - parcours des éléments : (i=0) * N + k
    *
    *   - parcours de la matrice b par colonne:
    *       - premiere colonne est donnée par : j=0
    *       - parcours des éléments : (j=0) + k * N
    */
    //if (i < N && j < N) // On s'assure de bien rester dans la matrice.
    {
        for(unsigned long long int k = 0; k < N; ++k)
            c[i * N + j] += a[i * N + k] * b[j + k * N];
    }
}

void displayMatrix(double *matrix, unsigned long long int N)
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
    * Initialisation de la taille de la matrice et allocation mémoire des trois matrices
    */
    unsigned long long int N = 1000;
    unsigned long long int sz_in_bytes = N * N * sizeof(double);

    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;
    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);

    /* CPU (host)
    * Initialisation des matrices A et B par des valeurs aléatoires
    */
    for(unsigned long long int i = 0 ; i < N * N ; ++i)
    {
	    h_a[i] = rand() % 100;
	    h_b[i] = rand() % 100;
        h_c[i] = 0;
    }

    /* CPU (host)
    * Affichage des matrices a et b
    */
    if(N <= 10)
    {
        displayMatrix(h_a, N);
        displayMatrix(h_b, N);
    }

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
    * Execution sur le GPU (calculs paralleles) en considérant N = 10
    *
    * dimBlock = 8 * 4 = 32 CUDA core (thread) dans un block correspondant à la taille d'un warp. Cela permet d'éviter d'avoir des threads sans calculs à effectuer au sein d'un warp.
    * Un SM démarre 4 warps de 32 threads, nous avons donc 128 threads par SM.
    * 
    * Calcul de la dimension de la grid necessaire en fonction du nombre de thread par blocks.
    * dimGrid =  2 * 3 = 6 CUDA blocks (thread block). 
    * Chacun des 6 blocks de la grid seront executé par des SM (streaming multiprocessor). 
    * Cela nous permet de faire du mapping et d'executer les calculs par blocks.
    *
    * Dans mon cas, la GTX 750Ti permet d'avoir 640 CUDA Core et d'executer 5 SM en même temps.
    *
    * Appel du Kernel depuis le CPU déclenchant la fonction autant de fois que le nombre total de CUDA Core demandé.
    */
    dim3  dimBlock(8, 4, 1); 
    dim3  dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N  + dimBlock.y - 1) / dimBlock.y, 1); 
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  

    printf("\nGPU Elapsed time: %3.1f ms\n", milliseconds);

    /* GPU (device) -> CPU (host)
    * Copie des variables résultantes du GPU vers le CPU*
    */
    checkCudaErrors(cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost));

    /* CPU (host)
    * Affichage de la matrice résultante c
    */
    if(N <= 10)
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
