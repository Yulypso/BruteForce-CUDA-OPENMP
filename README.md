# Richarallele (OpenMP and CUDA)

## Author

[![Linkedin: Thierry Khamphousone](https://img.shields.io/badge/-Thierry_Khamphousone-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/tkhamphousone/)](https://www.linkedin.com/in/tkhamphousone)

---

<br/>

## Setup

```bash
$ git clone https://github.com/Yulypso/Richarallele.git
$ cd Richarallele
```

---

<br/>

## Required Libraries and tools

```bash
$ sudo apt install gcc
```
> [GCC](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/)

> [OpenMP](https://www.openmp.org) 

> [CUDA](https://developer.nvidia.com/cuda-zone)

---

<br/>

## Compile with Clang

```bash
$ clang -Wall -o BIN -Xpreprocessor -fopenmp <file.c> -lomp
```

## Compile with GCC
```bash
$ gcc -Wall -o BIN -fopenmp <file.c>
```

## Execution with 8 threads:
```bash
$ OMP_NUM_THREADS=8 ./BIN
```

<br/>

## Questions - OpenMp[FR]

<br/>

### I - Modèle d'exécution
> gcc print_rank_1.c -o print_rank_1.pgr -fopenmp && OMP_NUM_THREADS=8 ./print_rank_1.pgr

<br/>

**Q1:** Ce programme affiche l’identifiant unique (rank) du thread courant parmis tous les threads demandés.   
Que font les fonctions omp_get_num_threads() et omp_get_thread_num() ?
> omp_get_num_threads(): affiche le nombre total de thread dans une région parallèle.  
> omp_get_thread_num(): Affiche le rang (id) local du thread.  

**Q2**: Pour le moment, les appels à ces fonctions et le print ne sont pas dans une région parallèle. Insérer une région parallèle OpenMP avec le pragma #pragma omp parallel.  

**Q3**: Insérer une barrière OpenMP (#pragma omp barrier avant le print. Que se passe-t-il ? Pourquoi ? Comment peut-on corriger le problème ?  
> #pragma omp barrier ajouté avant le printf()   
> #pragma omp parallel firstprivate(my_rank) shared(nb_threads) default(none) modifié:
> private permet de déclarer une variable locale à chaque région parallele initialisé aléatoirement.  
> firstprivate est une utilisation particuliere de private et permet de conserver la valeur d'initialisation avant d'entrer dans la région parallele. (Dans notre cas, il n'est pas utile de l'utiliser ici)  
> shared permet de définir une variable qui est partagée entre les threads.  

```c++
int my_rank, nb_threads;

#pragma omp parallel private(my_rank) shared(nb_threads) default(none)
{
    my_rank = omp_get_thread_num();

    # pragma omp single
    {
        nb_threads = omp_get_num_threads();
    }

    #pragma omp barrier
    printf("I am thread %d (for a total of %d threads)\n", my_rank, nb_threads);
}
```


<br/>

### II - Partage de travail
> gcc parallel_for_1_manual.c -o parallel_for_1_manual.pgr -fopenmp && OMP_NUM_THREADS=6 ./pa- rallel_for_1_manual.pgr

<br/>

**Q4**: Le programme parcourt un tableau d’entier et réalise la somme de chaque entier. Exécuter-le. Quel est le problème ?  
> Sum est différent de verif sum car on a essayé de paralleliser la somme non correctement.  
> Chaque thread calcule la meme partie de la somme et obtiennent tmp_sum = 4656. De plus, étant donné qu'on n'a pas de contrôle/protection lors de la somme de tous les tmp_sum, on se retrouve avec sum = 9312.  

**Q5**: Répartir le travail de parcourt de tableau et de somme entre les threads en utilisant le rang du thread.   
Utiliser une variable temporaire pour stocker la somme partielle, avant de sommer les sommes partielles pour obtenir la valeur finale.   
Penser à protéger la somme finale (avec un atomic ou une section critique).  
> Répartition du travail en donnant au premier thread les valeurs du tableau allant de 0 à 15, au second thread de 16 à 31, etc.  
> La somme partielle est stockée dans tmp_sum.  
> Atomic est utilisé ici pour autoriser la lecture et la modification de la variable à un thread à la fois.  
> Possibilité d'utiliser section critique mais plus couteux pour une seule instruction.


```c++
int tmp_sum = 0, sum = 0;

#pragma omp parallel firstprivate(tmp_sum) shared(array, sum, size, nb_threads) default(none)
{
    int j;
    for(j=omp_get_thread_num() * size/nb_threads; j<(omp_get_thread_num()+1) * size/nb_threads; j++)
        tmp_sum += array[j];

    #pragma omp atomic
    sum += tmp_sum;

    for(j=0; j<nb_threads; j++) {
        if (omp_get_thread_num() == j)
        {
            #pragma omp barrier
            printf("tmp_sum = %d \n", tmp_sum);
        }
    }
}
```

**Q6**: Il est possible de répartir automatiquement les itérations d’une boucle entre les différents threads avec le pragma #pragma omp for.   
Utiliser ce pragma pour remplacer votre découpage manuel.  
  
```c++
int tmp_sum = 0, sum = 0;

#pragma omp parallel firstprivate(tmp_sum) shared(array, sum, size, nb_threads) default(none)
{
    int j;
    #pragma omp for schedule(static, 1)
    for(j=0; j<size; j++)
        tmp_sum += array[j];

    #pragma omp atomic
    sum += tmp_sum;

    for(j=0; j<nb_threads; j++) {
        if (omp_get_thread_num() == j)
        {
            #pragma omp barrier
            printf("tmp_sum = %d \n", tmp_sum);
        }
    }
}
```

**Q7**: Au lieu de protéger la somme finale avec une section critique, il est possible de spécifier à une région parallèle (ou une boucle for) qu’une réduction à lieu dans celle-ci.   
Utiliser cette fonctionnalité.  

```c++
int tmp_sum = 0, sum = 0;

#pragma omp parallel firstprivate(tmp_sum) shared(array, sum, size, nb_threads) default(none)
{
    int j;
    #pragma omp for schedule(static, 1) reduction(+:sum)
    for(j=0; j<size; j++)
    {
        tmp_sum += array[j];
        sum += array[j];
    }

    for(j=0; j<nb_threads; j++) {
        if (omp_get_thread_num() == j)
        {
            #pragma omp barrier
            printf("tmp_sum = %d \n", tmp_sum);
        }
    }
}
```

**Q8**: Il est possible de fusionner les pragmas #pragma omp parallel et #pragma omp for en un seul pragma.  
Supprimer la boucle permettant l’affichage des sommes partielles, et utiliser ce pragma combiné.  

```c++
int tmp_sum = 0, sum = 0, j;

#pragma omp parallel for schedule(static, 1) firstprivate(tmp_sum) shared(array, size) reduction(+:sum) default(none)
{
    for(j=0; j<size; j++)
    {
        tmp_sum += array[j];
        sum += array[j];
    }
}
```

<br/>

### III - Paralléliser un code casseur de mot de passes
> gcc breaker_for.c -o breaker_for.pgr -fopenmp -lm -lcrypt && ./breaker_for.pgr

<br/>

**Q9**: Le programme fait une recherche gloutonne pour trouver un mot de passe à partir du mot de passe crypté.  
Pour le moment, le code test toutes les possibilités avec une boucle for.  
Paralléliser cette boucle pour que la recherche soit plus rapide.  
Ouvrir la page de man de la fonction crypt pour vérifier si celle-ci peut être utilisé en parallèle.  



<br/>

### IV - Devoir maison 

Arrêt des threads lorsque le mot de passe a été trouvé. 

<br/>

---

<br/>

## Questions - CUDA[FR]

<br/>

### I - Modèle d'exécution
> nvcc Ex1.cu -o Ex1.pgr

<br/>

**Q1:** Quelle partie du programme doit s’exécuter sur l’hôte ? Quelle partie sur
le device ?
> [Etape 1] hote: Initialisation et allocation mémoire sur le CPU (hote)
```c
int N = 1000;
int sz_in_bytes = N*sizeof(double);

double *h_a, *h_b, *h_c;
double *d_a, *d_b, *d_c;

h_a = (double*)malloc(sz_in_bytes);
h_b = (double*)malloc(sz_in_bytes);
h_c = (double*)malloc(sz_in_bytes);

// Initiate values on h_a and h_b
for(int i = 0 ; i < N ; i++)
{
    h_a[i] = 1./(1.+i);
    h_b[i] = (i-1.)/(i+1.);
}
```

</br>

> [Etape 2] device: Allocation mémoire sur le GPU (device)
```c
// 3-arrays allocation on device 
cudaMalloc((void**)&d_a, sz_in_bytes);
cudaMalloc((void**)&d_b, sz_in_bytes);
cudaMalloc((void**)&d_c, sz_in_bytes);
```  

</br>

> [Etape 3] host vers device: Transfert des données CPU vers GPU
```c
 // copy on device values pointed on host by h_a and h_b
// (the new values are pointed by d_a et d_b on device)
cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice);
```

</br>

> [Etape 4] device: Execution d'un noyau de calcul sur le GPU
```c
dim3  dimBlock(64, 1, 1);
dim3  dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);
kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);
``` 

</br>

> [Etape 5] device vers host: Rapatriement des données du GPU vers le CPU
```c
// Result is pointed by d_c on device
// Copy this result on host (result pointed by h_c on host)
cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost);
```

</br>

> [Etape 6] device: Liberation mémoire sur le GPU
```c
// freeing on device 
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

</br>

> [Etape 7] host: Liberation mémoire sur le CPU
```c
free(h_a);
free(h_b);
free(h_c);
```

</br>

**Q2:** Que calcule ce programme ?
> h_a:  1, 0.5, 0.3, 0.2, 0.16
> h_b: -1, 0.0, 0.3, 0.5, 0.6
> h_c:  0, 0.5, 0.6, 0.7, 0.82
> Le programme calcule f(x) = (1/(1+x)) + (x-1)/(x+1)


**Q3:** Combien y a-t-il de blocs au total ? Combien de threads par blocs ? Combien de threads au total ?
> Nombre total de blocs: (1000 + 64 - 1)/64 = (16.6) soit 17 blocs; l'idée est de s'ajouter 64 thread au numérateur pour s'assurer d'avoir suffisamment de bloc de 64 threads pour le calcul. 1000/64 = 15.6 (15 blocs => insuffisant) et 1064/64 = 16.6 (16 blocs, soit 1024 threads)
> Il y a 64 threads par blocs
> N vaut 1000, soit 1024 - 1000 = 24 threads non utilisé du bloc n°15. 

**Q4:** Émuler sur CPU le comportement du GPU sans utiliser le SDK CUDA. Pour ce faire, réécrire le programme en C/C++ avec les contraintes suivantes :
    1. utilisation d’une fonction kernel
    2. utillisation des grilles de blocs et de threads


<br/>

### II - Debugging

Nous considérons à présent le fichier err1.cu. Ce programme est censé calculer la somme de deux vecteurs. À la fin de l’exécution, une erreur relative est calculée entre le vecteur issu du GPGPU et celui calculé sur CPU.  

**Q5:** Compiler et exécuter le programme. Le résultat est-il correct ?
> Non il y a des erreurs
```bash
./BIN
CUDA error at err1.cu:35 (no error) 
CUDA error at err1.cu:36 (no error) 
CUDA error at err1.cu:37 (no error) 
CUDA error at err1.cu:39 (no error) 
CUDA error at err1.cu:40 (no error) 
CUDA error at err1.cu:46 (no error) 
CUDA error at err1.cu:48 (no error) 
CUDA error at err1.cu:49 (no error) 
CUDA error at err1.cu:50 (no error) 
ERROR (Relative error : 5.681e-01)
```

**Q6:** Encadrer chaque appel à CUDA par la macro checkCudaErrors.
```c
#define checkCudaErrors(val) \
        fprintf(stderr, "CUDA error at %s:%d (%s) \n", __FILE__, __LINE__, cudaGetErrorString(val)) 
```

**Q7:** Calculer le nombre total de threads. Comparer le à N. Qu’en déduire ?
> Nombre total de threads = nombre de blocs * nombre de threads par bloc
> Nombre total de threads = dimGrid * dimBlock
> Nombre total de threads = 10 * 64 = 640 threads.

> N = 1000
> On peut en déduire qu'il n'y a pas assez de threads.

**Q8:** Corriger le code CUDA selon les deux possibilités suivantes:
    1. Corriger le nombre de blocs pour traiter tous les indices des tableaux.
    2. Sans changer les tailles de la grille et des blocs, modifier le kernel (prendre en compte le nombre total de threads).

```c
// 1.
int N = 640;
SUCCESS (Relative error : 0.000e+00)
```

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2. le thread 0 va travailler avec i=0 et i=640
for(i; i < N; i += (blockDim.x * gridDim.x))
    c[i] = a[i] + b[i];
```

<br/>

### III - Traduction d’un code C en code CUDA

**Q9:** Le squelette de la traduction en CUDA du programme contenu dans le fichier Ex3_1.c vous est fourni dans le répertoire CODE (fichier Ex3_1.cu. A vous de remplir ce squelette (parties notées "A COMPLETER" dans le code) pour réaliser l’initialisation du tableau a sur le GPU.

**Q10:** Le schéma d’accès aux données du tableau a est-il efficace ?pourquoi ?

**Q11:** Nous allons vérifier cette hypothèse. Insérer des appels à gettimeofday pour mesurer le temps du kernel CUDA (et uniquement du kernel). Relevez le temps mesuré.

**Q12:** Nous allons maintenant changer le schéma d’accés aux données du tableau a en modifiant les valeurs dans le tableau d’indirection b. Remplacez la valeur actuelle de STEP par 1. Que cela change-t-il pour les accés au tableau a ? Relevez le temps mesuré. Est-il meilleur  ? Pourquoi ?

**Q13:** Jusqu’à présent, nous n’utilisions qu’un seul bloc de plusieurs threads. Nous allons changé cela. Modifiez la valeur de nBlocks pour la mettre à 16. Modifiez votre kernel pour avoir un calcul d’indice correct. Relevez le temps mesuré. Est-il meilleur ? Pourquoi ?

<br/>

### IV - Réduction somme en CUDA
Une réduction somme consiste à additionner toutes les valeurs d’une tableau. Une écriture séquentielle d’une réduction pourrait être la suivante :  
```c
float sum = 0 ;
for (int i = 0 ; i < ntot ; i++)
    sum += tab[i] ;
```

    1. Une première réduction dans chaque bloc. On obtient ainsi à la fin un tableau dimensionné au nombre de blocs et dont les valeurs sont les sommes partielles de chaque bloc.
    2. Une seconde réduction sur les sommes partielles. On obtient ainsi la somme totale des éléments du tableau.

**Q14:** Implémenter le kernel reduce_kernel(float *in, float *out) (voir fi-chier reduce.cu) permettant de faire les sommes partielles par bloc.  
in est le tableau de valeurs à réduire dimensionné au nombre total de threads dans la grille, et out le tableau de valeurs réduites par bloc, dimensionné au nombre de bloc.  
Pour réaliser cette réduction, vous utiliserez une méthode arborescente, ainsi que la fonction __syncthreads() qui permet de synchroniser à l’intérieur d’un kernel tous les threads d’un même bloc.
Nous nous placerons sous les hypothèses suivantes:

    1. Le nombre de blocs et de threads par bloc sont des puissances de 2. 
    2. La taille du tableau est égale au nombre de threads.

**Q15:** Utiliser le même kernel pour terminer la réduction (étape 2).

**Q16:** Généraliser la réduction à une taille quelconque de tableau.

<br/>

### V - Réduction somme en CUDA

Le calcul de l’indice global d’un thread est généralement très important dans un kernel CUDA. Nous allons augmenté le nombre de dimensions des grilles et des blocs dans le programme précédent pour s’habituer à manipuler plus de dimensions dans notre calcul d’indice global

**Q17:** Passez à des tailles de grille et de bloc à deux dimensions. Donnez le calcul de l’indice global du thread avec ces 2x2 dimensions.


---

<br/>

## Devoir maison - Multiplication de matrice[FR]

Ecrire un code cuda pour calculer la multiplication de matrice carré de taille N.

Pseudo code:
```c
for(i = 0; i < N; ++i)
    for(j = 0; j < N; ++j)
        for(k = 0; k < N; ++k) // restera sur le GPU
            c[i][j] += a[i][k] * b[k][j] // restera sur le GPU
```

Bien faire l'allocation mémoire et compiler Cuda

grand tableau N * N 
pour acceder: i * N + j

faire des blocs de taille max 32 threads = 1 warp (utiliser la puissance maximale du warp)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define checkCudaErrors(val) \
        fprintf(stderr, "CUDA check at %s:%d (%s) \n", __FILE__, __LINE__, cudaGetErrorString(val)) 

__global__ void kernel(double *a, double *b, double *c, int N)
{

    /* 
    * En definissant le nombre de bloc de thread necessaire correctement
    * Chaque thread va effectuer seulement un calcul de somme ligne [i] matrice A * colonne [j] matrice B
    * et sauvegarder le résultat dans la case [i][j] en question de la matrice C
    */

    int i = (blockIdx.x * blockDim.x + threadIdx.x); // i (= lines) prend des valeurs entre 0 à N-1
    int j = (blockIdx.y * blockDim.y + threadIdx.y); // j (= columns) prends des valeurs entre 0 à N-1

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
    if (i < N && j < N) // On s'assure de bien rester dans la matrice.
    {
        for(int k = 0; k < N; ++k)
            c[i * N + j] += a[i * N + k] * b[j + k * N];
    }
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
    //srand((unsigned) time(0));

    /* CPU (host)
    * Initialisation de la taille de la matrice et allocation mémoire des trois matrices
    */
    int N = 3;
    int sz_in_bytes = N * N * sizeof(double);

    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;
    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);

    /* CPU (host)
    * Initialisation des matrices A et B par des valeurs aléatoires
    */
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
    * Execution sur le GPU (calculs paralleles) en considérant N = 10
    *
    * dimBlock = 8 * 4 = 32 CUDA core (thread) dans un block correspondant à la taille d'un warp. Cela permet d'éviter d'avoir des threads sans calculs à effectuer au sein d'un warp.
    * Un SM démarre 4 warps de 32 threads, nous avons donc 128 threads par SM.
    * 
    * dimGrid =  N * N = 100 CUDA blocks (thread block). 
    * Chacun des 100 blocks de la grid seront executé par des SM (streaming multiprocessor). 
    * Cela nous permet de faire du mapping et d'executer les calculs par blocks.
    *
    * Appel du Kernel depuis le CPU déclenchant la fonction autant de fois que le nombre total de CUDA Core demandé.
    */
    dim3  dimBlock(8, 4, 1); 
    dim3  dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N  + dimBlock.y - 1) / dimBlock.y, 1); 
    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);

    /* GPU (device) -> CPU (host)
    * Copie des variables résultantes du GPU vers le CPU*
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
```