# Richarallele

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

## Questions [FR]

<br/>

### I - Modèle d'exécution


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

```c
int my_rank, nb_threads;
    
    #pragma omp parallel private(my_rank) shared(nb_threads) default(none)
    {
        my_rank = omp_get_thread_num();

        # pragma omp single // seul thread qui va dire on est cb
        {
            nb_threads = omp_get_num_threads();
        } // Implicit barrier

        #pragma omp barrier
        printf("I am thread %d (for a total of %d threads)\n", my_rank, nb_threads);
    }
```


<br/>

### II - Partage de travail

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

```c
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

**Q6**




<br/>

### III - Paralléliser un code casseur de mot de passes




<br/>

### IV - Devoir maison 

