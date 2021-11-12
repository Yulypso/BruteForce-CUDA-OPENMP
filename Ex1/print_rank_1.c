#include <omp.h>

#include <stdlib.h>
#include <stdio.h>


int main(int argc, char * argv[])
{
    int my_rank = -1;
    int nb_threads = -1;

    #pragma omp parallel firstprivate(my_rank) shared(nb_threads) default(none)
    {
        my_rank = omp_get_thread_num();

        # pragma omp single // seul thread qui va dire on est cb
        {
            nb_threads = omp_get_num_threads();
        }; // Implicit barrier

        #pragma omp barrier
        printf("I am thread %d (for a total of %d threads)\n", my_rank, nb_threads);

    }
    return 0;

}

/*
 * Variable d'environnement: OMP_NUM_THREADS=8 variable
 *
 * Q1: omp_get_num_threads(): affiche le numéro du thread
 *     omp_get_thread_num(): affiche le nombre total de thread
 *
 * Q2: Insertion région parallele OpenMP avec pragma omp parallel
 *
 * Q3: Insertion barriere OpenMP: en rajoutant la barriere, le programme ne fonctionne plus
 * on a que des thread 0. Pour que cela fonctionne on peut rajouter  private(my_rank, nb_threads) default(none)
 * à parallel. la barriere permet de dire que tant que tout le monde n'a pas traversé la barriere, on attend.
 */