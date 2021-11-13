#include <omp.h>
#include <stdio.h>


int main(int argc, char * argv[])
{
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
    return 0;
}