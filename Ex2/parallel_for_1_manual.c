#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[])
{
    int nb_threads=-1;
    #pragma omp parallel shared(nb_threads) default(none)
    {
        nb_threads = omp_get_num_threads();
    }

    int size = nb_threads*16, i; // 6 * 16
    int * array = (int *)malloc(sizeof(int)*size);

    for(i=0; i<size; i++)
        array[i] = i+1;

    int verif_sum = 0;
    for(i=0; i<size; i++)
        verif_sum += array[i];

    int tmp_sum = 0, sum = 0, j;
    #pragma omp parallel for schedule(static, 1) firstprivate(tmp_sum) shared(array, size) reduction(+:sum) default(none)
    {
        for(j=0; j<size; j++)
        {
            tmp_sum += array[j];
            sum += array[j];
        }
    }

    if(sum == verif_sum)
        printf("OK! sum = verif_sum! = %d\n", sum);
    else
        printf("Error! sum != verif_sum! (sum = %d ; verif_sum = %d)\n", sum, verif_sum);
    free(array);

    return 0;
}