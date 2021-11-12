#include <stdio.h>
#include <omp.h>

int main() {
    int id;
    #pragma omp parallel private(id) default(none)
    {
        id = omp_get_thread_num();
        printf("%d: Hello, World!\n", id);
    }
    return 0;
}