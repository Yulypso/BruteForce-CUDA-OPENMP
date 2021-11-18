//gcc breaker_for.c -o breaker.pgr -lcrypt -lm && ./breaker.pgr

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#define __USE_GNU
#ifdef __APPLE__
#include <unistd.h>
#else
#include <crypt.h>
#endif
#include <omp.h>

int search_all_1(char *crypted, int length, int first_char, int last_char)
{
#ifndef __APPLE__
    struct crypt_data data;
#endif
    int loop_size = last_char - first_char;
    int cryptlen = strlen(crypted);
    int max_iter = powl(loop_size, length);
    char tab[length];
    int i = 0, j = 0, ret = 0;

    tab[length - 1] = '\0';

    for (int j = 0; j < length; ++j)
        tab[j] = first_char;

    printf("max_iter = %lu \n", (unsigned long)max_iter);

#ifdef __APPLE__
#pragma omp parallel firstprivate(i, j, loop_size) private(tab) shared(max_iter, length, cryptlen, crypted, first_char, last_char, ret) default(none)
#else
#pragma omp parallel firstprivate(i, j, loop_size) private(tab, data) shared(max_iter, length, cryptlen, crypted, first_char, last_char, ret) default(none)
#endif
    {
        for (j = 0; j < length; ++j)
            tab[j] = first_char + (loop_size / omp_get_num_threads()) * omp_get_thread_num();

#pragma omp for private(i, j)
        for (i = 0; i < max_iter; ++i)
        {
#ifdef __APPLE__
            if (!strncmp(crypted, crypt(tab, "salt"), cryptlen))
#else
            if (!strncmp(crypted, crypt_r(tab, "salt", &data), cryptlen))
#endif
            {

                printf(">>> Thread n°%d: password found: %s\n", omp_get_thread_num(), tab);
                ret = i;
#pragma omp cancel for
            }
#pragma omp cancellation point for
            ++tab[0];

            for (j = 0; j < length - 1; ++j)
            {
                if (last_char == tab[j])
                {
                    tab[j] = first_char;
                    ++tab[j + 1];
                }
            }
        }
#pragma omp cancel parallel
    }
    return ret;
}

int main(int argc, char **argv)
{
    char *password;
    struct timeval t1;
    struct timeval t2;
    int first_char, last_char;
    int cmp;

    if (argc == 1)
    {
        password = "A$4c";
        first_char = 32;
        last_char = 126;
        /* ---ASCII values---
         * special characters: 	32 to 47
         * numbers: 		48 to 57
         * special characters: 	58 to 64
         * letters uppercase: 	65 to 90
         * special characters: 	91 to 96
         * letters lowercase: 	97 to 122
         * special characters: 	123 to 126
         * */
    }
    else if (argc == 4)
    {
        password = argv[1];
        first_char = atoi(argv[2]);
        last_char = atoi(argv[3]);
    }
    else
    {
        printf("usage: breaker <password> <first_ch> <last_ch>\n");
        printf("default: breaker A$4c 32 126\n");
        printf("exemple to break the binary password 1101000:\n");
        printf("breaker 1101000 48 49\n");
        exit(0);
    }
    char *crypted0 = crypt(password, "salt");
    char *crypted = (char *)malloc((strlen(crypted0) + 1) * sizeof(char));
    strcpy(crypted, crypted0);

    printf("*running parameters*\n");
    printf(" -password length:\t%lu digits\n", strlen(password));
    printf(" -digits:\t\tfrom -%c- to -%c-\n", first_char, last_char);
    printf(" -crypted to break:\t%s\n", crypted);

    gettimeofday(&t1, NULL);
    cmp = search_all_1(crypted, strlen(password), first_char, last_char);
    gettimeofday(&t2, NULL);

    double period = (double)((int)(t2.tv_sec - t1.tv_sec)) + ((double)(t2.tv_usec - t1.tv_usec)) / 1000000;

    printf("time: %dmin %.3fs \n", (int)((t2.tv_sec - t1.tv_sec)) / 60, (double)((int)(t2.tv_sec - t1.tv_sec) % 60) + ((double)(t2.tv_usec - t1.tv_usec)) / 1000000);
    printf("#tries: %d\n", cmp);
    printf("=> efficiency: %.f tries/s\n", (double)cmp / period);

    return EXIT_SUCCESS;
}
