/* 
* Thierry KHAMPHOUSONE, ESIEA MS-SIS 2021-2022
* Programmation parallele OpenMp & Cuda
* => Pour compiler et executer: make
*/

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
    long double loop_size = last_char - first_char;
    int cryptlen = strlen(crypted);
    long double max_iter = powl(loop_size, (long double)length);
    char tab[length];
    int i = 0, j = 0, ret = 0;

    printf("max_iter = %lu \n", (unsigned long)max_iter);

#ifdef __APPLE__
#pragma omp parallel firstprivate(i, j, loop_size) private(tab) shared(max_iter, length, cryptlen, crypted, first_char, last_char, ret) default(none)
#else
#pragma omp parallel firstprivate(i, j, loop_size) private(tab, data) shared(max_iter, length, cryptlen, crypted, first_char, last_char, ret) default(none)
#endif
    {
        /*
        * Initialisation des tableaux pour chaque thread à des emplacements différents 
        * Par exemple: 
        * [thread 0]: AAAA
        * [thread 1]: GGGG
        * [thread 2]: MMMM
        *  ...
        */
        for (j = 0; j < length; ++j)
            tab[j] = first_char + (loop_size / omp_get_num_threads()) * omp_get_thread_num();

#pragma omp for private(i, j)
        for (i = 0; i < (int)max_iter; ++i)
        {
#ifdef __APPLE__
            /*
            * La fonction crypt_r n'existe pas sur MacOS et crypt() n'est pas thread safe.
            */
            if (!strcmp(crypted, tab))
#else
            /*
            * Si le mot de passe chiffré correspond au mot de passe que nous venons de chiffrer par brute force
            * On demande au thread courant de quitter le for avec:  #pragma omp cancel for
            * Les autres threads vérifient si la boucle a été annulée ou pas lorsqu'ils verront: #pragma omp cancellation point for
            * Si oui, ils quittent également la boucle for, sinon ils continuent le brute-force.
            */
            if (!strncmp(crypted, crypt_r(tab, "salt", &data), cryptlen))
#endif
            {
                printf(">>> Thread n°%d: password found: %s\n", omp_get_thread_num(), tab);
                ret = i;
#pragma omp cancel for
            }
#if defined(__APPLE__)
            /*
            * #pragma omp cancellation point for ne fonctionne pas sur MacOs
            */
            if (ret != 0)
            {
#pragma omp cancel for
            }
#else
#pragma omp cancellation point for
#endif
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
#ifndef __APPLE__
#pragma omp cancel parallel
#endif
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
#ifdef __APPLE__
    cmp = search_all_1(password, strlen(password), first_char, last_char);
#else
    cmp = search_all_1(crypted, strlen(password), first_char, last_char);
#endif
    gettimeofday(&t2, NULL);

    double period = (double)((int)(t2.tv_sec - t1.tv_sec)) + ((double)(t2.tv_usec - t1.tv_usec)) / 1000000;

    printf("time: %dmin %.3fs \n", (int)((t2.tv_sec - t1.tv_sec)) / 60, (double)((int)(t2.tv_sec - t1.tv_sec) % 60) + ((double)(t2.tv_usec - t1.tv_usec)) / 1000000);
    printf("#tries: %d\n", cmp);
    printf("=> efficiency: %.f tries/s\n", (double)cmp / period);

    free((void *)crypted);
    crypted = NULL;

    return EXIT_SUCCESS;
}

/*
* - Systeme d'exploitation: MacOS
* Chiffrement possible avec crypt.h: Non
* Mot de passe testé: A$4c
* Temps (sans chiffrement): 0 min 0.126 sec
*
* - Systeme d'exploitation: Windows
* Chiffrement possible avec crypt.h: Oui
* Mot de passe testé: A$4c
* Temps: 0 min 36.913 sec
*
* - Systeme d'exploitation: Ubuntu 
* Chiffrement possible avec crypt.h: Oui
* Mot de passe testé: A$4c
* Temps: 0 min 44.364 sec
*/