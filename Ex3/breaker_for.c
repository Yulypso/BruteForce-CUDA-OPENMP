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




int search_all_1( char* crypted, int length, int first_char, int last_char ){
	int loop_size = last_char - first_char;
	int cryptlen = strlen(crypted);
	int max_iter = powl(loop_size, length);
    char tab[length];
    tab[length-1]='\0';
    char tabTmp[length];
    tabTmp[length-1]='\0';
	for(int j=0; j<length; j++) tab[j] = first_char;

	struct crypt_data data;

	int ret = 0;
    int i = 0, j = 0;
    char *cryptr;
	printf("max_iter = %lu \n", (unsigned long) max_iter);

    #pragma omp parallel firstprivate(data, i, j, tabTmp) private(cryptr) shared(max_iter, length, tab, cryptlen, crypted, first_char, last_char, ret) default(none)
    {
        #pragma omp for
        for (i = 0; i < max_iter; i++) {

            #pragma omp atomic write
            cryptr = crypt_r(tab, "salt", &data);

            if (!strncmp(crypted, cryptr, cryptlen))
            {
                strcpy(tabTmp, tab);
                for (int k = first_char; k < last_char; ++k) {
                    tabTmp[0] = k;
                    if(!strncmp(crypted, crypt_r(tabTmp, "salt", &data), cryptlen)) {
                        printf("%d: password found: %s\n", omp_get_thread_num(), tabTmp);
                    }
                }
                ret = i;
            }

            if(ret != 0)
            {
                #pragma omp cancel for
            }

            #pragma omp atomic update
            tab[0]++;

            #pragma omp cancellation point for

            for (j = 0; j < length - 1; j++) {
                if (last_char == tab[j]) {
                    #pragma omp atomic write
                    tab[j] = first_char;
                    #pragma omp atomic
                    tab[j + 1]++;
                }
            }
        }
        #pragma omp cancel parallel
    }
	return ret;
}


int main( int argc, char** argv ) {
	char* password; 
	struct timeval t1;
	struct timeval t2; 
	int first_char, last_char;
	int cmp;

	if( argc == 1 ) {
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
	} else if( argc == 4 ) {
		password = argv[1];
		first_char = atoi( argv[2] );
		last_char = atoi( argv[3] );
	} else {
		printf("usage: breaker <password> <first_ch> <last_ch>\n");
		printf("default: breaker A$4c 32 126\n");
		printf("exemple to break the binary password 1101000:\n");
		printf( "breaker 1101000 48 49\n" );
		exit( 0 );
	}
	char* crypted0 = crypt( password, "salt" );
	char* crypted = (char*) malloc( (strlen(crypted0)+1)*sizeof(char) );
	strcpy( crypted, crypted0 );

	printf( "*running parameters*\n" );
	printf( " -password length:\t%d digits\n", strlen(password) );
	printf( " -digits:\t\tfrom -%c- to -%c-\n", first_char, last_char );
	printf(	" -crypted to break:\t%s\n", crypted );

	gettimeofday(&t1, NULL);
	cmp = search_all_1( crypted, strlen( password ), first_char, last_char );
	gettimeofday(&t2, NULL);

	double period =(double)((int)(t2.tv_sec-t1.tv_sec))+((double)(t2.tv_usec-t1.tv_usec))/1000000;  

	printf( "time: %dmin %.3fs \n", (int)((t2.tv_sec-t1.tv_sec))/60, (double)((int)(t2.tv_sec-t1.tv_sec)%60)+((double)(t2.tv_usec-t1.tv_usec))/1000000 );
	printf( "#tries: %d\n", cmp );
	printf( "=> efficiency: %.f tries/s\n", (double)cmp/period );

	return EXIT_SUCCESS;
}
