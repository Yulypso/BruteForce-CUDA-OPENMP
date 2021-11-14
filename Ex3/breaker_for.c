//gcc breaker.c -o breaker -lcrypt && ./breaker


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#ifdef __APPLE__
    #include <unistd.h>
#else
    #include <crypt.h>
#endif

unsigned long cmp = 0;

int carry( char* tab, int i, int first_char, int last_char ) {
	if( i==0 )
		return 0;
	else
		if( tab[i-1] < last_char ) {
			tab[i-1] = tab[i-1]+1;
			return 1;
		} else {
			tab[i-1] = first_char;
			return carry( tab, i-1, first_char, last_char );
		}
}

int search_one(char* crypted, char* tab)
{
	cmp++;
    //printf("%d: %s : %s : %s\n", omp_get_thread_num(), crypted, crypt(tab, "pepper"), tab);
	if(!strcmp(crypted, crypt(tab, "pepper"))) {
		return 1;
	} else {
        return 0;
    }
}

void checkAdd(char *tab_tmp, int length, int first_char, int last_char, int position)
{
    if(tab_tmp[position] != last_char)
        tab_tmp[position] += 1;
    else
    {
        tab_tmp[position] = first_char;
        checkAdd(tab_tmp, length, first_char, last_char, position-1);
    }
}

int checkFull(char *tab_tmp, int length, int last_char)
{
    for (int i = 1; i < length; ++i) {
        if (tab_tmp[i] != last_char) {
            return 0;
        }
    }
    return 1;
}

void search_all( char* crypted, int length, int first_char, int last_char ){
	int i, j, k, l, decalage, loop = 1;
    char *tab = (char*)malloc((length+1)*sizeof(char));
    for(i=0; i<length; i++) tab[i] = first_char;
    tab[length] = '\0';

    char *tab_tmp = (char*)malloc((length+1)*sizeof(char));
    for(i=0; i<length; i++) tab_tmp[i] = first_char;
    tab_tmp[length] = '\0';

    printf( "let's break it...\n" );
	loop = !search_one( crypted, tab );

    /*while (loop) {
        if (tab[i] < last_char) {
            tab[i] = tab[i] + 1;
            loop = !search_one(crypted, tab);
        } else {
            tab[i] = first_char;
            if ((loop = carry(tab, i, first_char, last_char)))
                loop = !search_one(crypted, tab);
        }
    }*/

    i = 0; j = 0, k = 0, l = 0, decalage = 0;
    int isfound = 0;
    char pass[99];
    #pragma omp parallel for schedule(dynamic) firstprivate(j, k, l, tab_tmp, decalage) shared(tab, length, first_char, last_char, crypted, pass, isfound) default(none)
    {
        for (i = first_char; i < last_char + 1; i++) {
            tab_tmp[0] = i;

            do {
                checkAdd(tab_tmp, length, first_char, last_char, length - 1);
                printf("%d : %s\n", omp_get_thread_num(), tab_tmp);

                #pragma omp critical
                {
                    if (search_one(crypted, tab_tmp)) {
                        strcpy(pass, tab_tmp);
                        printf("password found: %s\n", pass);
                    }
                }
            } while (!checkFull(tab_tmp, length, last_char));
            strcpy(tab_tmp, tab);
        }
    }
    printf("password found: %s\n", pass);
}



int main( int argc, char** argv ) {
	char* password; 
	int first_char, last_char;
	float t1, t2; 

	if( argc == 1 ) {
		password = "ZZZZ";
		first_char = 65; //32 132
		last_char = 90;
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
	char* crypted0 = crypt( password, "pepper" );
	char* crypted = (char*) malloc( (strlen(crypted0)+1)*sizeof(char) );
	strcpy( crypted, crypted0 );

	printf( "*running parameters*\n" );
	printf( " -password length:\t%lu digits\n", strlen(password) );
	printf( " -digits:\t\tfrom -%c- to -%c-\n", first_char, last_char );
	printf(	" -crypted to break:\t%s\n", crypted );

	t1 = clock();
	search_all( crypted, strlen( password ), first_char, last_char );
	t2 = clock();

	float period = (t2-t1)/CLOCKS_PER_SEC;
	if( period < 60 )
		printf( "time: %.1f s \n", period );
	else
		printf( "time: %.1f min \n", period/60 );
	printf( "#tries: %lu\n", cmp );
	printf( "=> efficiency: %.f tries/s\n", (float)cmp/period );

	return EXIT_SUCCESS;
}
