#include "csv.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>

csv_cont create_csv_cont(const char* filename, const char* format_str, size_t num_columns) {
    csv_cont c = malloc(sizeof(csv_cont_t));
    if(!c) err(1, "Memory error while allocating csv controller\n");
    c->filename = malloc(sizeof(char) * (strlen(filename) + 1));
    c->format_str = malloc(sizeof(char) * (strlen(format_str) + 1));
    if(!(c->filename || c->format_str)) err(1, "Memory error while allocating csv strings\n");
    memcpy(c->filename, filename, sizeof(char) * (strlen(filename) + 1));
    memcpy(c->format_str, format_str, sizeof(char) * (strlen(format_str) + 1));
    c->num_columns = num_columns;
    return c;
}

void destroy_csv_cont(csv_cont cont) {
    if(cont) {
        free(cont->filename);
        free(cont->format_str);
        free(cont);
    }
}

void initialize_file(csv_cont cont, ...) {
    va_list vargs;
    va_start(vargs, cont);

    FILE* fp = fopen(cont->filename, "w+");

    char* fstr = malloc(sizeof(char) * ((5 * cont->num_columns) + 1));
    if(!fstr) err(1, "Memory error while allocating header row for csv\n");

    for(size_t a = 0; a < (5 * cont->num_columns);) {
        fstr[a++] = '"';
        fstr[a++] = '%';
        fstr[a++] = 's';
        fstr[a++] = '"';
        fstr[a++] = (a < ((5 * cont->num_columns) - 1)) ? ',' : '\n';
    }
    fstr[(5 * cont->num_columns)] = 0;

    printf("Saving string %s\n", fstr);
    vfprintf(fp, fstr, vargs);

    fclose(fp);
    free(fstr);
    va_end(vargs);
}

void append_data(csv_cont cont, ...) {
    va_list vargs;
    va_start(vargs, cont);

    FILE* fp = fopen(cont->filename, "a+");

    vfprintf(fp, cont->format_str, vargs);

    fclose(fp);
    va_end(vargs);
}