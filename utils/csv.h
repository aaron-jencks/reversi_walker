#pragma once

#include <stdarg.h>
#include <stddef.h>

typedef struct _csv_cont_t {
    char* filename;
    size_t num_columns;
    char* format_str;
} csv_cont_t;

typedef csv_cont_t* csv_cont;

#ifdef __cplusplus
extern "C" {
#endif

csv_cont create_csv_cont(const char* filename, const char* format_str, size_t num_columns);
void destroy_csv_cont(csv_cont cont);
void initialize_file(csv_cont cont, ...);
void append_data(csv_cont cont, ...);

#ifdef __cplusplus
}
#endif