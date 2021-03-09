#pragma once

#include <stdint.h>
#include <stddef.h>
#include "tarraylist.hpp"
#include "dict_def.h"

typedef struct _fdict_t {
    Arraylist<dict_usage_pair_t>* bins;
    size_t bin_count;
    size_t max_element_count;
    size_t size;
    size_t flush_count;
} fdict_t;

typedef fdict_t* fdict;

fdict create_fixed_size_dictionary(size_t max_element_count, size_t flush_count);
void destroy_fixed_dictionary(fdict d);

void fdict_flush(fdict d);
dict_element_t* fdict_get_all(fdict d);

uint8_t* fdict_get(fdict d, __uint128_t k);
void fdict_put(fdict d, __uint128_t k, uint8_t* value);

double fdict_load_factor(fdict d);