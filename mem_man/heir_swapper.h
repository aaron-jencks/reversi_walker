#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct __bin_dict_t {
    __uint128_t** keys;
    size_t* indices;
    uint8_t*** bins;        // Cointains pointers to thye bins, whether they are on disk, or in memory
    uint8_t*** mappings;    // Contains pointers to the bins, that map to the pointers in the bin.
    size_t** usage_counters;
    uint8_t** allocation_flags;
    size_t* bin_sizes;
    size_t bin_count;
    size_t bin_size;
    size_t element_size;
    size_t loaded_count;
    size_t element_count;
} bin_dict_t;

typedef bin_dict_t* bin_dict;

bin_dict create_bin_dict(size_t num_bins, size_t bin_size, size_t element_size);
void destroy_bin_dict(bin_dict d);

uint8_t* bin_dict_get(bin_dict d, __uint128_t k);
uint8_t* bin_dict_put(bin_dict d, __uint128_t k, uint8_t* ptr);
double bin_dict_load_factor(bin_dict d);
