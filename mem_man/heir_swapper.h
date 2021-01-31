#pragma once

#include "./heir.h"

#include <stdint.h>

typedef struct __bin_dict_t {
    __uint128_t** keys;
    size_t indices;
    uint8_t*** bins;
    uint8_t*** mappings;
    size_t bin_count;
} bin_dict_t;

typedef bin_dict_t* bin_dict;

bin_dict create_bin_dict(size_t num_bins);
void destroy_bin_dict(bin_dict d);

uint8_t* bin_dict_get(bin_dict d, __uint128_t k);
void bin_dict_put(bin_dict d, __uint128_t k);