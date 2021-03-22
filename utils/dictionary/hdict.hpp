#pragma once

#include <stdint.h>
#include <stddef.h>
#include "dict_def.h"
#include "dmempage.h"

typedef struct _hdict_t {
    dmempage bins;
    __uint128_t bin_count;
    __uint128_t size;
} hdict_t;

typedef hdict_t* hdict;

hdict create_rehashing_dictionary(size_t bin_count, size_t initial_bin_size);
void destroy_rehashing_dictionary(hdict d);

dmempage_buff hdict_get_all(hdict d);

uint8_t* hdict_get(hdict d, __uint128_t k);
void hdict_put(hdict d, __uint128_t k, uint8_t* value);

double hdict_load_factor(hdict d);