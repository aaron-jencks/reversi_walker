#pragma once

#include "dict_def.h"

#include <stddef.h>
#include <stdint.h>

#pragma region hashtable dmempage

typedef struct {
    dict_pair_t*** pages;
    size_t** bin_counts;
    size_t page_count;
    size_t count_per_page;
    __uint128_t num_bins;
    size_t bin_size;
} dmempage_str;

typedef dmempage_str* dmempage;

#ifdef __cplusplus
extern "C" {
#endif

dmempage create_dmempage(size_t page_max, __uint128_t bin_count, size_t bin_size);

void destroy_dmempage(dmempage mp);

// __uint128_t dmempage_get(dmempage mp, __uint128_t key_index);

// void dmempage_put(dmempage mp, __uint128_t index, void* data);

void dmempage_append_bin(dmempage mp, __uint128_t bin_index, dict_pair_t value);
void dmempage_remove(dmempage mp, __uint128_t bin_index, __uint128_t value);
uint8_t dmempage_value_in_bin(dmempage mp, __uint128_t bin_index, __uint128_t value);
uint8_t* dmempage_get(dmempage mp, __uint128_t bin_index, __uint128_t key);
void dmempage_clear_all(dmempage mp);
void dmempage_realloc(dmempage mp, __uint128_t bin_count);

#ifdef __cplusplus
}
#endif

#pragma endregion
#pragma region hashtable rehash buffer

typedef struct {
    dict_pair_t** pages;
    size_t page_count;
    size_t count_per_page;
    __uint128_t num_element;
} dmempage_buff_str;

typedef dmempage_buff_str* dmempage_buff;

#ifdef __cplusplus
extern "C" {
#endif

dmempage_buff create_dmempage_buff(__uint128_t num_elements, size_t page_size);
void destroy_dmempage_buff(dmempage_buff buff);

void dmempage_buff_put(dmempage_buff buff, __uint128_t index, dict_pair_t value);
dict_pair_t dmempage_buff_get(dmempage_buff buff, __uint128_t index);

#pragma endregion

#ifdef __cplusplus
}
#endif