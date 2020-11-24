#pragma once

#include <stddef.h>
#include <stdint.h>

#pragma region hashtable mempage

typedef struct _mempage_str {
    __uint128_t*** pages;
    size_t** bin_counts;
    size_t* access_counts;
    uint8_t* page_present;
    size_t page_count;
    size_t count_per_page;
    __uint128_t num_bins;
    size_t bin_size;
    char* swap_directory;
} mempage_str;

typedef mempage_str* mempage;

mempage create_mempage(size_t page_max, __uint128_t bin_count, size_t bin_size);

void destroy_mempage(mempage mp);

// __uint128_t mempage_get(mempage mp, __uint128_t key_index);

// void mempage_put(mempage mp, __uint128_t index, void* data);

void mempage_append_bin(mempage mp, __uint128_t bin_index, __uint128_t value);
uint8_t mempage_value_in_bin(mempage mp, __uint128_t bin_index, __uint128_t value);
void mempage_clear_all(mempage mp);
void mempage_realloc(mempage mp, __uint128_t bin_count);

#pragma endregion
#pragma region hashtable rehash buffer

typedef struct _mempage_buff_str {
    __uint128_t** pages;
    size_t page_count;
    size_t count_per_page;
    __uint128_t num_element;
} mempage_buff_str;

typedef mempage_buff_str* mempage_buff;

mempage_buff create_mempage_buff(__uint128_t num_elements, size_t page_size);
void destroy_mempage_buff(mempage_buff buff);

void mempage_buff_put(mempage_buff buff, __uint128_t index, __uint128_t value);
__uint128_t mempage_buff_get(mempage_buff buff, __uint128_t index);

#pragma endregion