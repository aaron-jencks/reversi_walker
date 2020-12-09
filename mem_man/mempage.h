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
    size_t save_interv_counter;
} mempage_str;

typedef mempage_str* mempage;

mempage create_mempage(size_t page_max, __uint128_t bin_count, size_t bin_size);

void destroy_mempage(mempage mp);

uint8_t mempage_page_exists(mempage mp, size_t page_index);
size_t mempage_find_least_used_page(mempage mp);
size_t mempage_find_total_size(mempage mp);

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
    uint8_t* page_present;
    char* swap_directory;
    size_t save_interv_counter;
} mempage_buff_str;

typedef mempage_buff_str* mempage_buff;

mempage_buff create_mempage_buff(__uint128_t num_elements, size_t page_size);
void destroy_mempage_buff(mempage_buff buff);

uint8_t mempage_buff_page_exists(mempage_buff mp, size_t page_index);
size_t mempage_buff_find_least_used_page(mempage_buff mp);
size_t mempage_buff_find_total_size(mempage_buff buff);

void mempage_buff_put(mempage_buff buff, __uint128_t index, __uint128_t value);
__uint128_t mempage_buff_get(mempage_buff buff, __uint128_t index);

#pragma endregion
#pragma region mmap mempages

#pragma region hashtable mempage with mmap

typedef struct _mmap_mempage_str {
    __uint128_t*** pages;
    size_t** bin_counts;
    size_t page_count;
    size_t size_per_page;
    __uint128_t num_bins;
    size_t bin_size;
    char* swap_directory;
    char** page_filenames;
} mmap_mempage_str;

typedef mmap_mempage_str* mmap_mempage;

/**
 * @brief Create a mmap mempage object
 * 
 * You need at least 2*page_max so that if you place all of the elements into one bin, then there is enough space to fill 
 * 
 * @param page_max 
 * @param bin_count 
 * @param bin_size 
 * @return mmap_mempage 
 */
mmap_mempage create_mmap_mempage(size_t page_max, __uint128_t bin_count, size_t bin_size);

void destroy_mmap_mempage(mmap_mempage mp);

void mmap_mempage_append_bin(mempage mp, __uint128_t bin_index, __uint128_t value);
uint8_t mmap_mempage_value_in_bin(mempage mp, __uint128_t bin_index, __uint128_t value);

void mmap_mempage_realloc(mempage mp, __uint128_t bin_count);

#pragma endregion
#pragma region hashtable rehash buffer with mmap

typedef struct _mempage_buff_str {
    __uint128_t** pages;
    size_t page_count;
    size_t count_per_page;
    __uint128_t num_element;
    uint8_t* page_present;
    char* swap_directory;
    size_t save_interv_counter;
} mempage_buff_str;

typedef mempage_buff_str* mempage_buff;

mempage_buff create_mempage_buff(__uint128_t num_elements, size_t page_size);
void destroy_mempage_buff(mempage_buff buff);

uint8_t mempage_buff_page_exists(mempage_buff mp, size_t page_index);
size_t mempage_buff_find_least_used_page(mempage_buff mp);
size_t mempage_buff_find_total_size(mempage_buff buff);

void mempage_buff_put(mempage_buff buff, __uint128_t index, __uint128_t value);
__uint128_t mempage_buff_get(mempage_buff buff, __uint128_t index);

#pragma endregion

#pragma endregion