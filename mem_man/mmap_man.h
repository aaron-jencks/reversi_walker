#pragma once

#include <stdint.h>
#include <stddef.h>

/*
 * We need a linked list that is specific to this manager
 * We need a manager that acts as a heap and updates the linked list parents upon percolation/bubbling heapify
 * The mmap size will be determined elsewhere, so will the bin size of the ll
 * 
 * But only do that if there isn't room in the page for the new bin extension
 * Find the largest bin id in the current page, then move it (and all of it's linked children) to a new page, and place the new linked bin into it's slot
 */

// typedef struct __mmap_bin_str {
//     uint8_t usage_flag;
//     __uint128_t bid;
//     __uint128_t* elements;  // null-terminated
//     mmap_bin_str* link;
// } mmap_bin_str;

// typedef mmap_bin_str* mmap_bin;

// mmap_bin create_mmap_bin(uint8_t in_use, __uint128_t bid, size_t num_elements);
// void destroy_mmap_bin(mmap_bin bin);

typedef struct __mmap_page_str {
    uint8_t* map;
    uint8_t* free_pointer;
    size_t size;
    char* filename;
    int fd;
} mmap_page_str;

typedef mmap_page_str* mmap_page;

mmap_page create_mmap_page(const char* filename, size_t size);
void destroy_mmap_page(mmap_page page, size_t size);

typedef struct __mmap_man_str {
    mmap_page* pages;
    size_t num_pages;
    size_t bins_per_page;    // for use in bins property
    size_t elements_per_bin; // how many elements can be in a bin before it's extended
    size_t max_page_size;    // in bytes
    char* file_directory;
} mmap_man_str;

typedef mmap_man_str* mmap_man;

mmap_man create_mmap_man(size_t page_size, size_t bin_size);
void destroy_mmap_man(mmap_man man);

size_t find_mmap_bin_total_size(size_t initial_bin_size);

/**
 * @brief We need a way to allocate new arrays if they haven't been allocated yet
 * 
 * @param man 
 * @return uint8_t*
 */
uint8_t* mmap_allocate_bin(mmap_man man);

// /**
//  * @brief We need a way to insert new elements
//  * 
//  * @param man 
//  * @param bin_index 
//  * @param value 
//  */
// void mmap_man_append_bin(mmap_man man, __uint128_t bin_index, __uint128_t value);

// /**
//  * @brief We need a way to extend previously allocated bins
//  * 
//  * @param man 
//  * @param bin_index 
//  * @return mmap_bin 
//  */
// mmap_bin mmap_man_generate_bin_extension(mmap_man man, __uint128_t bin_index);

// /**
//  * @brief We need a way to move previously allocated bins to make room for bin extensions
//  * 
//  * We assume that the bin is already in the page that we don't want it in anymore.
//  * 
//  * @param man 
//  * @param bin_index 
//  */
// void mmap_man_migrate_bin(mmap_man man, __uint128_t bin_index);
