#pragma once

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#include "mmap_man.h"
#include "../utils/dictionary/dict_def.h"
#include "../utils/dictionary/fdict.hpp"
#include "../utils/dictionary/hdict.hpp"
#include "../utils/semaphore.hpp"

typedef struct __heirarchy_str {
    // void** first_level;
    mmap_man final_level;
    fdict fixed_cache;
    hdict rehashing_cache;
    fdict temp_board_cache;
    // size_t num_bits_per_level;
    size_t num_bits_per_final_level;
    // size_t num_levels;
    size_t page_size;
    size_t collision_count;
    size_t highest_complete_level;
    Arraylist<__uint128_t>** level_mappings;
    MultiReadSemaphore* sem;
    MultiReadSemaphore* csem;
} heirarchy_str;

typedef heirarchy_str* heirarchy;

extern pthread_mutex_t heirarchy_lock;
extern pthread_mutex_t heirarchy_cache_lock;

heirarchy create_heirarchy(char* file_directory);
void destroy_heirarchy(heirarchy h);

uint8_t heirarchy_insert(heirarchy h, __uint128_t key, size_t level);
uint8_t* heirarchy_insert_all(heirarchy h, __uint128_t* keys, size_t* levels, size_t n);
uint8_t heirarchy_insert_cache(heirarchy h, __uint128_t key);

void heirarchy_purge_level(heirarchy h, size_t level);

void to_file_heir(FILE* fp, heirarchy h);

heirarchy from_file_heir(FILE* fp);
