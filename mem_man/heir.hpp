#pragma once

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#include "mmap_man.h"
#include "../utils/dictionary/dict_def.h"
#include "../utils/dictionary/fdict.hpp"
#include "../utils/dictionary/hdict.hpp"

typedef struct __heirarchy_str {
    // void** first_level;
    mmap_man final_level;
    fdict fixed_cache;
    hdict rehashing_cache;
    // size_t num_bits_per_level;
    size_t num_bits_per_final_level;
    // size_t num_levels;
    size_t page_size;
    size_t collision_count;
} heirarchy_str;

typedef heirarchy_str* heirarchy;

extern pthread_mutex_t heirarchy_lock;

heirarchy create_heirarchy(char* file_directory);
void destroy_heirarchy(heirarchy h);

uint8_t heirarchy_insert(heirarchy h, __uint128_t key);

void to_file_heir(FILE* fp, heirarchy h);

heirarchy from_file_heir(FILE* fp);
