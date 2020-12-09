#pragma once

#include "arraylist.h"

#include <stdint.h>

typedef struct _cache_index_str {
    uint8_t player;
    // uint64_t upper;
    // uint64_t lower;
    __uint128_t index;
} cache_index_str;

/**
 * @brief Represents a bit index in the 3D bit cache
 * 
 */
typedef cache_index_str* cache_index;

typedef struct _bit_cache_str {
    arraylist bits;
    uint64_t buffer_size;
} bit_cache_str;

/**
 * @brief Represents the 3D bit cache
 * 
 */
typedef bit_cache_str* bit_cache;

/**
 * @brief Create a bit cache object
 * 
 * @param player_count The number of players in the cache
 * @param upper_count The number of initial bins for the upper half of the board (bytes 0-n/2)
 * @param lower_count The number of initial bits for the lower half of the board (bytes n/2-n)
 * @param buffer_size The number of extra bits to pad each array with when reallocating
 * @return bit_cache 
 */
bit_cache create_bit_cache(uint8_t player_count, uint64_t buffer_size);

/**
 * @brief Destroys a bit cache object
 * 
 * @param c 
 */
void destroy_bit_cache(bit_cache c);

/**
 * @brief Attempts to insert a bit into the cache, if it already exists, 
 * then the insertion failes and the function returns 0
 * 
 * @param cache The cache to inser teh bit into
 * @param index The index to insert the bit at
 * @return uint8_t Returns 1 if the bit was inserted, and 0 otherwise
 */
uint8_t conditional_insert_bit(bit_cache cache, cache_index index);