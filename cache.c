#include "cache.h"

#include <stdlib.h>
#include <err.h>

/**
 * @brief Create a bit cache object
 * 
 * @param player_count The number of players in the cache
 * @param upper_count The number of initial bins for the upper half of the board (bytes 0-n/2)
 * @param lower_count The number of initial bits for the lower half of the board (bytes n/2-n)
 * @param buffer_size The number of extra bits to pad each array with when reallocating
 * @return bit_cache 
 */
bit_cache create_bit_cache(uint8_t player_count, uint64_t buffer_size) {
    bit_cache c = malloc(sizeof(bit_cache_str));
    if(!c) err(1, "Memory Error while allocating bit cache\n");

    c->buffer_size = buffer_size;

    // Allocate the bits
    c->bits = create_arraylist(player_count);
    for(uint8_t i = 0; i < player_count; i++) {
        append_al(c->bits, create_arraylist(buffer_size));
    }

    return c;
}

/**
 * @brief Destroys a bit cache object
 * 
 * @param c 
 */
void destroy_bit_cache(bit_cache c) {
    if(c) {
        // de-allocate the bits
        if(c->bits) {
            for(uint8_t i = 0; i < c->bits->size; i++) {
                for(uint64_t j = 0; j < ((arraylist)c->bits->data[i])->size; j++) 
                    destroy_arraylist(((arraylist)c->bits->data[i])->data[j]);
                destroy_arraylist(c->bits->data[i]);
            }
            destroy_arraylist(c->bits);
        }
        free(c);
    }
}

/**
 * @brief Attempts to insert a bit into the cache, if it already exists, 
 * then the insertion failes and the function returns 0
 * 
 * @param cache The cache to inser teh bit into
 * @param index The index to insert the bit at
 * @return uint8_t Returns 1 if the bit was inserted, and 0 otherwise
 */
uint8_t conditional_insert_bit(bit_cache cache, cache_index index) {
    uint64_t byte = (index->index) >> 3;
    uint8_t bit = byte % 8;

    // if(index->player >= cache->bits->size) {
    //     // Allocate more player space
    //     insert_al(cache->bits, index->player, create_arraylist(index->upper + 64));
    //     for(uint8_t i = 0; i < index->upper + 64; i++) append_al(cache->bits->data[index->player], create_arraylist(byte + 64)); 
    // }

    // arraylist d1page = (arraylist)cache->bits->data[index->player];

    // if(index->upper >= d1page->size)
    //     // Allocate more upper space
    //     insert_al(d1page, index->upper, create_arraylist(byte + 64));

    // arraylist d2page = (arraylist)d1page->data[index->upper];
    // uint64_t ptr = byte >> 3;

    // if(ptr >= d2page->size) {
    //     // Allocate more bit space
    //     insert_al(d2page, ptr, 0);
    // }

    // uint8_t* bytes = (uint8_t*)d2page->data;
    // uint8_t selector = 1 << bit;

    // if(!(bytes[byte] & selector)) {
    //     bytes[byte] |= selector;
    //     return 1;
    // }

    if(index->player >= cache->bits->size) {
        // Allocate more player space
        insert_al(cache->bits, index->player, create_arraylist((index->index >> 6) + cache->buffer_size));
    }

    arraylist d1page = (arraylist)cache->bits->data[index->player];
    uint64_t ptr = byte >> 3;

    if(ptr >= d1page->size) {
        // Allocate more bit space
        insert_al(d1page, ptr, 0);
    }

    uint8_t* bytes = (uint8_t*)d1page->data;
    uint8_t selector = ((uint8_t)1) << bit;

    if(!(bytes[byte] & selector)) {
        bytes[byte] |= selector;
        return 1;
    }

    return 0;
}