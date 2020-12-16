#include "heir.h"
#include "../utils/arraylist.h"

#include <err.h>
#include <unistd.h>
#include <stdlib.h>

#define INITIAL_CACHE_SIZE 34359738368
#define INITIAL_PAGE_SIZE 5368709120

heirarchy create_heirarchy() {
    heirarchy h = malloc(sizeof(heirarchy_str));
    if(!h) err(1, "Memory error while allocating heirarchical memory system\n");

    // The number of pointers that will fit in one page of memory
    h->page_size = getpagesize() / sizeof(void*);   // This is why you don't have to use the full 12 bits, 4096 / 8 bytes, gives you something less

    // Find the minimum number of levels
    __uint128_t b;
    for(b = 1; b < h->page_size; b = b << 1);
    size_t shifts = 0;
    while(b) {
        shifts++;
        b = b >> 1;
    }

    h->num_bits_per_level = shifts;
    h->num_levels = 128 / shifts;

    h->final_level = create_mmap_man(INITIAL_PAGE_SIZE, 1 << (shifts - 3));

    h->first_level = calloc(h->page_size, sizeof(void*));
    if(!h->first_level) err(1, "Memory error while allocating heirarchical memory system\n");

    return h;
}

void destroy_heirarchy(heirarchy h) {
    if(h) {
        destroy_bit_mempage(h->final_level);

        // free the first and all following levels, use DFS?
        ptr_arraylist stack = create_ptr_arraylist(h->page_size * h->num_levels + 1);
        uint64_arraylist dstack = create_uint64_arraylist(h->page_size * h->num_levels + 1);

        // Queue up the first level
        for(size_t i = 0; i < h->page_size; i++) {
            append_pal(stack, h->first_level[i]);
            append_dal(dstack, 1);
        }
        free(h->first_level);

        // Perform DFS
        while(stack->pointer) {
            void** k = (void**)pop_back_pal(stack);
            uint64_t d = pop_back_dal(dstack);
            if(d < h->num_levels && k) 
                for(size_t i = 0; i < h->page_size; i++) {
                    append_pal(stack, k[i]);
                    append_dal(dstack, d + 1);
                }
            free(k);
        }

        free(h);
    }
}

void heirarchy_insert(heirarchy h, __uint128_t key) {
    __uint128_t key_copy = key, bit_placeholder = 0; 
    size_t bits, level = 1;
    void** phase = h->first_level;

    // Create the placeholder to extract the correct number of bits every time
    for(size_t b = 0; b < h->num_bits_per_level; b++) {
        bit_placeholder = bit_placeholder << 1;
        bit_placeholder++;
    }

    // Traverse through all of the levels
    for(level = 1; level < h->num_levels; level++) {
        bits = (size_t)(key_copy & bit_placeholder);
        key_copy = key_copy >> h->num_bits_per_level;
        phase = (void**)phase[bits];
    }

    // Traverse the last level
    // Use 3 bits to determine the bit
    // Use the rest to determine the byte

    // Create the placeholder to extract the correct number of bits every time, for the edge case of the final level, where the least 3 bits are used for bit position
    bit_placeholder = 0;
    for(size_t b = 0; b < (h->num_bits_per_level - 3); b++) {
        bit_placeholder = bit_placeholder << 1;
        bit_placeholder++;
    }

    // Extract the bit from the last level
    size_t bits = (size_t)((key_copy >> 3) & bit_placeholder);
    uint8_t bit = bit_placeholder & 7, byte;

    if(!(uint8_t*)phase[bits]) {
        // Allocate a new bin
        phase[bits] = mmap_allocate_bin(h->final_level);
    }

    // Insert the new bit
    uint8_t* bytes = (uint8_t*)phase[bits];
    byte = bytes[bits];
    uint8_t ph = 1 << bit;
    byte |= ph;
}