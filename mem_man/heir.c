#include "heir.h"
#include "../utils/arraylist.h"

#include <err.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

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

    h->num_bits_per_level = --shifts;
    h->num_levels = 128 / shifts;
    h->num_bits_per_final_level = shifts + (128 % shifts); // To ensure that we use every bit

    h->final_level = create_mmap_man(INITIAL_PAGE_SIZE, 1 << (h->num_bits_per_final_level - 3));

    h->first_level = calloc(h->page_size, sizeof(void*));
    if(!h->first_level) err(1, "Memory error while allocating heirarchical memory system\n");

    return h;
}

void destroy_heirarchy(heirarchy h) {
    if(h) {
        destroy_mmap_man(h->final_level);

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
            if(d < h->num_levels && k) {
                for(size_t i = 0; i < h->page_size; i++) {
                    append_pal(stack, k[i]);
                    append_dal(dstack, d + 1);
                }
                free(k);
            }
        }

        free(h);
    }
}

uint8_t heirarchy_insert(heirarchy h, __uint128_t key) {
    __uint128_t key_copy = key, bit_placeholder = 0; 
    size_t bits, level = 1;
    void** phase = h->first_level;

    // Create the placeholder to extract the correct number of bits every time
    bit_placeholder = (1 << h->num_bits_per_level) - 1;

    // Traverse through all of the levels
    for(level = 1; level < h->num_levels; level++) {
        bits = (size_t)(key_copy & bit_placeholder);
        key_copy = key_copy >> h->num_bits_per_level;

        if(!phase[bits]) {
            phase[bits] = calloc(h->page_size, sizeof(void*));
            if(!phase[bits]) err(1, "Memory Error while allocating bin for key hash\n");
        }

        if(level == (h->num_levels - 1) && !phase[bits]) {
            // Allocate a new bin
            phase[bits] = mmap_allocate_bin(h->final_level);
            for(size_t b = 0; b < h->final_level->max_page_size; b++) ((uint8_t*)(phase[bits]))[b] = 0;
        }
        
        phase = (void**)phase[bits];
    }

    // Traverse the last level
    // Use 3 bits to determine the bit
    // Use the rest to determine the byte

    // Create the placeholder to extract the correct number of bits every time, for the edge case of the final level, where the least 3 bits are used for bit position
    bit_placeholder = (1 << (h->num_bits_per_final_level - 2)) - 1;

    // Extract the bit from the last level
    bits = (size_t)(key_copy >> 3);
    uint8_t bit = key_copy & 7;

    // Insert the new bit if it's not already in the array
    uint8_t* bytes = (uint8_t*)phase[bits];
    uint8_t ph = 1 << bit;

    if(bytes[bits] & ph) return 0;
    bytes[bits] |= ph;
    return 1;
}

void to_file_heir(FILE* fp, heirarchy h) {
    // Traverse through all the layers and number each array with an id,
    // Then we need to traverse backwards through the layers and insert them into the file.
    // Ideally we could use DFS
    // File Format:
    // [level][array id][contents]...0
    // [bits per level][num levels][page size][reverse order traversal]...

    if(h) {
        fwrite(&h->num_bits_per_level, sizeof(h->num_bits_per_level), 1, fp);
        fwrite(&h->num_levels, sizeof(h->num_levels), 1, fp);
        fwrite(&h->page_size, sizeof(h->page_size), 1, fp);

        size_t* level_ids = calloc(h->num_levels + 1, sizeof(size_t));
        if(!level_ids) err(1, "Memory Error while allocating array id array for heirarchy file saving algorithm\n");

        typedef struct __array_file_cluster_t {
            void** ptr;
            size_t id;
            size_t level;
        } array_file_cluster_t;

        // These capacity numbers are purposely low to reduce memory consumption
        ptr_arraylist q = create_ptr_arraylist(h->num_levels * h->page_size), 
                      qq = create_ptr_arraylist(h->num_levels * h->page_size),
                      qb = create_ptr_arraylist(h->num_levels * h->page_size);

        array_file_cluster_t* v = malloc(sizeof(array_file_cluster_t)), *vc;
        if(!v) err(1, "Memory error while allocating page information for array\n");
        v->ptr = h->first_level;
        v->id = level_ids[0]++;
        v->level = 0;

        append_pal(qq, v);

        // Perform BFS and migrate all of the values into q
        while(qq->pointer) {
            v = pop_front_pal(qq);

            uint16_t* contents = malloc(sizeof(uint16_t) * h->page_size);
            if(!contents) err(1, "Memory error while allocating id array for array\n");

            size_t vc_level = v->level + 1;

            for(size_t c = 0; c < h->page_size; c++) {
                if(v->ptr[c]) {
                    vc = malloc(sizeof(array_file_cluster_t));
                    if(!vc) err(1, "Memory error while allocating page information for array\n");
                    vc->ptr = (void**)(v->ptr[c]);
                    vc->id = level_ids[vc_level]++;
                    vc->level = vc_level;

                    if(vc_level < h->num_levels) append_pal(qq, vc);
                    else {
                        // We are at the bit level
                        append_pal(qb, vc);
                    }

                    contents[c] = vc->id;
                }
                else contents[c] = 65535;
            }

            v->ptr = contents;

            append_pal(q, v);
        }

        // So that I know how many arrays are in each level
        fwrite(level_ids, sizeof(size_t), h->num_levels + 1, fp);

        // Write the bit results first
        size_t bytes_per_final_level = 1 << (h->num_bits_per_level - 3);
        while(qb->pointer) {
            v = pop_back_pal(qb);
            uint8_t* bytes = (uint8_t*)v->ptr;

            #ifdef heirdebug
                printf("Saving array (address: %p, level: %lu, id: %lu) to file\nContents: [", bytes, v->level, v->id);
                for(size_t b = 0; b < bytes_per_final_level; b++) {
                    if(b) printf(", ");
                    printf("%u", bytes[b]);
                }
                printf("]\n");
            #endif

            fwrite(&v->level, sizeof(v->level), 1, fp);
            fwrite(&v->id, sizeof(v->id), 1, fp);
            fwrite(bytes, sizeof(uint8_t), bytes_per_final_level, fp);

            // free((uint16_t*)bytes); Don't free ranges of the file
            free(v);
        }

        // Reverse q and place the elements into the file
        while(q->pointer) {
            v = pop_back_pal(q);
            fwrite(&v->level, sizeof(v->level), 1, fp);
            fwrite(&v->id, sizeof(v->id), 1, fp);
            fwrite(v->ptr, sizeof(uint16_t), h->page_size, fp);

            free(v->ptr);
            free(v);
        }

        free(level_ids);
        destroy_ptr_arraylist(q);
        destroy_ptr_arraylist(qq);
        destroy_ptr_arraylist(qb);
    }
}

heirarchy from_file_heir(FILE* fp) {
    // File Format:
    // [level][array id][contents]...0
    // [bits per level][num levels][page size][reverse order traversal]...

    heirarchy h = malloc(sizeof(heirarchy_str));
    if(!h) err(1, "Memory error while allocating new heirarchy for file read\n");

    fread(&h->num_bits_per_level, sizeof(h->num_bits_per_level), 1, fp);
    fread(&h->num_levels, sizeof(h->num_levels), 1, fp);
    fread(&h->page_size, sizeof(h->page_size), 1, fp);

    size_t *level_counts  = malloc(sizeof(size_t) * (h->num_levels + 1));
    if(!level_counts) err(1, "Memory error while allocating level counts for heirarchy file read\n");
    fread(level_counts, sizeof(size_t), h->num_levels + 1, fp);

    h->final_level = create_mmap_man(INITIAL_PAGE_SIZE, 1 << (h->num_bits_per_level - 3));

    size_t level, id;
    uint8_t* mmap_ptr;
    uint16_t* arr_contents = malloc(sizeof(uint16_t) * h->page_size), arr_element;
    if(!arr_contents) err(1, "Memory error while allocating contents for array in heirarchy file read\n");
    void** ptr;

    // Generate a lookup table to store mappings from ids to pointer values
    void*** ptr_mappings = malloc(sizeof(void**) * (h->num_levels + 1));
    if(!ptr_mappings) err(1, "Memory error while allocating ptr mapping array for heirarchy file read\n");
    for(size_t m = 0; m < (h->num_levels + 1); m++) {
        ptr_mappings[m] = calloc(level_counts[m], sizeof(void*));
        if(!ptr_mappings[m]) err(1, "Memory error while allocating ptr mapping array for heirarchy file read\n");
    }

    while(1) {
        fread(&level, sizeof(size_t), 1, fp);
        if(level < h->num_levels) {
            // It's a normal array
            fread(&id, sizeof(size_t), 1, fp);
            fread(arr_contents, sizeof(uint16_t), h->page_size, fp);

            ptr = malloc(sizeof(void*) * h->page_size);
            if(!ptr) err(1, "Memory error while allocating array in heirarchy file read\n");
            
            for(size_t c = 0; c < h->page_size; c++) ptr[c] = (arr_contents[c] != 65535) ? ptr_mappings[level + 1][arr_contents[c]] : 0; 
            ptr_mappings[level][id] = ptr;

            if(!level) {
                // We've read in the root array
                h->first_level = ptr;
                break;
            }
        }
        else {
            // It's part of the bit string
            mmap_ptr = mmap_allocate_bin(h->final_level);

            fread(&id, sizeof(size_t), 1, fp);
            fread(mmap_ptr, sizeof(uint8_t), 1 << (h->num_bits_per_level - 3), fp);

            ptr_mappings[level][id] = mmap_ptr;
        }
    }

    return h;
}