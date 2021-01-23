#include "heir.h"
#include "../utils/arraylist.h"

#include <err.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#define INITIAL_CACHE_SIZE 34359738368
#define INITIAL_PAGE_SIZE 5368709120

pthread_mutex_t heirarchy_lock;

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

    h->final_level = create_mmap_man(INITIAL_PAGE_SIZE, (1 << (h->num_bits_per_final_level - 3)) + sizeof(__uint128_t) * (h->num_bits_per_final_level - 3));

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
            if(d < (h->num_levels - 1) && k) {
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
    // #ifdef heirdebug
    //     printf("Inserting %lu %lu into the cache\n", ((uint64_t*)&key)[1], ((uint64_t*)&key)[0]);
    // #endif

    __uint128_t key_copy = key, bit_placeholder = 0; 
    size_t bits, level = 1;
    void** phase = h->first_level;

    // Create the placeholder to extract the correct number of bits every time
    bit_placeholder = ((1 << h->num_bits_per_level) - 1) << (128 - h->num_bits_per_level);

    // Traverse through all of the levels
    for(level = 1; level < h->num_levels; level++) {
        bits = (size_t)((key_copy & bit_placeholder) >> (128 - h->num_bits_per_level));
        key_copy = key_copy << h->num_bits_per_level;

        // #ifdef heirdebug
        //     printf("Bits for level %lu is %lu\n", level, bits);
        // #endif

        if(level < (h->num_levels - 1) && !phase[bits]) {
            // printf("Allocating %lu[%lu]\n", level, bits);
            phase[bits] = calloc(h->page_size, sizeof(void*));
            if(!phase[bits]) err(1, "Memory Error while allocating bin for key hash\n");
        }
        else if(!phase[bits]) {
            // Allocate a new bin
            phase[bits] = mmap_allocate_bin(h->final_level);

            // for(size_t b = 0; b < h->final_level->max_page_size; b++) ((uint8_t*)(phase[bits]))[b] = 0;
            ((__uint128_t*)phase[bits])[0] = key >> h->num_bits_per_final_level + 3;
            size_t num_jumps = h->final_level->elements_per_bin / 8;
            for(size_t b = 2; b < num_jumps; b++) ((uint64_t*)(phase[bits]))[b] = 0;
            for(size_t b = 0; b < h->final_level->elements_per_bin % 8; b++) ((uint8_t*)(phase[bits]))[num_jumps + b] = 0;
            
            #ifdef heirdebug
                printf("Generating a new bin for entry %lu[%lu] @ %p\n", level - 1, bits, phase[bits]);
            #endif
        }
        
        phase = (void**)phase[bits];
    }

    // Traverse the last level
    // Use 3 bits to determine the bit
    // Use the rest to determine the byte

    // Create the placeholder to extract the correct number of bits every time, for the edge case of the final level, where the least 3 bits are used for bit position
    // bit_placeholder = (1 << (h->num_bits_per_final_level - 2)) - 1;

    // Extract the bit from the last level
    key_copy = key_copy >> (128 - h->num_bits_per_final_level);
    bits = (size_t)(key_copy >> 3);
    uint8_t bit = key_copy & 7;

    // #ifdef heirdebug
    //     printf("Bits for final level is %lu, byte index is %u\n", bits, bit);
    // #endif

    // Insert the new bit if it's not already in the array
    // printf("Bit value: %lu\n", bits);
    uint8_t byte = ((uint8_t*)phase)[bits];
    uint8_t ph = 1 << bit;

    while(pthread_mutex_trylock(&heirarchy_lock)) sched_yield();

    if(byte & ph) {
        // #ifdef heirdebug
        //     printf("%lu %lu is already in the cache\n", ((uint64_t*)&key)[1], ((uint64_t*)&key)[0]);
        // #endif

        pthread_mutex_unlock(&heirarchy_lock);

        return 0;
    }

    // #ifdef heirdebug
    //     printf("%lu %lu inserted into the cache\n", ((uint64_t*)&key)[1], ((uint64_t*)&key)[0]);
    // #endif

    ((uint8_t*)phase)[bits] |= ph;

    pthread_mutex_unlock(&heirarchy_lock);

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
        if(!fwrite(&h->num_bits_per_level, sizeof(h->num_bits_per_level), 1, fp)) err(12, "Error while writing file data\n");
        fwrite(&h->num_levels, sizeof(h->num_levels), 1, fp);
        fwrite(&h->page_size, sizeof(h->page_size), 1, fp);

        printf("Heirarchy stats:\n\tBits: %lu\n\tLevels: %lu\n\tSize: %lu\n", h->num_bits_per_level, h->num_levels, h->page_size);

        size_t* level_ids = calloc(h->num_levels + 1, sizeof(size_t));
        if(!level_ids) err(1, "Memory Error while allocating array id array for heirarchy file saving algorithm\n");

        typedef struct __file_node_cluster_t {
            size_t level;
            size_t index;
            uint64_arraylist children;
            size_t num_children;
            size_t id;
            void** data;
        } file_node_cluster_t;

        typedef struct __file_bin_cluster_t {
            uint8_t* bin;
            size_t index;
            size_t id;
        } file_bin_cluster_t;

        file_node_cluster_t* v, *vc;
        file_bin_cluster_t* b;

        printf("Iterating over heirarchy\n");

        // These capacity numbers are purposely low to reduce memory consumption
        ptr_arraylist q = create_ptr_arraylist(h->num_levels * h->page_size), 
                      qq = create_ptr_arraylist(h->num_levels * h->page_size),
                      qb = create_ptr_arraylist(h->num_levels * h->page_size);

        uint64_arraylist first_children = create_uint64_arraylist(h->page_size);

        // Iterate over the first level and queue up the non-zero elements
        for(size_t c = 0; c < h->page_size; c++) {
            if(h->first_level[c]) {
                v = malloc(sizeof(file_node_cluster_t)), *vc;
                if(!v) err(1, "Memory error while allocating page information for array\n");
                v->level = 0;
                v->id = level_ids[0]++;
                v->num_children = 0;
                v->index = c;
                v->data = (void**)h->first_level[c];
                v->children = create_uint64_arraylist(h->page_size);

                append_pal(qq, v);
                append_pal(q, v);
                append_dal(first_children, v->id);
            }
        }

        // Write the ids of the first level's children
        fwrite(&first_children->pointer, sizeof(first_children->pointer), 1, fp);
        fwrite(first_children->data, sizeof(uint64_t), first_children->pointer, fp);

        destroy_uint64_arraylist(first_children);

        // Try BFS and create a path int that marks which keys map to which bins
        size_t current_level = 0, num_save = 0, previous_save = 1, explore_counter = 0;
        while(qq->pointer) {
            ++explore_counter;
            v = pop_front_pal(qq);

            if(v->level > current_level) {
                printf("\nSaving level %lu\n", current_level);

                for(size_t i = 1; i <= previous_save && q->pointer; i++) {
                    printf("%lu/%lu\r", i, previous_save);
                    fflush(stdout);

                    vc = pop_front_pal(q);
                    if(vc->level == v->level) break;

                    fwrite(&vc->level, sizeof(vc->level), 1, fp);
                    fwrite(&vc->index, sizeof(vc->index), 1, fp);
                    fwrite(&vc->id, sizeof(vc->id), 1, fp);
                    fwrite(&vc->children->pointer, sizeof(vc->children->pointer), 1, fp);
                    fwrite(vc->children->data, sizeof(uint64_t), vc->children->pointer, fp);

                    destroy_uint64_arraylist(vc->children);
                    free(vc);
                }
                explore_counter = 0;
                current_level = v->level;
                previous_save = num_save;
                num_save = 0;

                printf("\n");
            }

            size_t vc_level = v->level + 1;
            // printf("Iterating over page %lu[%lu]\n", v->level, v->index);
            for(size_t c = 0; c < h->page_size; c++) {
                printf("Iterating over page %lu[%lu](%lu/%lu): %lu/%lu\r", v->level, v->index, explore_counter, previous_save, c + 1, h->page_size);
                fflush(stdout);

                if(v->data[c]) {
                    if(v->level < h->num_levels - 2) {
                        vc = malloc(sizeof(file_node_cluster_t));
                        if(!vc) err(1, "Memory error while allocating page information for array\n");
                        vc->data = (void**)(v->data[c]);
                        vc->id = level_ids[vc_level]++;
                        vc->level = vc_level;
                        vc->index = c;
                        vc->children = create_uint64_arraylist(h->page_size);

                        append_pal(qq, vc);
                        append_pal(q, vc);
                        append_dal(v->children, vc->id);
                        num_save++;
                    }
                    else {
                        b = malloc(sizeof(file_bin_cluster_t));
                        if(!b) err(1, "Memory error while allocating bit page information for file\n");
                        b->bin = (uint8_t*)v->data[c];
                        b->id = level_ids[vc_level]++;
                        b->index = c;

                        append_pal(qb, b);
                        append_dal(v->children, b->id);
                    }
                }
            }
        }

        printf("\nSaving remaining pages\n");
        num_save = q->pointer;
        for(size_t i = 1; i <= num_save && q->pointer; i++) {
            printf("%lu/%lu\r", i, num_save);

            vc = pop_front_pal(q);

            fwrite(&vc->level, sizeof(vc->level), 1, fp);
            fwrite(&vc->index, sizeof(vc->index), 1, fp);
            fwrite(&vc->id, sizeof(vc->id), 1, fp);
            fwrite(&vc->children->pointer, sizeof(vc->children->pointer), 1, fp);
            fwrite(vc->children->data, sizeof(uint64_t), vc->children->pointer, fp);

            destroy_uint64_arraylist(vc->children);
            free(vc);
        }

        printf("\nSaving bits\n");

        // Write the bit results first
        num_save = qb->pointer;
        for(size_t i = 1; i <= num_save && qb->pointer; i++) {
            printf("%lu/%lu\r", i, num_save);

            b = pop_back_pal(qb);
            uint8_t* bytes = (uint8_t*)(b->bin);

            #ifdef heirdebug
                printf("Saving array (address: %p, level: %lu, id: %lu, count: %lu) to file\nContents: [", bytes, v->level, v->id, h->final_level->elements_per_bin);
                for(size_t b = 0; b < h->final_level->elements_per_bin; b++) {
                    if(b) printf(", ");
                    printf("%u", bytes[b]);
                }
                printf("]\n");
            #endif

            fwrite(&b->id, sizeof(b->id), 1, fp);
            fwrite(bytes, sizeof(uint8_t), h->final_level->elements_per_bin, fp);

            // free((uint16_t*)bytes); Don't free ranges of the file
            free(b);
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

    if(!fread(&h->num_bits_per_level, sizeof(h->num_bits_per_level), 1, fp)) err(11, "Error while reading data from file\n");
    fread(&h->num_levels, sizeof(h->num_levels), 1, fp);
    fread(&h->page_size, sizeof(h->page_size), 1, fp);
    h->num_bits_per_final_level = h->num_bits_per_level + (128 % h->num_bits_per_level);

    printf("Heirarchy stats:\n\tBits: %lu\n\tLevels: %lu\n\tSize: %lu\n", h->num_bits_per_level, h->num_levels, h->page_size);

    size_t *level_counts  = malloc(sizeof(size_t) * (h->num_levels + 1));
    if(!level_counts) err(1, "Memory error while allocating level counts for heirarchy file read\n");
    fread(level_counts, sizeof(size_t), h->num_levels + 1, fp);

    h->final_level = create_mmap_man(INITIAL_PAGE_SIZE, 1 << (h->num_bits_per_final_level - 3));

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
        fread(&id, sizeof(size_t), 1, fp);

        #ifdef heirdebug
            printf("Reading in info for %lu[%lu]\n", level, id);
        #endif

        if(level < (h->num_levels - 1)) {
            #ifdef heirdebug
                printf("Reading in level %lu\n", level);
            #endif

            // It's a normal array
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

            // printf("%lu\n", h->final_level->elements_per_bin);
            fread(mmap_ptr, sizeof(uint8_t), h->final_level->elements_per_bin, fp);

            #ifdef heirdebug
                printf("Read in bits %lu\n", id);
            #endif

            ptr_mappings[level][id] = mmap_ptr;
        }
    }

    return h;
}