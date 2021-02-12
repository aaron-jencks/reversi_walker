#include "heir.h"
#include "../utils/arraylist.h"

#include <err.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <pthread.h>

#define INITIAL_CACHE_SIZE 34359738368
#define INITIAL_PAGE_SIZE 5368709120
#define INITIAL_BIN_COUNT 1000000
#define SMALL_INITIAL_BIN_COUNT 10

pthread_mutex_t heirarchy_lock;

heirarchy create_heirarchy(char* file_directory) {
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

    // h->num_bits_per_level = --shifts;
    // h->num_levels = 128 / shifts;
    shifts--;
    h->num_bits_per_final_level = shifts + (128 % shifts); // To ensure that we use every bit

    // Setup the bit directory
    struct stat st;
    char* temp = malloc(sizeof(char) * (strlen(file_directory) + 6));
    temp = memcpy(temp, file_directory, strlen(file_directory));
    memcpy(temp + strlen(file_directory), "/bits", 6);
    if(stat(temp, &st) == -1) mkdir(temp, 0700);
    size_t bin_size = (1 << (h->num_bits_per_final_level - 3)) + sizeof(__uint128_t) * (h->num_bits_per_final_level - 3);
    h->final_level = create_mmap_man(INITIAL_PAGE_SIZE, bin_size, temp);
    free(temp);

    #ifdef heir_small_bin_count
        h->bin_map = create_bin_dict(SMALL_INITIAL_BIN_COUNT, (1 << (h->num_bits_per_final_level - 3)), bin_size);
    #else
        h->bin_map = create_bin_dict(INITIAL_BIN_COUNT, (1 << (h->num_bits_per_final_level - 3)), bin_size);
    #endif

    // // Setup the level directory
    // temp = malloc(sizeof(char) * (strlen(file_directory) + 8));
    // temp = memcpy(temp, file_directory, strlen(file_directory));
    // memcpy(temp + strlen(file_directory), "/levels", 8);
    // if(stat(temp, &st) == -1) mkdir(temp, 0700);
    // h->level_map = create_mmap_man(INITIAL_PAGE_SIZE, sizeof(void*) * h->page_size, temp);
    // free(temp);

    // h->first_level = (void**)mmap_allocate_bin(h->level_map);
    // for(size_t i = 0; i < h->page_size; i++) h->first_level[i] = 0;

    return h;
}

void destroy_heirarchy(heirarchy h) {
    if(h) {
        destroy_mmap_man(h->final_level);
        destroy_bin_dict(h->bin_map);
        // destroy_mmap_man(h->level_map);
        free(h);
    }
}

uint8_t heirarchy_insert(heirarchy h, __uint128_t key) {
    // #ifdef heirdebug
    //     printf("Inserting %lu %lu into the cache\n", ((uint64_t*)&key)[1], ((uint64_t*)&key)[0]);
    // #endif

    while(pthread_mutex_trylock(&heirarchy_lock)) sched_yield();

    __uint128_t lower_key = key >> h->num_bits_per_final_level;
    uint8_t* dict_resp = bin_dict_get(h->bin_map, lower_key);

    if(!dict_resp) {
        // Allocate a new bin for it
        uint8_t* new_bin = mmap_allocate_bin(h->final_level);
        dict_resp = bin_dict_put(h->bin_map, lower_key, new_bin);
    }



    // __uint128_t key_copy = key, bit_placeholder = 0; 
    // size_t bits, level = 1;
    // void** phase = h->first_level;

    // // Create the placeholder to extract the correct number of bits every time
    // bit_placeholder = ((1 << h->num_bits_per_level) - 1) << (128 - h->num_bits_per_level);

    // // Traverse through all of the levels
    // for(level = 1; level < h->num_levels; level++) {
    //     bits = (size_t)((key_copy & bit_placeholder) >> (128 - h->num_bits_per_level));
    //     key_copy = key_copy << h->num_bits_per_level;

    //     #ifdef heirdebug
    //         printf("Bits for level %lu is %lu\n", level, bits);
    //     #endif

    //     if(level < (h->num_levels - 1) && !phase[bits]) {
    //         // printf("Allocating %lu[%lu]\n", level, bits);
    //         phase[bits] = mmap_allocate_bin(h->level_map);
    //         for(size_t i = 0; i < h->page_size; i++) 
    //             ((uint64_t*)phase[bits])[i] = 0;
    //     }
    //     else if(!phase[bits]) {
    //         // Allocate a new bin
    //         phase[bits] = mmap_allocate_bin(h->final_level);

    //         // for(size_t b = 0; b < h->final_level->max_page_size; b++) ((uint8_t*)(phase[bits]))[b] = 0;
    //         ((__uint128_t*)phase[bits])[0] = key >> h->num_bits_per_final_level + 3;
    //         size_t num_jumps = h->final_level->elements_per_bin / 8;
    //         for(size_t b = 2; b < num_jumps; b++) ((uint64_t*)(phase[bits]))[b] = 0;
    //         for(size_t b = 0; b < h->final_level->elements_per_bin % 8; b++) ((uint8_t*)(phase[bits]))[num_jumps + b] = 0;
            
    //         #ifdef heirdebug
    //             printf("Generating a new bin for entry %lu[%lu] @ %p\n", level - 1, bits, phase[bits]);
    //         #endif
    //     }
        
    //     phase = (void**)phase[bits];
    // }

    // Traverse the last level
    // Use 3 bits to determine the bit
    // Use the rest to determine the byte

    // Create the placeholder to extract the correct number of bits every time, for the edge case of the final level, where the least 3 bits are used for bit position
    // bit_placeholder = (1 << (h->num_bits_per_final_level - 2)) - 1;

    // Extract the bit from the last level
    // __uint128_t key_copy = key >> (128 - h->num_bits_per_final_level);

    __uint128_t key_ph = key & ((((__uint128_t)1) << h->num_bits_per_final_level) - 1);
    size_t bits = (size_t)((key_ph >> 3) + sizeof(__uint128_t));
    uint8_t bit = key_ph & 7;

    #ifdef heirdebug
        printf("Key Stats: %lu %lu %lu %u\n", ((uint64_t*)&lower_key)[0], ((uint64_t*)&lower_key)[1], bits, bit);
        printf("Bits for final level is %lu, byte index is %u\n", bits, bit);
    #endif

    // Insert the new bit if it's not already in the array
    // printf("Bit value: %lu\n", bits);
    uint8_t byte = dict_resp[bits];
    uint8_t ph = 1 << bit;

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

    dict_resp[bits] |= ph;

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
        if(!fwrite(&h->num_bits_per_final_level, sizeof(h->num_bits_per_final_level), 1, fp)) err(12, "Error while writing file data\n");
        // fwrite(&h->num_levels, sizeof(h->num_levels), 1, fp);
        fwrite(&h->page_size, sizeof(h->page_size), 1, fp);
        mmap_to_file(h->final_level, fp);

        printf("Heirarchy stats:\n\tBits: %lu\n\tSize: %lu\n", h->num_bits_per_final_level, h->page_size);

        // size_t* level_ids = calloc(h->num_levels + 1, sizeof(size_t));
        // if(!level_ids) err(1, "Memory Error while allocating array id array for heirarchy file saving algorithm\n");

        // typedef struct __file_node_cluster_t {
        //     size_t level;
        //     size_t index;
        //     uint64_arraylist children;
        //     size_t num_children;
        //     size_t id;
        //     void** data;
        // } file_node_cluster_t;

        // typedef struct __file_bin_cluster_t {
        //     uint8_t* bin;
        //     size_t index;
        //     size_t id;
        // } file_bin_cluster_t;

        // file_node_cluster_t* v, *vc;
        // file_bin_cluster_t* b;

        printf("Iterating over heirarchy\n");

        // These capacity numbers are purposely low to reduce memory consumption
        // ptr_arraylist q = create_ptr_arraylist(h->num_levels * h->page_size), 
        //               qq = create_ptr_arraylist(h->num_levels * h->page_size),
        //               qb = create_ptr_arraylist(h->num_levels * h->page_size);

        // uint64_arraylist first_children = create_uint64_arraylist(h->page_size);

        // // Iterate over the first level and queue up the non-zero elements
        // for(size_t c = 0; c < h->page_size; c++) {
        //     if(h->first_level[c]) {
        //         v = malloc(sizeof(file_node_cluster_t)), *vc;
        //         if(!v) err(1, "Memory error while allocating page information for array\n");
        //         v->level = 0;
        //         v->id = level_ids[0]++;
        //         v->num_children = 0;
        //         v->index = c;
        //         v->data = (void**)h->first_level[c];
        //         v->children = create_uint64_arraylist(h->page_size);

        //         append_pal(qq, v);
        //         append_pal(q, v);
        //         append_dal(first_children, v->id);
        //     }
        // }

        // // Write the ids of the first level's children
        // fwrite(&first_children->pointer, sizeof(first_children->pointer), 1, fp);
        // fwrite(first_children->data, sizeof(uint64_t), first_children->pointer, fp);

        // destroy_uint64_arraylist(first_children);

        // // Try BFS and create a path int that marks which keys map to which bins
        // size_t current_level = 0, num_save = 0, previous_save = 1, explore_counter = 0;
        // while(qq->pointer) {
        //     ++explore_counter;
        //     v = pop_front_pal(qq);

        //     if(v->level > current_level) {
        //         printf("\nSaving level %lu\n", current_level);

        //         for(size_t i = 1; i <= previous_save && q->pointer; i++) {
        //             printf("%lu/%lu\r", i, previous_save);
        //             fflush(stdout);

        //             vc = pop_front_pal(q);
        //             if(vc->level == v->level) break;

        //             fwrite(&vc->level, sizeof(vc->level), 1, fp);
        //             fwrite(&vc->index, sizeof(vc->index), 1, fp);
        //             fwrite(&vc->id, sizeof(vc->id), 1, fp);
        //             fwrite(&vc->children->pointer, sizeof(vc->children->pointer), 1, fp);
        //             fwrite(vc->children->data, sizeof(uint64_t), vc->children->pointer, fp);

        //             destroy_uint64_arraylist(vc->children);
        //             free(vc);
        //         }
        //         explore_counter = 0;
        //         current_level = v->level;
        //         previous_save = num_save;
        //         num_save = 0;

        //         printf("\n");
        //     }

        //     size_t vc_level = v->level + 1;
        //     // printf("Iterating over page %lu[%lu]\n", v->level, v->index);
        //     for(size_t c = 0; c < h->page_size; c++) {
        //         printf("Iterating over page %lu[%lu](%lu/%lu): %lu/%lu\r", v->level, v->index, explore_counter, previous_save, c + 1, h->page_size);
        //         fflush(stdout);

        //         if(v->data[c]) {
        //             if(v->level < h->num_levels - 2) {
        //                 vc = malloc(sizeof(file_node_cluster_t));
        //                 if(!vc) err(1, "Memory error while allocating page information for array\n");
        //                 vc->data = (void**)(v->data[c]);
        //                 vc->id = level_ids[vc_level]++;
        //                 vc->level = vc_level;
        //                 vc->index = c;
        //                 vc->children = create_uint64_arraylist(h->page_size);

        //                 append_pal(qq, vc);
        //                 append_pal(q, vc);
        //                 append_dal(v->children, vc->id);
        //                 num_save++;
        //             }
        //             else {
        //                 b = malloc(sizeof(file_bin_cluster_t));
        //                 if(!b) err(1, "Memory error while allocating bit page information for file\n");
        //                 b->bin = (uint8_t*)v->data[c];
        //                 b->id = level_ids[vc_level]++;
        //                 b->index = c;

        //                 append_pal(qb, b);
        //                 append_dal(v->children, b->id);
        //             }
        //         }
        //     }
        // }

        // printf("\nSaving remaining pages\n");
        // num_save = q->pointer;
        // for(size_t i = 1; i <= num_save && q->pointer; i++) {
        //     printf("%lu/%lu\r", i, num_save);

        //     vc = pop_front_pal(q);

        //     fwrite(&vc->level, sizeof(vc->level), 1, fp);
        //     fwrite(&vc->index, sizeof(vc->index), 1, fp);
        //     fwrite(&vc->id, sizeof(vc->id), 1, fp);
        //     fwrite(&vc->children->pointer, sizeof(vc->children->pointer), 1, fp);
        //     fwrite(vc->children->data, sizeof(uint64_t), vc->children->pointer, fp);

        //     destroy_uint64_arraylist(vc->children);
        //     free(vc);
        // }

        printf("\nSaving bits\n");

        // Write the bit results first
        // num_save = qb->pointer;
        // for(size_t i = 1; i <= num_save && qb->pointer; i++) {
        //     printf("%lu/%lu\r", i, num_save);

        //     b = pop_back_pal(qb);
        //     uint8_t* bytes = (uint8_t*)(b->bin);

        //     #ifdef heirdebug
        //         printf("Saving array (address: %p, level: %lu, id: %lu, count: %lu) to file\nContents: [", bytes, v->level, v->id, h->final_level->elements_per_bin);
        //         for(size_t b = 0; b < h->final_level->elements_per_bin; b++) {
        //             if(b) printf(", ");
        //             printf("%u", bytes[b]);
        //         }
        //         printf("]\n");
        //     #endif

        //     fwrite(&b->id, sizeof(b->id), 1, fp);
        //     fwrite(bytes, sizeof(uint8_t), h->final_level->elements_per_bin, fp);

        //     // free((uint16_t*)bytes); Don't free ranges of the file
        //     free(b);
        // }

        // free(level_ids);
        // destroy_ptr_arraylist(q);
        // destroy_ptr_arraylist(qq);
        // destroy_ptr_arraylist(qb);
    }
}

heirarchy from_file_heir(FILE* fp) {
    // File Format:
    // [level][array id][contents]...0
    // [bits per level][num levels][page size][reverse order traversal]...

    heirarchy h = malloc(sizeof(heirarchy_str));
    if(!h) err(1, "Memory error while allocating new heirarchy for file read\n");

    if(!fread(&h->num_bits_per_final_level, sizeof(h->num_bits_per_final_level), 1, fp)) err(11, "Error while reading data from file\n");
    // fread(&h->num_levels, sizeof(h->num_levels), 1, fp);
    fread(&h->page_size, sizeof(h->page_size), 1, fp);
    // h->num_bits_per_final_level = h->num_bits_per_level + (128 % h->num_bits_per_level);

    printf("Heirarchy stats:\n\tBits: %lu\n\tSize: %lu\n", h->num_bits_per_final_level, h->page_size);

    h->final_level = mmap_from_file(fp);

    h->bin_map = create_bin_dict(INITIAL_BIN_COUNT, 1 << (h->num_bits_per_final_level - 3), h->final_level->elements_per_bin);

    printf("Loading previous bins... This might take awhile\n");
    for(size_t p = 0; p < h->final_level->num_pages; p++) {
        for(size_t b = 0; b < h->final_level->pages[p]->size; b++) {
            size_t bin_index = h->final_level->elements_per_bin * b;
            mmap_page mp = h->final_level->pages[p];
            bin_dict_put(h->bin_map, ((__uint128_t*)(mp->map + bin_index))[0], mp->map + bin_index);
        }
    }
    printf("Bin loading Complete!\n");

    // size_t level, id;
    // uint8_t* mmap_ptr;
    // uint16_t* arr_contents = malloc(sizeof(uint16_t) * h->page_size), arr_element;
    // if(!arr_contents) err(1, "Memory error while allocating contents for array in heirarchy file read\n");
    // void** ptr;

    // // Generate a lookup table to store mappings from ids to pointer values
    // void*** ptr_mappings = malloc(sizeof(void**) * (h->num_levels + 1));
    // if(!ptr_mappings) err(1, "Memory error while allocating ptr mapping array for heirarchy file read\n");
    // for(size_t m = 0; m < (h->num_levels + 1); m++) {
    //     ptr_mappings[m] = calloc(level_counts[m], sizeof(void*));
    //     if(!ptr_mappings[m]) err(1, "Memory error while allocating ptr mapping array for heirarchy file read\n");
    // }

    // while(1) {
    //     fread(&level, sizeof(size_t), 1, fp);
    //     fread(&id, sizeof(size_t), 1, fp);

    //     #ifdef heirdebug
    //         printf("Reading in info for %lu[%lu]\n", level, id);
    //     #endif

    //     if(level < (h->num_levels - 1)) {
    //         #ifdef heirdebug
    //             printf("Reading in level %lu\n", level);
    //         #endif

    //         // It's a normal array
    //         fread(arr_contents, sizeof(uint16_t), h->page_size, fp);

    //         ptr = malloc(sizeof(void*) * h->page_size);
    //         if(!ptr) err(1, "Memory error while allocating array in heirarchy file read\n");
            
    //         for(size_t c = 0; c < h->page_size; c++) ptr[c] = (arr_contents[c] != 65535) ? ptr_mappings[level + 1][arr_contents[c]] : 0; 
    //         ptr_mappings[level][id] = ptr;

    //         if(!level) {
    //             // We've read in the root array
    //             h->first_level = ptr;
    //             break;
    //         }
    //     }
    //     else {
    //         // It's part of the bit string
    //         mmap_ptr = mmap_allocate_bin(h->final_level);

    //         // printf("%lu\n", h->final_level->elements_per_bin);
    //         fread(mmap_ptr, sizeof(uint8_t), h->final_level->elements_per_bin, fp);

    //         #ifdef heirdebug
    //             printf("Read in bits %lu\n", id);
    //         #endif

    //         ptr_mappings[level][id] = mmap_ptr;
    //     }
    // }

    return h;
}