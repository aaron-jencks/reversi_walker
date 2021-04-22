#include "heir.hpp"
#include "../utils/tarraylist.hpp"
#include "../utils/dictionary/dict_def.h"

#include <err.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <pthread.h>

#define INITIAL_CACHE_SIZE 34359738368
#define INITIAL_PAGE_SIZE 5368709120

pthread_mutex_t heirarchy_lock;
pthread_mutex_t heirarchy_cache_lock;


heirarchy create_heirarchy(char* file_directory) {
    heirarchy h = (heirarchy)malloc(sizeof(heirarchy_str));
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
    // shifts--;
    h->num_bits_per_final_level = --shifts; // To ensure that we use every bit

    // Setup the bit directory
    struct stat st;
    char* temp = (char*)malloc(sizeof(char) * (strlen(file_directory) + 6));
    temp = (char*)memcpy(temp, file_directory, strlen(file_directory));
    memcpy(temp + strlen(file_directory), "/bits", 6);
    if(stat(temp, &st) == -1) mkdir(temp, 0700);
    size_t bin_size = (1 << (h->num_bits_per_final_level - 3)) + sizeof(__uint128_t);
    h->final_level = create_mmap_man(INITIAL_PAGE_SIZE, bin_size, temp);
    free(temp);

    h->fixed_cache = create_fixed_size_dictionary(INITIAL_BIN_COUNT * INITIAL_BIN_SIZE, FLUSH_COUNT, INITIAL_BIN_COUNT, INITIAL_BIN_SIZE);
    h->rehashing_cache = create_rehashing_dictionary(INITIAL_BIN_COUNT, INITIAL_BIN_SIZE);

    h->temp_board_cache = create_fixed_size_dictionary(186000000, 124000000, 3200000, 60);

    h->collision_count = 0;

    h->highest_complete_level = 0;
    h->level_mappings = (Arraylist<__uint128_t>**)malloc(sizeof(Arraylist<__uint128_t>*) * 64);
    if(!h->level_mappings) err(1, "Memory error while allocating level mappings for heirarchy\n");
    for(size_t i = 0; i < 64; i++) h->level_mappings[i] = new Arraylist<__uint128_t>(1000);

    return h;
}

void destroy_heirarchy(heirarchy h) {
    if(h) {
        destroy_mmap_man(h->final_level);
        destroy_fixed_dictionary(h->fixed_cache);
        destroy_rehashing_dictionary(h->rehashing_cache);
        for(size_t i = 0; i < 64; i++) delete h->level_mappings[i];
        free(h->level_mappings);
        free(h);
    }
}

uint8_t heirarchy_insert(heirarchy h, __uint128_t key, size_t level) {
    // #ifdef heirdebug
    //     printf("Inserting %lu %lu into the cache\n", ((uint64_t*)&key)[1], ((uint64_t*)&key)[0]);
    // #endif

    // h->sem->signal_read();

    __uint128_t lower_key = key >> h->num_bits_per_final_level;
    uint8_t* dict_resp = fdict_get(h->fixed_cache, lower_key);

    // h->sem->signal_read_finish();

    if(!dict_resp) {
        // h->sem->signal_write();
        while(pthread_mutex_trylock(&heirarchy_lock)) sched_yield();
        dict_resp = fdict_get(h->fixed_cache, lower_key);  // check one more time, just in case.

        if(!dict_resp) {
            if(!(dict_resp = hdict_get(h->rehashing_cache, lower_key))) {
                // Allocate a new bin for it
                uint8_t* new_bin = mmap_allocate_bin(h->final_level);

                ((__uint128_t*)new_bin)[0] = lower_key;
                new_bin = new_bin + sizeof(__uint128_t);

                hdict_put(h->rehashing_cache, lower_key, new_bin);
                dict_resp = new_bin;
            }
            fdict_put(h->fixed_cache, lower_key, dict_resp);
        }

        // h->sem->signal_write_finish();
        pthread_mutex_unlock(&heirarchy_lock);
    }

    // Extract the bit from the last level
    // 00111

    __uint128_t key_ph = key & ((((__uint128_t)1) << h->num_bits_per_final_level) - 1);
    size_t bits = (size_t)((key_ph >> 3) + sizeof(__uint128_t));
    uint8_t bit = key_ph & 7;

    #ifdef heirdebug
        printf("Key Stats: %lu %lu %lu %u\n", ((uint64_t*)&lower_key)[0], ((uint64_t*)&lower_key)[1], bits, bit);
        printf("Bits for final level is %lu, byte index is %u\n", bits, bit);
    #endif

    // h->sem->signal_read();

    // Insert the new bit if it's not already in the array
    // printf("Bit value: %lu\n", bits);
    uint8_t byte = dict_resp[bits];
    uint8_t ph = 1 << bit;

    if(byte & ph) {
        // #ifdef heirdebug
        //     printf("%lu %lu is already in the cache\n", ((uint64_t*)&key)[1], ((uint64_t*)&key)[0]);
        // #endif

        h->collision_count++;

        // pthread_mutex_unlock(&heirarchy_lock);
        // h->sem->signal_read_finish();

        return 0;
    }
    else {
        // h->sem->signal_read_finish();
        // h->sem->signal_write();
        while(pthread_mutex_trylock(&heirarchy_lock)) sched_yield();

        // Check one more time just in case.
        byte = dict_resp[bits];

        if(!(byte & ph)) {
            dict_resp[bits] |= ph;
            h->level_mappings[level]->append(lower_key);
            // h->sem->signal_write_finish();
            pthread_mutex_unlock(&heirarchy_lock);
            return 1;
        }
        else {
            // h->sem->signal_write_finish();
            pthread_mutex_unlock(&heirarchy_lock);
            return 0;
        }
    }
}

void heirarchy_purge_level(heirarchy h, size_t level) {
    for(size_t i = 0; i < h->level_mappings[level]->pointer; i++) {
        __uint128_t key = h->level_mappings[level]->data[i];
        if(fdict_get(h->fixed_cache, key)) {
            fdict_remove(h->fixed_cache, key);
        }
        if(hdict_get(h->rehashing_cache, key)) {
            hdict_remove(h->rehashing_cache, key);
        }
        h->level_mappings[level]->pop(h->level_mappings[level]->index(key));
    }
}

uint8_t heirarchy_insert_cache(heirarchy h, __uint128_t key) {
    // #ifdef heirdebug
    //     printf("Inserting %lu %lu into the cache\n", ((uint64_t*)&key)[1], ((uint64_t*)&key)[0]);
    // #endif

    // while(pthread_mutex_trylock(&heirarchy_lock)) sched_yield();

    // h->csem->signal_read();

    uint8_t dict_resp = fdict_exists(h->temp_board_cache, key);

    // h->csem->signal_read_finish();

    if(!dict_resp) {
        // h->csem->signal_write();
        while(pthread_mutex_trylock(&heirarchy_cache_lock)) sched_yield();

        dict_resp = fdict_exists(h->temp_board_cache, key); // Check one more time just in case.

        if(!dict_resp) {
            fdict_put(h->temp_board_cache, key, NULL);
            pthread_mutex_unlock(&heirarchy_cache_lock);
            return 1;
        }

        // h->csem->signal_write_finish();
        pthread_mutex_unlock(&heirarchy_cache_lock);
    }

    h->collision_count++;
    return 0;
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
    }
}

heirarchy from_file_heir(FILE* fp) {
    // File Format:
    // [level][array id][contents]...0
    // [bits per level][num levels][page size][reverse order traversal]...

    heirarchy h = (heirarchy)malloc(sizeof(heirarchy_str));
    if(!h) err(1, "Memory error while allocating new heirarchy for file read\n");

    if(!fread(&h->num_bits_per_final_level, sizeof(h->num_bits_per_final_level), 1, fp)) err(11, "Error while reading data from file\n");
    // fread(&h->num_levels, sizeof(h->num_levels), 1, fp);
    fread(&h->page_size, sizeof(h->page_size), 1, fp);
    // h->num_bits_per_final_level = h->num_bits_per_level + (128 % h->num_bits_per_level);

    printf("Heirarchy stats:\n\tBits: %lu\n\tSize: %lu\n", h->num_bits_per_final_level, h->page_size);

    h->final_level = mmap_from_file(fp);

    h->fixed_cache = create_fixed_size_dictionary(INITIAL_BIN_COUNT * INITIAL_BIN_SIZE, FLUSH_COUNT, INITIAL_BIN_COUNT, INITIAL_BIN_SIZE);
    h->rehashing_cache = create_rehashing_dictionary(INITIAL_BIN_COUNT, INITIAL_BIN_SIZE);

    printf("Loading previous bins... This might take awhile\n");
    size_t total = 0, current = 0;
    for(size_t p = 0; p < h->final_level->num_pages; p++) total += h->final_level->pages[p]->size;
    for(size_t p = 0; p < h->final_level->num_pages; p++) {
        for(size_t b = 0; b < h->final_level->pages[p]->size; b++) {
            printf("\rIterating over %'lu/%'lu", ++current, total);
            size_t bin_index = h->final_level->elements_per_bin * b;
            mmap_page mp = h->final_level->pages[p];
            hdict_put(h->rehashing_cache, ((__uint128_t*)(mp->map + bin_index))[0], mp->map + bin_index);
        }
    }
    printf("\nBin loading Complete!\n");

    return h;
}