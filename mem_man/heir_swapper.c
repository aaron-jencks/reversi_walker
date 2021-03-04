#include "./heir_swapper.h"
#include "../utils/heapsort.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

#define MAX_LOAD_COUNT 51000000
#define SMALL_LOAD_COUNT 5

double bin_dict_load_factor(bin_dict d) { return (double)d->element_count / (double)d->bin_count; }

bin_dict create_bin_dict(size_t num_bins, size_t bin_size, size_t element_size) {
    bin_dict d = malloc(sizeof(bin_dict_t));
    if(!d) err(1, "Memory error while allocating bin_dict\n");

    d->bin_size = bin_size;
    d->bin_count = num_bins;
    d->element_size = element_size;
    d->loaded_count = 0;
    d->element_count = 0;
    d->bins = malloc(sizeof(uint8_t**) * num_bins);
    // d->mappings = malloc(sizeof(uint8_t**) * num_bins);
    d->indices = calloc(num_bins, sizeof(size_t));
    d->keys = malloc(num_bins * sizeof(__uint128_t*));
    d->bin_sizes = calloc(num_bins, sizeof(size_t));
    d->usage_counters = malloc(num_bins * sizeof(size_t*));
    if(!(d->bins || d->indices || 
         d->keys || d->bin_sizes || 
         d->usage_counters)) err(1, "Memory error while allocating bins for bin_dict\n");
    d->cache = create_hashtable(1000000);

    return d;
}

void destroy_bin_dict(bin_dict d) {
    if(d) {
        for(size_t b = 0; b < d->bin_count; b++) {
            if(d->indices[b]) {
                for(size_t e = 0; e < d->indices[b]; e++) if(d->bins[b][e] != d->mappings[b][e]) free(d->bins[b][e]);
                free(d->bins[b]);
                free(d->keys[b]);
                free(d->usage_counters[b]);
                // free(d->mappings[b]);
            }
        }
        free(d->usage_counters);
        // free(d->mappings);
        free(d->bins);
        free(d->keys);
        free(d->bin_sizes);
        free(d->indices);
        free(d);
    }
}

size_t* find_swap_target(bin_dict d) {
    size_t* result = malloc(sizeof(size_t) * 2);
    size_t bin_min = 0, element_min = INT64_MAX;
    for(size_t b = 0; b < d->bin_count; b++) {
        for(size_t e = 0; e < d->indices[b]; e++) {
            if((element_min != INT64_MAX && 
                    d->usage_counters[b][e] < d->usage_counters[bin_min][element_min] && 
                    d->bins[b][e] != d->mappings[b][e]) || 
                d->bins[b][e] != d->mappings[b][e]) {
                bin_min = b;
                element_min = e;
            }
        }
    }
    result[0] = bin_min;
    result[1] = element_min;
    return result;
}

size_t** find_n_swap_targets(bin_dict d, size_t n) {
    // TODO collect all of the loaded values
    size_t **result = malloc(sizeof(size_t*) * n), **arr = malloc(sizeof(size_t*) * d->loaded_count);
    if(!(result || arr)) err(1, "Memory error while allocating bin_dict swap targets\n");

    size_t arr_index = 0, b, e;
    for(size_t b = 0; b < d->bin_count; b++) {
        for(size_t e = 0; e < d->indices[b]; e++) {
            arr[arr_index] = malloc(sizeof(size_t) * 3);
            if(!arr[arr_index]) err(1, "Memory error while allocating bin for swap targets\n");
            arr[arr_index][0] = d->usage_counters[b][e];
            arr[arr_index][1] = b;
            arr[arr_index++][2] = e;
        }
    }

    printf("Found %lu loaded values\n", arr_index);

    // printf("Unsorted array is: [%lu %p, %lu %p, %lu %p, %lu %p, %lu %p]\n", arr[0][0], arr[0], arr[1][0], arr[1], arr[2][0], arr[2], arr[3][0], arr[3], arr[4][0], arr[4]);
    // sort the loaded values
    heapsort(arr, arr_index);
    // printf("Sorted array is: [%lu %p, %lu %p, %lu %p, %lu %p, %lu %p]\n", arr[0][0], arr[0], arr[1][0], arr[1], arr[2][0], arr[2], arr[3][0], arr[3], arr[4][0], arr[4]);
    
    // slice of the lowest n values
    memcpy(result, arr, sizeof(size_t*) * ((arr_index < n) ? arr_index : n));

    for(size_t a = n; a < arr_index; a++) {
        free(arr[a]);
    }
    free(arr);

    return result;
}

void bin_dict_load_page(bin_dict d, size_t bin, size_t e) {
    #ifdef bin_dict_small_load
        if(d->loaded_count < SMALL_LOAD_COUNT) {
    #else
        if(d->loaded_count < MAX_LOAD_COUNT) {
    #endif
        uint8_t* newbin = malloc(sizeof(uint8_t) * d->element_size);
        if(!newbin) err(1, "Memory error while allocating bin for bin_dict\n");
        memcpy(newbin, d->bins[bin][e], d->element_size);
        d->bins[bin][e] = newbin;
        d->loaded_count++;
    }
    else {
        printf("Unloading bins\n");
        #ifdef bin_dict_small_load
            size_t** data = find_n_swap_targets(d, 3);
        #else
            size_t** data = find_n_swap_targets(d, 30000000);
        #endif
        printf("Found the swap targets\n");
        #ifdef bin_dict_small_load
            for(size_t i = 0; i < 3; i++) {
                printf("\r%lu/3", i + 1);
        #else
            for(size_t i = 0; i < 30000000; i++) {
                printf("\r%lu/30,000,000", i + 1);
        #endif
            size_t* target = data[i];
            size_t bt = target[1], et = target[2];
            if(et == INT64_MAX) err(12, "No valid target found for swapping %lu[%lu]\n", bin, e);
            // printf("Swapping out %lu[%lu] for %lu[%lu]\n", bt, et, bin, e);
            memcpy(d->mappings[bt][et], d->bins[bt][et], sizeof(uint8_t) * d->element_size);
            free(d->bins[bt][et]);
            free(data[i]);
            d->bins[bt][et] = d->mappings[bt][et];
        }
        free(data);
        printf("\n");

        #ifdef bin_dict_small_load
            d->loaded_count -= 2;
        #else
            d->loaded_count -= 29999999;
        #endif

        uint8_t* newbin = malloc(sizeof(uint8_t) * d->element_size);
        if(!newbin) err(1, "Memory error while allocating bin for bin_dict\n");
        memcpy(newbin, d->bins[bin][e], d->element_size);
        d->bins[bin][e] = newbin;
    }
}

uint8_t* bin_dict_get(bin_dict d, __uint128_t k) {
    size_t bin = k % d->bin_count;
    if(d->indices[bin]) {
        for(size_t e = 0; e < d->indices[bin]; e++) if(d->keys[bin][e] == k) {
            d->usage_counters[bin][e]++;
            
            // if(d->bins[bin][e] == d->mappings[bin][e]) bin_dict_load_page(d, bin, e);

            return d->bins[bin][e];
        }
    }
    // TODO miss, check the cache
    if(exists_hs(d->cache, k)) return get_hs(d->cache, k);
    
    return 0;
}

uint8_t* bin_dict_put(bin_dict d, __uint128_t k, uint8_t* ptr) {
    size_t bin = k % d->bin_count;
    if(!d->indices[bin]) {
        d->keys[bin] = malloc(sizeof(__uint128_t) * d->bin_size);
        d->bins[bin] = malloc(sizeof(uint8_t*) * d->bin_size);
        // d->mappings[bin] = malloc(sizeof(uint8_t*) * d->bin_size);
        d->usage_counters[bin] = calloc(d->bin_size, sizeof(size_t));
        if(!(d->keys[bin] || d->bins[bin] || // d->mappings || 
             d->usage_counters[bin])) err(1, "Memory error while allocating bin for bin_dict\n");
        d->bin_sizes[bin] = d->bin_size;
    }

    // printf("Inserting new element into %lu %lu @ %p\n", bin, d->indices[bin], ptr);

    d->bins[bin][d->indices[bin]] = ptr;
    // d->mappings[bin][d->indices[bin]] = ptr;
    d->keys[bin][d->indices[bin]] = k;
    d->usage_counters[bin][d->indices[bin]] = 1;
    d->element_count++;

    if(bin_dict_load_factor(d) > 15) {
        size_t** targets = find_n_swap_targets(d, 14 * d->bin_count);

        for(size_t t = 0; t < 14 * d->bin_count; t++) {
            size_t* target = targets[t];
            size_t bt = target[1], et = target[2];
            // printf("Swapping out %lu[%lu] for %lu[%lu]\n", bt, et, bin, e);
            put_hs(d->cache, d->keys[bt][et], d->bins[bt][et]);
            d->bins[bt][et] = 0;
            free(targets[t]);
        }

        for(size_t b = 0; b < d->bin_count; b++) {
            for(size_t e = 0; e < d->indices[b]; e++) {
                if(!d->bins[b][e]) {
                    for(size_t et = e; et < d->indices[b]; et++) {
                        d->bins[b][et] = d->bins[b][et + 1];
                        d->keys[b][et] = d->keys[b][et + 1];
                        d->usage_counters[b][et] = d->usage_counters[b][et + 1];
                    }
                }
            }
        }
    }

    // bin_dict_load_page(d, bin, d->indices[bin]);

    d->indices[bin] += 1;

    if(d->indices[bin] == d->bin_sizes[bin]) {
        // printf("Reallocating bin %lu\n", bin);
        d->bin_sizes[bin] += d->bin_size;
        d->bins[bin] = realloc(d->bins[bin], sizeof(uint8_t*) * d->bin_sizes[bin]);
        // d->mappings[bin] = realloc(d->mappings[bin], sizeof(uint8_t*) * d->bin_sizes[bin]);
        d->keys[bin] = realloc(d->keys[bin], sizeof(__uint128_t) * d->bin_sizes[bin]);
        d->usage_counters[bin] = realloc(d->usage_counters[bin], sizeof(size_t) * d->bin_sizes[bin]);
        if(!(d->bins[bin] || d->keys[bin] ||
             d->usage_counters[bin])) err(1, "Memory error while allocating bin for bin_dict\n");
    }

    return d->bins[bin][d->indices[bin] - 1];
}

double bin_dict_load_factor(bin_dict d) { return (double)d->element_count / (double)d->bin_count; }