#include "./heir_swapper.h"

#include <stdlib.h>
#include <string.h>
#include <err.h>

#define MAX_LOAD_COUNT 34000000

bin_dict create_bin_dict(size_t num_bins, size_t bin_size) {
    bin_dict d = malloc(sizeof(bin_dict_t));
    if(!d) err(1, "Memory error while allocating bin_dict\n");

    d->bin_size = bin_size;
    d->bin_count = num_bins;
    d->loaded_count = 0;
    d->bins = malloc(sizeof(uint8_t**) * num_bins);
    d->mappings = malloc(sizeof(uint8_t**) * num_bins);
    d->indices = calloc(num_bins, sizeof(size_t));
    d->keys = malloc(num_bins * sizeof(__uint128_t*));
    d->bin_sizes = calloc(num_bins, sizeof(size_t));
    d->usage_counters = malloc(num_bins * sizeof(size_t*));
    if(!(d->bins || d->indices || 
         d->keys || d->bin_sizes || 
         d->mappings || d->usage_counters)) err(1, "Memory error while allocating bins for bin_dict\n");

    return d;
}

void destroy_bin_dict(bin_dict d) {
    if(d) {
        for(size_t b = 0; b < d->bin_count; b++) {
            if(d->indices[b]) {
                free(d->bins[b]);
                free(d->keys[b]);
                free(d->usage_counters[b]);
            }
        }
        free(d->bins);
        free(d->keys);
        free(d->bin_sizes);
        free(d->indices);
        free(d);
    }
}

uint64_t find_swap_target(bin_dict d) {
    uint64_t result = 0;
    size_t bin_min = 0, element_min = 0;
    for(size_t b = 0; b < d->bin_count; b++) {
        for(size_t e = 0; e < d->bin_sizes[b]; e++) {
            if(d->usage_counters[b][e] < d->usage_counters[bin_min][element_min]) {
                bin_min = b;
                element_min = e;
            }
        }
    }
    result = (((uint64_t)bin_min) << 32) + element_min;
}

void bin_dict_load_page(bin_dict d, size_t bin, size_t e) {
    if(d->loaded_count < MAX_LOAD_COUNT) {
        uint8_t* newbin = malloc(sizeof(uint8_t) * d->bin_size);
        if(!newbin) err(1, "Memory error while allocating bin for bin_dict\n");
        memcpy(newbin, d->bins[bin][e], d->bin_size);
        d->bins[bin][e] = newbin;
        d->loaded_count++;
    }
    else {
        uint64_t target = find_swap_target(d);
        size_t bt = target >> 32, et = (size_t)target;
        memcpy(d->mappings[bt][et], d->bins[bt][et], sizeof(uint8_t) * d->bin_size);
        free(d->bins[bt][et]);
        d->bins[bt][et] = d->mappings[bt][et];

        uint8_t* newbin = malloc(sizeof(uint8_t) * d->bin_size);
        if(!newbin) err(1, "Memory error while allocating bin for bin_dict\n");
        memcpy(newbin, d->bins[bin][e], d->bin_size);
        d->bins[bin][e] = newbin;
    }
}

uint8_t* bin_dict_get(bin_dict d, __uint128_t k) {
    size_t bin = k % d->bin_count;
    if(d->indices[bin]) {
        for(size_t e = 0; e < d->indices[bin]; e++) if(d->keys[bin][e] == k) {
            d->usage_counters[bin][e]++;
            
            if(d->bins[bin][e] == d->mappings[bin][e]) bin_dict_load_page(d, bin, e);

            return d->bins[bin][e];
        }
    }
    return 0;
}

uint8_t* bin_dict_put(bin_dict d, __uint128_t k, uint8_t* ptr) {
    size_t bin = k % d->bin_count;
    if(!d->indices[bin]) {
        d->keys[bin] = malloc(sizeof(__uint128_t) * d->bin_size);
        d->bins[bin] = malloc(sizeof(uint8_t*) * d->bin_size);
        d->mappings[bin] = malloc(sizeof(uint8_t*) * d->bin_size);
        d->usage_counters[bin] = calloc(d->bin_size, sizeof(size_t));
        if(!(d->keys[bin] || d->bins[bin] || d->mappings || 
             d->usage_counters[bin])) err(1, "Memory error while allocating bin for bin_dict\n");
        d->bin_sizes[bin] = d->bin_size;
    }

    d->bins[bin][d->indices[bin]] = ptr;
    d->mappings[bin][d->indices[bin]] = ptr;
    d->keys[bin][d->indices[bin]] = k;
    d->usage_counters[bin][d->indices[bin]] = 1;

    bin_dict_load_page(d, bin, d->indices[bin]);

    d->indices[bin] += 1;

    if(d->indices[bin] == d->bin_sizes[bin]) {
        d->bin_sizes[bin] += d->bin_size;
        d->bins[bin] = realloc(d->bins[bin], sizeof(uint8_t*) * d->bin_sizes[bin]);
        d->mappings[bin] = realloc(d->mappings[bin], sizeof(uint8_t*) * d->bin_sizes[bin]);
        d->keys[bin] = realloc(d->keys[bin], sizeof(__uint128_t*) * d->bin_sizes[bin]);
        d->usage_counters[bin] = realloc(d->usage_counters[bin], sizeof(size_t) * d->bin_sizes[bin]);
        if(!(d->bins[bin] || d->keys[bin] ||
             d->usage_counters[bin])) err(1, "Memory error while allocating bin for bin_dict\n");
    }

    return d->bins[bin][d->indices[bin] - 1];
}