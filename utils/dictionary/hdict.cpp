#include "hdict.hpp"
#include "../heapsort.h"

#include <stdlib.h>
#include <stdio.h>
#include <err.h>

hdict create_rehashing_dictionary(size_t bin_count, size_t initial_bin_size) {
    hdict d = (hdict)malloc(sizeof(hdict_t));
    if(!d) err(1, "Memory error while allocating rehashing dictionary\n");

    d->bins = create_dmempage(BIN_PAGE_COUNT, bin_count, initial_bin_size);
    if(!(d->bins)) err(1, "Memory error while allocating rehashing dictionary\n");

    d->bin_count = bin_count;
    d->size = 0;

    return d;
}

void destroy_rehashing_dictionary(hdict d) {
    if(d) {
        destroy_dmempage(d->bins);
        free(d);
    }
}

dmempage_buff hdict_get_all(hdict d) {
    dmempage_buff result = create_dmempage_buff(d->size, BIN_PAGE_COUNT);

    __uint128_t i = 0;
    uint8_t finished = 0;
    for(size_t p = 0; p < d->bins->page_count && !finished; p++) 
        for(size_t b = 0; b < d->bins->count_per_page && !finished; b++) 
            for(size_t e = 0; e < d->bins->bin_counts[p][b] && !finished; e++) {
                dmempage_buff_put(result, i++, d->bins->pages[p][b][e]);
                if(i == d->size) {
                    finished = 1;
                    break;
                }
            }

    return result;
}

uint8_t* hdict_get(hdict d, __uint128_t k) {
    __uint128_t bin = k % d->bin_count;
    if(dmempage_value_in_bin(d->bins, bin, k)) return dmempage_get(d->bins, bin, k);
    return 0;
}

void hdict_put(hdict d, __uint128_t k, uint8_t* value) {
    __uint128_t bin = k % d->bin_count;
    dmempage_append_bin(d->bins, bin, dict_pair_t{k, value});
    d->size++;

    if(hdict_load_factor(d) >= DICT_LOAD_LIMIT) {
        printf("Rehashing hashtable\n");

        printf("Collecting data\n");
        dmempage_buff contents = hdict_get_all(d);

        printf("Flushing mempage\n");
        dmempage_clear_all(d->bins);

        printf("Reallocating mempage\n");
        dmempage_realloc(d->bins, d->bin_count << 1);
        d->bin_count = d->bin_count << 1;
        for(__uint128_t e = 0; e < contents->num_element; e++) {
            dict_pair_t v = dmempage_buff_get(contents, e);
            dmempage_append_bin(d->bins, v.key % d->bin_count, v);
        }
    }
}

__uint128_t hdict_remove(hdict d, __uint128_t k) {
    __uint128_t bin = k % d->bin_count;
    dmempage_remove(d->bins, bin, k);
    d->size--;
    return k;
}

double hdict_load_factor(hdict d) { return (double)d->size / (double)d->bin_count; }