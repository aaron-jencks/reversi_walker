#include "fdict.hpp"
#include "../heapsort.h"

#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <assert.h>

fdict create_fixed_size_dictionary(size_t max_element_count, size_t flush_count, size_t bin_count, size_t initial_bin_size) {
    fdict d = (fdict)malloc(sizeof(fdict_t));
    if(!d) err(1, "Memory error while allocating fixed dictionary\n");

    d->bins = (Arraylist<dict_usage_pair_t>**)malloc(sizeof(Arraylist<dict_usage_pair_t>*) * bin_count);
    if(!(d->bins)) err(1, "Memory error while allocating fixed dictionary\n");

    for(size_t b = 0; b < bin_count; b++) {
        d->bins[b] = new Arraylist<dict_usage_pair_t>(initial_bin_size);
    }

    d->bin_count = bin_count;
    d->bin_size = initial_bin_size;
    d->max_element_count = max_element_count;
    d->flush_count = flush_count;
    d->size = 0;

    return d;
}

void destroy_fixed_dictionary(fdict d) {
    if(d) {
        for(size_t b = 0; b < d->bin_count; b++) {
            delete d->bins[b];
        }
        free(d->bins);
        free(d);
    }
}

void fdict_flush(fdict d) {
    for(size_t b = 0; b < d->bin_count; b++) {
        d->bins[b]->realloc(d->bin_size);
        d->bins[b]->pointer = 0;
    }
}

dict_element_t* fdict_get_all(fdict d) {
    dict_element_t* result = (dict_element_t*)malloc(sizeof(dict_element_t) * d->size);

    size_t i = 0;
    for(size_t b = 0; b < d->bin_count; b++) {
        for(size_t e = 0; e < d->bins[b]->pointer; e++) {
            printf("\rFound %lu elements", i);
            result[i].pair = d->bins[b]->data[e];
            result[i].bin = b;
            result[i++].element = e;
        }
    }

    printf("\nFound %lu elements in the dictionary of size %lu\n", i, d->size);
    assert(i == d->size);

    return result;
}

uint8_t* fdict_get(fdict d, __uint128_t k) {
    size_t bin = k % d->bin_count;
    for(size_t e = 0; e < d->bins[bin]->pointer; e++) {
        if(k == d->bins[bin]->data[e].pair.key) {
            d->bins[bin]->data[e].usage++;
            return d->bins[bin]->data[e].pair.value;
        }
    }
    return 0;
}

uint8_t fdict_exists(fdict d, __uint128_t k) {
    size_t bin = k % d->bin_count;
    for(size_t e = 0; e < d->bins[bin]->pointer; e++) {
        if(k == d->bins[bin]->data[e].pair.key) {
            d->bins[bin]->data[e].usage++;
            return 1;
        }
    }
    return 0;
}

void fdict_put(fdict d, __uint128_t k, uint8_t* value) {
    if(d->size >= d->max_element_count) {
        if(d->size <= d->flush_count) {
            fdict_flush(d);
            d->size = 0;
        }
        else {
            printf("Flushing dictionary of %lu elements from %lu\n", d->flush_count, d->size);
            dict_element_t* elements = fdict_get_all(d);
            // printf("Sorting dictionary\n");
            heapsort_dict(elements, d->size - 1);
            heapsort_dict_removal_order(elements, d->flush_count);
            // printf("Removing elements from dictionary\n");
            for(size_t e = 0; e < d->flush_count; e++) {
                // printf("\r%lu/%lu", e + 1, d->flush_count);
                d->bins[elements[e].bin]->pop(elements[e].element);
                
                // for(size_t et = e; et < d->flush_count; et++) 
                //     if(elements[et].bin == elements[e].bin && elements[et].element > elements[e].element) 
                //         elements[et].element--;
            }
            free(elements);
            d->size -= d->flush_count;
            // printf("\nFlushing complete\n");
        }
    }

    size_t bin = k % d->bin_count;
    d->bins[bin]->append(dict_usage_pair_t{k, value, 0});
    d->size++;
}

double fdict_load_factor(fdict d) { return (double)d->size / (double)d->bin_count; }