#include "fdict.h"
#include "heapsort.h"

#include <stdlib.h>
#include <err.h>

fdict create_fixed_size_dictionary(size_t max_element_count, size_t flush_count) {
    fdict d = malloc(sizeof(fdict_t));
    if(!d) err(1, "Memory error while allocating fixed dictionary\n");

    d->bins = malloc(sizeof(uint128_arraylist) * INITIAL_BIN_COUNT);
    d->ptrs = malloc(sizeof(ptr_arraylist) * INITIAL_BIN_COUNT);
    d->usage_counters = malloc(sizeof(uint64_arraylist) * INITIAL_BIN_COUNT);
    if(!(d->bins && d->ptrs && d->usage_counters)) err(1, "Memory error while allocating fixed dictionary\n");

    for(size_t b = 0; b < INITIAL_BIN_COUNT; b++) {
        d->bins[b] = create_uint128_arraylist(INITIAL_BIN_SIZE);
        d->ptrs[b] = create_ptr_arraylist(INITIAL_BIN_SIZE);
        d->usage_counters[b] = create_uint64_arraylist(INITIAL_BIN_SIZE);
    }

    d->bin_count = INITIAL_BIN_COUNT;
    d->max_element_count = max_element_count;
    d->flush_count = flush_count;
    d->size = 0;

    return d;
}

void destroy_fixed_dictionary(fdict d) {
    if(d) {
        for(size_t b = 0; b < d->bin_count; b++) {
            destroy_uint128_arraylist(d->bins[b]);
            destroy_ptr_arraylist(d->ptrs[b]);
            destroy_uint64_arraylist(d->usage_counters[b]);
        }
        free(d->bins);
        free(d->ptrs);
        free(d->usage_counters);
        free(d);
    }
}

void fdict_flush(fdict d) {
    for(size_t b = 0; b < INITIAL_BIN_COUNT; b++) {
        realloc_ddal(d->bins[b], INITIAL_BIN_SIZE);
        d->bins[b]->pointer = 0;
        realloc_pal(d->ptrs[b], INITIAL_BIN_SIZE);
        d->ptrs[b]->pointer = 0;
        realloc_dal(d->usage_counters[b], INITIAL_BIN_SIZE);
        d->usage_counters[b]->pointer = 0;
    }
}

dict_element_t* fdict_get_all(fdict d) {
    dict_element_t* result = malloc(sizeof(dict_element_t) * d->size);

    size_t i = 0;
    for(size_t b = 0; b < INITIAL_BIN_COUNT; b++) {
        for(size_t e = 0; e < d->bins[b]->pointer; e++) {
            result[i].pair.pair.key = d->bins[b]->data[e];
            result[i].pair.pair.value = d->ptrs[b]->data[e];
            result[i].bin = b;
            result[i].element = e;
            result[i++].pair.usage = d->usage_counters[b]->data[e];
        }
    }

    return result;
}

uint8_t* fdict_get(fdict d, __uint128_t k) {
    size_t bin = k % d->bin_count;
    for(size_t e = 0; e < d->bins[bin]->pointer; e++) {
        if(k == d->bins[bin]->data[e]) {
            d->usage_counters[bin]->data[e]++;
            return d->ptrs[bin]->data[e];
        }
    }
    return 0;
}

void fdict_put(fdict d, __uint128_t k, uint8_t* value) {
    if(d->size++ >= d->max_element_count) {
        if(d->size <= d->flush_count) {
            fdict_flush(d);
            d->size = 0;
        }
        else {
            dict_element_t* elements = fdict_get_all(d);
            heapsort_dict(elements, d->size);
            for(size_t e = 0; e < d->flush_count; e++) {
                pop_ddal(d->bins[elements[e].bin], elements[e].element);
                pop_pal(d->ptrs[elements[e].bin], elements[e].element);
                pop_dal(d->usage_counters[elements[e].bin], elements[e].element);

                for(size_t et = e; et < d->flush_count; et++) 
                    if(elements[et].bin == elements[e].bin && elements[et].element > elements[e].element) 
                        elements[et].element--;
            }
            free(elements);
            d->size -= d->flush_count;
        }
    }

    size_t bin = k % d->bin_count;
    append_ddal(d->bins[bin], k);
    append_pal(d->ptrs[bin], value);
    append_dal(d->usage_counters[bin], 0);
    d->size++;
}

double fdict_load_factor(fdict d) { return (double)d->size / (double)d->bin_count; }