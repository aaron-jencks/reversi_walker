#include "dict_test.hpp"
#include "../utils/dictionary/dict_def.h"
#include "../utils/dictionary/fdict.hpp"
#include "../utils/dictionary/hdict.hpp"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void fdict_purge_test() {
    printf("Testing fixed dictionary purging mechanic\n");
    fdict d = create_fixed_size_dictionary(1500000, 1000000, 100000, 15);
    for(__uint128_t e = 0; e < ((1500000) << 1); e++) {
        printf("\rInserting value %lu %lu", ((uint64_t*)&e)[1], ((uint64_t*)&e)[0]);
        fdict_put(d, e, 0);
        assert(d->size <= 1500000);
        printf(" Load factor %.4f", fdict_load_factor(d));
    }
    destroy_fixed_dictionary(d);
}

void hdict_rehash_test() {
    printf("Testing rehashing dictionary rehashing mechanic\n");
    hdict d = create_rehashing_dictionary(1000, 15);
    for(__uint128_t e = 0; e < ((1000 * 1000) << 1); e++) {
        printf("\rInserting value %lu %lu", ((uint64_t*)&e)[1], ((uint64_t*)&e)[0]);
        hdict_put(d, e, 0);
        printf(" Load factor %.4f", hdict_load_factor(d));
        assert(hdict_load_factor(d) <= DICT_LOAD_LIMIT);
    }
    destroy_rehashing_dictionary(d);
}