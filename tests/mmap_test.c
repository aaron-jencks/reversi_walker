#include "../mem_man/heir.h"
#include "../mem_man/mmap_man.h"
#include "mmap_test.h"

#include <stdio.h>
#include <assert.h>
#include <err.h>


void mmap_test_readback() {
    heirarchy h = create_heirarchy();
    printf("Heirarchy statistics:\nBits per level: %lu\nNumber of bits in the final level: %lu\nNumber of levels: %lu\n", h->num_bits_per_level, h->num_bits_per_final_level, h->num_levels);

    for(size_t b = 0; b < 128; b++) {
        __uint128_t k = ((__uint128_t)1) << b;
        printf("Testing insertion of a new bit %lu\n", b);
        assert(heirarchy_insert(h, k));
        printf("Testing duplicate insertion\n");
        assert(!heirarchy_insert(h, k));
    }
    destroy_heirarchy(h);
}