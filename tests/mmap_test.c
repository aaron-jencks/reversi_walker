#include "../mem_man/heir.h"
#include "../mem_man/mmap_man.h"
#include "mmap_test.h"

#include <stdio.h>
#include <assert.h>
#include <err.h>


void mmap_test_readback() {
    heirarchy h = create_heirarchy();
    for(size_t b = 0; b < 128; b++) {
        __uint128_t k = 1 << b;
        printf("Testing insertion of a new bit\n");
        assert(heirarchy_insert(h, k));
        printf("Testing duplicate insertion\n");
        assert(!heirarchy_insert(h, k));
    }
    destroy_heirarchy(h);
}