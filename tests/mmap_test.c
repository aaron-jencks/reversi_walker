#include "../mem_man/heir.h"
#include "../mem_man/mmap_man.h"
#include "mmap_test.h"

#include <stdio.h>
#include <assert.h>
#include <err.h>


void mmap_test_readback() {
    heirarchy h = create_heirarchy();
    printf("Testing insertion of a new bit\n");
    assert(heirarchy_insert(h, 1));
    printf("Testing duplicate insertion\n");
    assert(!heirarchy_insert(h, 1));
    destroy_heirarchy(h);
}