#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "tests/capturecounts_test.h"
#include "tests/legal_moves_test.h"
#include "tests/board_placement.h"
#include "tests/mempage_test.h"
#include "tests/fileio_test.h"
#include "tests/mmap_test.h"

const uint64_t test_count = 11;
void (*tests[11])() = {
    cc_test_directions,
    lm_test_initial,
    lm_test_from_coord,
    board_placement_test,
    mp_test_index,
    mp_test_clear,
    mp_test_realloc,
    mp_buff_test_index,
    fio_test_hashtable_write,
    mmap_test_readback,
    mmap_spiral_hash_test
};


int main() {
    for(uint64_t t = 0; t < test_count; t++) tests[t]();
}