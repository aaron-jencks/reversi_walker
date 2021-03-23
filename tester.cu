#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "tests/capturecounts_test.h"
#include "tests/legal_moves_test.cuh"
#include "tests/board_placement.cuh"
#include "tests/mempage_test.h"
#include "tests/mmap_test.hpp"
#include "tests/dict_test.hpp"
#include "tests/heapsort_test.h"
#include "tests/arraylist_test.hpp"

const uint64_t test_count = 13;
void (*tests[13])() = {
    cc_test_directions,
    lm_test_initial,
    lm_cuda_test,
    board_placement_test,
    cuda_board_placement_test,
    heapsort_test,
    arr_insertion_test,
    arr_deletion_test,
    // mp_test_index,
    // mp_test_clear,
    // mp_test_realloc,
    // mp_buff_test_index,
    // fio_test_hashtable_write,
    fdict_purge_test,
    hdict_rehash_test,
    mmap_test_readback,
    mmap_bin_test,
    mmap_spiral_hash_test
};


int main() {
    for(uint64_t t = 0; t < test_count; t++) tests[t]();
}