#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "tests/capturecounts_test.h"
#include "tests/legal_moves_test.h"
#include "tests/board_placement.h"

uint64_t test_count = 4;
void (*tests[4])() = {
    cc_test_directions,
    lm_test_initial,
    lm_test_from_coord,
    board_placement_test
};


int main() {
    for(uint64_t t = 0; t < test_count; t++) tests[t]();
}