#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "../gameplay/reversi.h"
#include "capturecounts_test.h"

void cc_test_directions() {
    uint64_t count = 0;
    for(int i = 0; i < 8; i++) {
        printf("Testing direction %d\n", i);
        count = capture_count_put_count(count, i, 1);
        uint8_t c = capture_count_get_count(count, i);
        assert(c == 1);
    }
}