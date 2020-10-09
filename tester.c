#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "tests/capturecounts_test.h"

uint64_t test_count = 1;
void (*tests[1])() = {
    cc_test_directions
};


int main() {
    for(uint64_t t = 0; t < test_count; t++) tests[t]();
}