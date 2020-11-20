#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "board_placement.h"
#include "../reversi.h"

void board_placement_test() {
    board b = create_board(1, 255, 255);
    for(uint8_t r = 0; r < 255; r++) {
        for(uint8_t c = 0; c < 255; c++) {
            for(uint8_t v = 1; v < 3; v++) {
                board_put(b, r, c, v);
                assert(board_get(b, r, c) == v);
            }
        }
    }

    destroy_board(b);
}