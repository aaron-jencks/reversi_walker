#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "board_placement.cuh"
#include "../gameplay/reversi.h"
#include "../cuda/reversi.cuh"

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

__host__ void cuda_board_placement_test() {
    board b = create_board_cuda(1, 255, 255); 
    board_str bc;
    
    for(uint8_t r = 0; r < 255; r++) {
        for(uint8_t c = 0; c < 255; c++) {
            for(uint8_t v = 1; v < 3; v++) {
                board_put_cuda(b, r, c, v);
                assert(board_get_cuda(b, r, c) == v);
            }
        }
    }

    // Testing cloning functionality
    clone_into_board_cuda(&bc, b);

    for(uint8_t r = 0; r < 255; r++) {
        for(uint8_t c = 0; c < 255; c++) {
            assert(board_get_cuda(b, r, c) == 2);
        }
    }

    destroy_board_cuda(b);
}