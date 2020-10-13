#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "legal_moves_test.h"
#include "../reversi.h"
#include "../walker.h"

uint8_t is_valid_coord(coord c, coord* accept) {
    for(coord* ct = accept; *ct; ct++) if(c->row == (*ct)->row && c->column == (*ct)->column) return 1;
    return 0;
}

void lm_test_initial() {
    board b = create_board(1, 8, 8);
    coord* cs = find_next_boards(b);
    coord acceptance[5] = {
        create_coord(3, 2),
        create_coord(2, 3),
        create_coord(5, 4),
        create_coord(4, 5),
        0
    };

    for(coord* ct = cs; *ct; ct++) {
        printf("Testing coordinate (%u, %u)\n", (*ct)->row, (*ct)->column);
        assert(is_valid_coord(*ct, acceptance));
    }
}

void lm_test_from_coord() {
    board b = create_board(2, 8, 8);
    board_place_piece(b, 5, 3);

    coord* cs = find_next_boards_from_coord(b, create_coord(5, 3));
    coord acceptance[3] = {
        create_coord(5, 2),
        create_coord(5, 4),
        0
    };

    for(coord* ct = cs; *ct; ct++) {
        printf("Testing coordinate (%u, %u)\n", (*ct)->row, (*ct)->column);
        assert(is_valid_coord(*ct, acceptance));
    }

    destroy_board(b);
}
