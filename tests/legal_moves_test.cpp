#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "legal_moves_test.hpp"
#include "../gameplay/reversi.h"
#include "../gameplay/walker.hpp"
#include "../utils/tarraylist.hpp"

bool is_valid_coord(coord_t c, coord_t* accept, size_t na) {
    for(size_t ai = 0; ai < na; ai++) {
        coord_t ct = accept[ai];
        if(c.row == ct.row && c.column == ct.column) return 1;
    }
    return 0;
}

void lm_test_initial() {
    board_t b = create_board(1, 8, 8);
    Arraylist<coord_t> coord_buff(65);

    find_next_boards(b, &coord_buff);
    coord_t* cs = (coord_t*)coord_buff.data;
    coord_t acceptance[4] = {
        create_coord(3, 2),
        create_coord(2, 3),
        create_coord(5, 4),
        create_coord(4, 5)
    };

    for(size_t i = 0; i < coord_buff.pointer; i++) {
        coord_t ct = coord_buff.data[i];
        printf("Testing coordinate (%u, %u)\n", ct.row, ct.column);
        assert(is_valid_coord(ct, acceptance, 4));
    }

    destroy_board(b);
}

// void lm_test_from_coord() {
//     board b = create_board(2, 8, 8);
//     board_place_piece(b, 5, 3);
//     coord c = create_coord(5, 3);

//     coord* cs = find_next_boards_from_coord(b, c);
//     coord acceptance[3] = {
//         create_coord(5, 2),
//         create_coord(5, 4),
//         0
//     };

//     for(coord* ct = cs; *ct; ct++) {
//         printf("Testing coordinate (%u, %u)\n", (*ct)->row, (*ct)->column);
//         assert(is_valid_coord(*ct, acceptance));
//         free(*ct);
//     }

//     destroy_board(b);
//     free(cs);
//     free(c);

//     for(coord* ct = acceptance; *ct; ct++) free(*ct);
// }
