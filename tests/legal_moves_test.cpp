#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "legal_moves_test.hpp"
#include "../gameplay/reversi.h"
#include "../gameplay/walker.hpp"
#include "../utils/tarraylist.hpp"

uint8_t is_valid_coord(coord c, coord* accept) {
    for(coord* ct = accept; *ct; ct++) if(c->row == (*ct)->row && c->column == (*ct)->column) return 1;
    return 0;
}

void lm_test_initial() {
    board b = create_board(1, 8, 8);
    Arraylist<void*> coord_buff(65), coord_cache(1000);

    for(size_t c = 0; c < 1000; c++) coord_cache.append(create_coord(0, 0));
    find_next_boards(b, &coord_buff, &coord_cache);
    coord* cs = (coord*)coord_buff.data;
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
        free(*ct);
    }

    destroy_board(b);
    free(cs);

    for(coord* ct = acceptance; *ct; ct++) free(*ct);
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
