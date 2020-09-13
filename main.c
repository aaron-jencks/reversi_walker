#include <stdio.h>
#include <stdlib.h>
#include <err.h>

#include "reversi.h"
#include "cache.h"
#include "walker.h"
#include "ll.h"

void display_board(board b) {
    if(b) {
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c++) {
                printf("%c", board_get(b, r, c) + '0');
            }
            printf("\n");
        }
        printf("\n");
    }
}

cache_index board_index_8(board b) {
    cache_index ci = malloc(sizeof(cache_index_str));
    if(!ci) err(1, "Memory Error while allocating cache index\n");

    ci->player = b->player == 2;

    // uint8_t counter = 0;
    // ci->lower = 0;
    // ci->upper = 0;
    // for(uint8_t r = 0; r < b->height; r++) {
    //     for(uint8_t c = 0; c < b->width; c++) {
    //         if(counter++ < 32) {
    //             ci->lower += board_get(b, r, c);
    //             ci->lower = ci->lower << 2;
    //         }
    //         else {
    //             ci->upper += board_get(b, r, c);
    //             ci->upper = ci->upper << 2;
    //         }
    //     }
    // }

    // ci->lower -= 576;

    display_board(b);

    ci->index = 0;
    for(uint8_t r = 0; r < b->height; r++) {
        for(uint8_t c = 0; c < b->width; c++) {
            ci->index += board_get(b, r, c);
            ci->index = ci->index << 2;
            printf("%lld, ", ci->index);
        }
        printf("\n");
    }

    ci->index -= 432345564227567040;

    printf("%lld\n", ci->index);

    return ci;
}

// cache_index board_index_6(board b) {
//     cache_index ci = malloc(sizeof(cache_index_str));
//     if(!ci) err(1, "Memory Error while allocating cache index\n");

//     ci->player = b->player == 2;

//     uint8_t counter = 0;
//     ci->lower = 0;
//     ci->upper = 0;
//     for(uint8_t r = 0; r < b->height; r++) {
//         for(uint8_t c = 0; c < b->width; c++) {
//             if(counter++ < 32) {
//                 ci->lower += board_get(b, r, c);
//                 ci->lower = ci->lower << 2;
//             }
//             else {
//                 ci->upper += board_get(b, r, c);
//                 ci->upper = ci->upper << 2;
//             }
//         }
//     }

//     ci->lower -= 144;

//     return ci;
// }

int main() {
    board b = create_board(1, 8, 8), bc = create_board(2, 8, 8);
    // cache_index ci = board_index_8(b);
    linkedlist coords = create_ll();
    append_ll(coords, b);
    append_ll(coords, bc);
    bit_cache cache = create_bit_cache(2, 64);

    printf("Starting walk...\n");

    uint64_t count = 0;
    while(coords->size) {
        board cb = pop_back_ll(coords);

        coord* next_moves = find_next_boards(cb);
        for(char im = 0; next_moves[im]; im++) {
            coord m = next_moves[im];
            board ccb = clone_board(cb);
            printf("Before placement\n");
            display_board(ccb);
            board_place_piece(ccb, m->row, m->column);
            printf("After placement\n");
            display_board(ccb);

            cache_index ci = board_index_8(ccb);

            if(conditional_insert_bit(cache, ci))
                append_ll(coords, ccb);
            else
                destroy_board(ccb);

            free(ci);
            free(m);
        }
        free(next_moves);

        printf("\rFinished %ld boards", count++);
    }

    printf("There are %ld possible board states\n", count);
}