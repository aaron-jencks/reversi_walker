#include <stdio.h>
#include <stdlib.h>
#include <err.h>

#include "reversi.h"
#include "hashtable.h"
#include "lookup3.h"
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


__uint128_t board_hash(void* brd) {
    if(brd) {
        board b = (board)brd;

        __uint128_t result = 0;

        result += b->player;
        result = result << 2;

        // You can fit 2 spaces in 3 bits if you really try,
        // so on an 8x8 board, 
        // we end up only using 96 bits instead of the entire 128.
        // well, 98 since we are including the player now
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c += 2) {
                uint8_t s1 = board_get(b, r, c), 
                        s2 = board_get(b, r, c + 1);

                result += (!s1 && !s2) ? 4 : 
                          (s1 == 1 && s2 == 1) ? 3 : 
                          (s1 == 2 && s2 == 2) ? 0 : 
                          (s1 == 2) ? 1 : 2;

                result = result << 3;
            }
        }

        // we still need to use the entire 128 bits though,
        // because the hashing algorithm works best in powers of 2
        uint32_t upperupper, upperlower, lowerupper, lowerlower;
        hashlittle2(&result, 8, &upperupper, &upperlower);
        hashlittle2(((char*)&result) + 8, 8, &lowerupper, &lowerlower);
        result = 0;
        result += upperupper;
        result = result << 32;
        result += upperlower;
        result = result << 32;
        result += lowerupper;
        result = result << 32;
        result += lowerlower;

        return result;
    }
    return 0;
}



int main() {
    board b = create_board(1, 8, 8), bc = create_board(2, 8, 8);
    // cache_index ci = board_index_8(b);
    linkedlist coords = create_ll();
    append_ll(coords, b);
    append_ll(coords, bc);
    hashtable cache = create_hashtable(1000000, &board_hash);

    printf("Starting walk...\n");

    uint64_t count = 0;
    while(coords->size) {
        board cb = pop_back_ll(coords);

        coord* next_moves = find_next_boards(cb);
        for(char im = 0; next_moves[im]; im++) {
            coord m = next_moves[im];
            board ccb = clone_board(cb);
            // printf("Before placement\n");
            // display_board(ccb);
            board_place_piece(ccb, m->row, m->column);
            // printf("After placement\n");
            // display_board(ccb);

            if(!exists_hs(cache, ccb)) {
                put_hs(cache, ccb);
                append_ll(coords, ccb);
            }

            free(m);
        }
        free(next_moves);

        printf("\rFinished %ld boards", count++);
    }

    printf("There are %ld possible board states\n", count);
}