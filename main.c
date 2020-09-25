#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <err.h>

#include "reversi.h"
#include "hashtable.h"
#include "lookup3.h"
#include "walker.h"
#include "ll.h"
#include "arraylist.h"

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
    board b = create_board(1, 8, 8); // , bc = create_board(2, 8, 8);
    
    // TODO Convert the moves lists into uint64_t bit strings
    uint16_arraylist search_stack = create_uint16_arraylist(1000);
    uint64_arraylist previous_caps = create_uint64_arraylist(1000); // Keeps track of which pieces where captured when, for easy undoing during upward traversal
    uint8_arraylist parent_index_count_stack = create_uint8_arraylist(1000);
    linkedlist moves_white = create_ll(), moves_black = create_ll(); // Used to keep track of moves that were discovered in previous moves
               // previous_caps = create_ll(); // Keeps track of which pieces where captured when, for easy undoing during upward traversal

    coord* next_moves = find_next_boards(b);

    uint8_t move_count = 0;
    for(char im = 0; next_moves[im]; im++) {
        coord m = next_moves[im];
        append_ll(search_stack, m);
        append_ll(moves_white, m);
        move_count++;
    }
    append_cal(parent_index_count_stack, move_count);

    free(next_moves);

    // append_ll(search_stack, b);
    // append_ll(search_stack, bc);
    hashtable cache = create_hashtable(1000000, &board_hash);

    printf("Starting walk...\n");

    uint64_t count = 0;
    while(search_stack->size) {
        uint16_t sm = pop_back_sal(search_stack);
        coord m = short_to_coord(sm);

        uint64_t cc = board_place_piece(b, m->row, m->column);
        parent_index_count_stack->data[parent_index_count_stack->pointer - 1]--;

        if(!exists_hs(cache, b)) {
            append_dal(previous_caps, cc);

            coord* next_moves = find_next_boards_from_coord(b, m);

            move_count = 0;
            for(char im = 0; next_moves[im]; im++) {
                coord mm = next_moves[im];
                append_ll((b->player == 1) ? moves_white : moves_black, mm); // TODO Perform existence check
                // append_sal(search_stack, coord_to_short(mm));
                move_count++;
            }

            if(move_count) append_cal(parent_index_count_stack, move_count);

            // TODO Append all the current legal moves that aren't the current coordinate onto the search stack

            // If the parent_index_count is 0, then we need to pop off the that parent index, as well as the parent available moves.

            free(next_moves);
        }
        else {
            // TODO Undo move
        }

        free(m);

        // // TODO Use the previous_caps stack to make the move_finding algorithm faster
        // // Remember to eliminate non-legal moves from the stack as you go, 
        // // or not, 
        // // how will you handle that when traversing back up?
        // // Just keep the valid moves and check them all every time you play a piece to see if they're valid
        // coord* next_moves = find_next_boards(cb);
        // uint8_t move_count = 0;
        // for(char im = 0; next_moves[im]; im++) {
        //     coord m = next_moves[im];
        //     board ccb = clone_board(cb);
        //     // printf("Before placement\n");
        //     // display_board(ccb);
        //     capture_count cc = board_place_piece(ccb, m->row, m->column);
        //     // printf("After placement\n");
        //     // display_board(ccb);

        //     if(!exists_hs(cache, ccb)) {
        //         put_hs(cache, ccb);
        //         append_ll(search_stack, ccb);
        //         append_ll(previous_caps, cc);
        //         move_count++;
        //         // TODO Find valid moves for the opposite color, and the current color, based on the newly place piece, then add them to the moves stack.
        //     }

        //     free(m);
        // }
        // free(next_moves);

        // TODO if there are no moves, then pop from the previous_caps until its size matches the size of the search_stack and undo each move
        // that way we can use coords instead of boards in the search stack and just keep a singular board in memory at all times
        // this will reduce the memory cost in the search_stack from 64 bits, down to 16 bits

        // TODO if there are no moves, then we need to start removing the invalid moves from the moves stack as we reverse the board

        printf("\rFinished %ld boards", count++);
    }

    printf("There are %ld possible board states\n", count);
}