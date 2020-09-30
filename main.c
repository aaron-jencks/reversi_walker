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
#include "valid_moves.h"

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
    
    // Setup the stacks
    uint16_arraylist search_stack = create_uint16_arraylist(1000);
    uint8_arraylist parent_index_count_stack = create_uint8_arraylist(1000);
    uint64_arraylist moves_white = create_uint64_arraylist(1000), moves_black = create_uint64_arraylist(1000);

    // Populate the stacks with their initial data
    // Populate the moves_white stack
    coord* next_moves = find_next_boards(b);

    uint64_t wmoves = 0;
    for(char im = 0; next_moves[im]; im++) {
        coord m = next_moves[im];
        uint16_t sm = coord_to_short(m);
        append_sal(search_stack, sm);
        encode_valid_position(wmoves, m->row, m->column);
    }
    for(char im = 0; next_moves[im]; im++) append_dal(moves_white, wmoves);
    append_cal(parent_index_count_stack, 4);

    free(next_moves);

    // Populate the moves_black stack
    b->player = 2;

    next_moves = find_next_boards(b);

    wmoves = 0;
    for(char im = 0; next_moves[im]; im++) {
        coord m = next_moves[im];
        uint16_t sm = coord_to_short(m);
        append_sal(search_stack, sm);
        encode_valid_position(wmoves, m->row, m->column);
    }
    for(char im = 0; next_moves[im]; im++) append_dal(moves_black, wmoves);
    append_cal(parent_index_count_stack, 4);

    free(next_moves);

    // Reset the player and create the cache
    b->player = 1;

    hashtable cache = create_hashtable(1000000, &board_hash);

    printf("Starting walk...\n");

    uint64_t count = 0, iter = 0;
    while(search_stack->pointer) {
        uint8_t ccount = --parent_index_count_stack->data[parent_index_count_stack->pointer - 1];
        uint64_t moves = pop_back_dal((b->player == 1) ? moves_white : moves_black), movesc;
        uint16_t sm = pop_back_sal(search_stack);
        coord m = short_to_coord(sm);

        display_board(b);
        printf("player is %u, move is (%u, %u)\n", b->player, m->row, m->column);
        uint64_t cc = board_place_piece(b, m->row, m->column);
        printf("player is %u\n", b->player);

        if(cc && !exists_hs(cache, b)) {

            uint64_arraylist mlist = (b->player == 1) ? moves_white : moves_black;

            next_moves = find_next_boards_from_coord(b, m);

            if(next_moves) {
                // Retrieve the parent valid moves
                movesc = mlist->data[mlist->pointer - 1];

                // Encode the new valid moves
                for(char im = 0; next_moves[im]; im++) {
                    coord mm = next_moves[im];
                    encode_valid_position(movesc, mm->row, mm->column);
                    free(mm);
                }

                free(next_moves);

                // Find all of the coordinates
                next_moves = retrieve_all_valid_positions(movesc);

                // If the move is legal, then append it to the search stack
                for(char im = 0; next_moves[im]; im++) {
                    coord mm = next_moves[im];
                    if(board_is_legal_move(b, mm->row, mm->column)) append_sal(search_stack, coord_to_short(mm));
                    append_dal(mlist, movesc);
                    free(mm);
                }

                free(next_moves);
            }
            else {
                // There were no valid moves for this color, switch to the other color and see if that works
                printf("No moves for current player, trying opponent\n");
                b->player = (b->player == 1) ? 2 : 1;

                mlist = (b->player == 1) ? moves_white : moves_black;

                // Retrieve the parent valid moves
                movesc = mlist->data[mlist->pointer - 1];

                // Find all of the coordinates
                next_moves = retrieve_all_valid_positions(movesc);

                if(next_moves) {
                    // If the move is legal, then append it to the search stack
                    for(char im = 0; next_moves[im]; im++) {
                        coord mm = next_moves[im];
                        if(board_is_legal_move(b, mm->row, mm->column)) append_sal(search_stack, coord_to_short(mm));
                        append_dal(mlist, movesc);
                        free(mm);
                    }

                    free(next_moves);
                }
                else {
                    // Neither player has any moves the game has ended
                    if(!ccount) {
                        while(!ccount) {
                            printf("Popped off a child, switching players\n");
                            pop_back_cal(parent_index_count_stack);
                            ccount = parent_index_count_stack->data[parent_index_count_stack->pointer - 1];
                            b->player = (b->player == 1) ? 2 : 1;
                        }
                    }
                    count++;
                }
            }
        }
        else if(!cc) {
            // There were no valid moves for this color, switch to the other color and see if that works
            printf("No captures for current player, trying opponent\n");
            // b->player = (b->player == 1) ? 2 : 1;    // This was already done by place piece

            uint64_arraylist mlist = (b->player == 1) ? moves_white : moves_black;

            // Retrieve the parent valid moves
            movesc = mlist->data[mlist->pointer - 1];

            // Find all of the coordinates
            next_moves = retrieve_all_valid_positions(movesc);

            if(next_moves) {
                // If the move is legal, then append it to the search stack
                for(char im = 0; next_moves[im]; im++) {
                    coord mm = next_moves[im];
                    if(board_is_legal_move(b, mm->row, mm->column)) append_sal(search_stack, coord_to_short(mm));
                    append_dal(mlist, movesc);
                    free(mm);
                }

                free(next_moves);
            }
            else {
                // Neither player has any moves the game has ended
                if(!ccount) {
                    while(!ccount) {
                        printf("Popped off a child, switching players\n");
                        pop_back_cal(parent_index_count_stack);
                        ccount = parent_index_count_stack->data[parent_index_count_stack->pointer - 1];
                        b->player = (b->player == 1) ? 2 : 1;
                    }
                }
                count++;
            }
        }
        else {
            if(!ccount) {
                while(!ccount) {
                    printf("Popped off a child, switching players\n");
                    pop_back_cal(parent_index_count_stack);
                    ccount = parent_index_count_stack->data[parent_index_count_stack->pointer - 1];
                    b->player = (b->player == 1) ? 2 : 1;
                }
            }
        }

        free(m);

        printf("\rFinished %ld boards, iteration: %ld", count, iter++);
        fflush(stdout);
    }

    printf("There are %ld possible board states\n", count);
}