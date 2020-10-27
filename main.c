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

// TODO you can use the previous two board states to predict the next set of valid moves.

/**
 * I had to get rid of the stacks because I had no way to keep track of when a move generates new moves for BOTH colors .W.
 * or when a capture generates new moves
 * 
 * ie.
 * 
 * 00000000
 * 00000000
 * 00012000
 * 00021000
 * 00000000
 * 00000000
 * 
 * --->
 * 
 * 00000000
 * 00001000
 * 00011000
 * 00021000
 * 00x00000 <-- Where x is a new move generated for white by a captured piece -___-
 * 00000000
 * 
 * I need to optimize the linkedlists in the hashtable, and optimize the memory usage of the DFS stack by getting rid of the pointers.
 * 
 * Re work the hashtable so that it doesn't error out when the size gets too big.
 * 
 */


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


void display_capture_counts(uint64_t cc) {
    /*
    * 0: upper-left
    * 1: up
    * 2: upper-right
    * 3: left
    * 4: right
    * 5: lower-left
    * 6: lower
    * 7: lower-right
    */
    printf("Capture Counts:\n");
    uint8_t c;
    for(uint8_t i = 0; i < 8; i++) {
        c = capture_count_get_count(cc, i);
        switch(i) {
            case 0:
                printf("\tNorthwest: ");
                break;
            case 1:
                printf("\tNorth: ");
                break;
            case 2:
                printf("\tNortheast: ");
                break;
            case 3:
                printf("\tWest: ");
                break;
            case 4:
                printf("\tEast: ");
                break;
            case 5:
                printf("\tSouthwest: ");
                break;
            case 6:
                printf("\tSouth: ");
                break;
            case 7:
                printf("\tSoutheast: ");
                break;
        }
        printf("%u\n", c);
    }
    printf("\n");
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
        uint32_t upperupper = 0, upperlower = 0, lowerupper = 0, lowerlower = 0;
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


// TODO make work on a 6x6
// Modify to copy boards, instead of 
int main() {
    board b = create_board(1, 6, 6); // , bc = create_board(2, 8, 8);
    
    // Setup the stacks
    ptr_arraylist search_stack = create_ptr_arraylist(1000);
    // uint16_arraylist coord_pairs = create_uint16_arraylist(1000);
    // uint64_arraylist moves_white = create_uint64_arraylist(1000), moves_black = create_uint64_arraylist(1000), overrides = create_uint64_arraylist(1000);

    // Populate the stacks with their initial data
    // Populate the moves_white stack
    coord* next_moves = find_next_boards(b);

    uint64_t wmoves = 0;
    for(char im = 0; next_moves[im]; im++) {
        coord m = next_moves[im];
        uint16_t sm = coord_to_short(m);
        board cb = clone_board(b);
        board_place_piece(cb, m->row, m->column);
        append_pal(search_stack, cb);
        // append_sal(coord_pairs, sm);
        // wmoves = encode_valid_position(wmoves, m->row, m->column);
        free(m);
    }
    // for(char im = 0; next_moves[im]; im++) append_dal(moves_white, wmoves);

    free(next_moves);

    // Populate the moves_black stack
    b->player = 2;

    next_moves = find_next_boards(b);

    wmoves = 0;
    for(char im = 0; next_moves[im]; im++) {
        coord m = next_moves[im];
        uint16_t sm = coord_to_short(m);
        board cb = clone_board(b);
        board_place_piece(cb, m->row, m->column);
        append_pal(search_stack, cb);
        // append_sal(coord_pairs, sm);
        wmoves = encode_valid_position(wmoves, m->row, m->column);
        free(m);
    }
    // for(char im = 0; next_moves[im]; im++) append_dal(moves_black, wmoves);

    free(next_moves);

    // Reset the player and create the cache
    // b->player = 1;

    hashtable cache = create_hashtable(1000000, &board_hash);

    printf("Starting walk...\n");

    uint64_t count = 0, iter = 0, intercap = 0;
    while(search_stack->pointer) {
        // uint64_t moves = pop_back_dal((b->player == 1) ? moves_white : moves_black), movesc, movescc, movesccc;
        board sb = pop_back_pal(search_stack), bc;

        #ifdef debug
            display_board(sb);
        #endif

        // uint64_arraylist mlist = (b->player == 1) ? moves_white : moves_black;  // TODO Needs to be the opposing color

        // if(!overrides->pointer) movesc = pop_back_dal(mlist);
        // else movesc = pop_back_dal(overrides);

        // next_moves = retrieve_all_valid_positions(movesc);

        next_moves = find_next_boards(sb);

        if(next_moves[0]) {
            uint8_t move_count = 0;
            // If the move is legal, then append it to the search stack
            for(uint8_t im = 0; next_moves[im]; im++) {
                coord mm = next_moves[im];
                bc = clone_board(sb);
                // movescc = movesc;   // TODO needs to be opposite color

                if(board_is_legal_move(bc, mm->row, mm->column)) {
                    board_place_piece(bc, mm->row, mm->column);
                    append_pal(search_stack, bc);
                    move_count++;

                    // // Figure out if the current move also generated new valid moves for the current player
                    // b->player = (b->player == 1) ? 2 : 1;
                    // mlist = (b->player == 1) ? moves_white : moves_black;
                    // movesccc = pop_back_dal(mlist); // TODO needs to be the current color

                    // coord* next_moves_1 = find_next_boards_from_coord_opposing_player(b, mm);

                    // // Encode the new valid moves
                    // for(char im = 0; next_moves_1[im]; im++) {
                    //     coord mmm = next_moves_1[im];
                    //     movesccc = encode_valid_position(movesccc, mmm->row, mmm->column);
                    //     free(mmm);
                    // }

                    // if(movesccc != movesc) append_dal(overrides, movesccc);

                    // free(next_moves_1);

                    // // Find the valid moves that were generated from this move
                    // movescc = find_valid_positions_from_coord(
                    //     remove_valid_position(movescc, mm->row, mm->column), 
                    //     bc, mm->row, mm->column);
                    // append_dal(mlist, movescc); //  TODO needs to be opposing color
                }
                else {
                    destroy_board(bc);
                }

                free(mm);
            }

            #ifdef debug
                printf("Found %u moves\n", move_count);
            #endif

            free(next_moves);
        }
        else {
            // The opponenet has no moves, try the other player
            #ifdef debug
                printf("No moves for opponent, switching back to the current player\n");
            #endif
            sb->player = (sb->player == 1) ? 2 : 1;

            free(next_moves);
            next_moves = find_next_boards(sb);

            if(next_moves[0]) {
                uint8_t move_count = 0;
                // If the move is legal, then append it to the search stack
                for(uint8_t im = 0; next_moves[im]; im++) {
                    coord mm = next_moves[im];
                    bc = clone_board(sb);
                    // movescc = movesc;   // TODO needs to be opposite color

                    if(board_is_legal_move(bc, mm->row, mm->column)) {
                        board_place_piece(bc, mm->row, mm->column);
                        append_pal(search_stack, bc);
                        move_count++;

                        // // Figure out if the current move also generated new valid moves for the current player
                        // b->player = (b->player == 1) ? 2 : 1;
                        // mlist = (b->player == 1) ? moves_white : moves_black;
                        // movesccc = pop_back_dal(mlist); // TODO needs to be the current color

                        // coord* next_moves_1 = find_next_boards_from_coord_opposing_player(b, mm);

                        // // Encode the new valid moves
                        // for(char im = 0; next_moves_1[im]; im++) {
                        //     coord mmm = next_moves_1[im];
                        //     movesccc = encode_valid_position(movesccc, mmm->row, mmm->column);
                        //     free(mmm);
                        // }

                        // if(movesccc != movesc) append_dal(overrides, movesccc);

                        // free(next_moves_1);

                        // // Find the valid moves that were generated from this move
                        // movescc = find_valid_positions_from_coord(
                        //     remove_valid_position(movescc, mm->row, mm->column), 
                        //     bc, mm->row, mm->column);
                        // append_dal(mlist, movescc); //  TODO needs to be opposing color
                    }
                    else {
                        destroy_board(bc);
                    }

                    free(mm);
                }

                free(next_moves);
            }
            else {
                // The opponenet has no moves, try the other player
                #ifdef debug
                    printf("No moves for anybody, game has ended.\n");
                #endif

                free(next_moves);

                if(!exists_hs(cache, sb)) {
                    put_hs(cache, sb);
                    count++;
                }
                else {
                    #ifdef debug
                        printf("The given board is already counted\n");
                    #endif
                }
            }
        }

        destroy_board(sb);

        printf("\rFinished %ld boards, iteration: %ld", count, iter++);
        fflush(stdout);
    }

    printf("\nThere are %ld possible board states\n", count);
}