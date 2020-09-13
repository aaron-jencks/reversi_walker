#include "reversi.h"

#include <stdlib.h>
#include <err.h>

board create_board(uint8_t starting_player, uint8_t height, uint8_t width) {
    board b = calloc(1, sizeof(board_str));
    if(!b) err(1, "Memory Error while allocating the board\n");
    b->player = starting_player;
    b->height = height;
    b->width = width;

    b->board = calloc((height * width) >> 2, sizeof(char));
    if(!b->board) err(1, "Memory Error while allocating board's board array\n");

    /* <1 byte> 
     * +-+-+-+-+-+-+-+-+
     * | | | | | | | | | <-- 2 bytes | Byte 0,1
     * +-+-+-+-+-+-+-+-+
     * | | | | | | | | | <-- Byte 2,3
     * +-+-+-+-+-+-+-+-+
     * | | | | | | | | | <-- Byte 4,5
     * +-+-+-+-+-+-+-+-+
     * | | | |2|1| | | | <-- Byte 6,7
     * +-+-+-+-+-+-+-+-+
     * | | | |1|2| | | | <-- Byte 8,9
     * +-+-+-+-+-+-+-+-+
     * | | | | | | | | | <-- Byte 10,11
     * +-+-+-+-+-+-+-+-+
     * | | | | | | | | | <-- Byte 12,13
     * +-+-+-+-+-+-+-+-+
     * | | | | | | | | | <-- Byte 14,15
     * +-+-+-+-+-+-+-+-+
     */

    /** To create the starting position we need to fill the bits as such:
     * +--+--+--+--+--+--+--+--+
     * |00|00|00|10|01|00|00|00| <-- 2,64 for byte 6 and 7
     * +--+--+--+--+--+--+--+--+
     * |00|00|00|01|10|00|00|00| <-- 1,128 for byte 8 and 9
     * +--+--+--+--+--+--+--+--+
     */

    board_put(b, (height >> 1) - 1, (width >> 1) - 1, 2);
    board_put(b, (height >> 1) - 1, width >> 1, 1);
    board_put(b, height >> 1, (width >> 1) - 1, 1);
    board_put(b, height >> 1, width >> 1, 2);

    return b;
}

/**
 * @brief Create a board object, copying another board
 * 
 * @param board Board to clone
 * @return othelloboard 
 */
board clone_board(board b) {
    board bc = malloc(sizeof(board_str));

    if(!bc) err(1, "Memory Error Occured while allocating a board.");

    if(b) {
        bc->height = b->height;
        bc->width = b->width;
        bc->player = b->player;
        for(char i = 0; i < 16; i++) bc->board[i] = b->board[i];
    }
    else err(2, "Cannot clone an empty board pointer");

    return b;
}

void destroy_board(board b) {
    if(b) {
        if(b->board) free(b->board);
        free(b);
    }
}

// Transfer the boolean board values in and out of the bits in the struct
uint8_t board_get(board b, uint8_t row, uint8_t column) {
    /* column: 0 1 2 3 4 5 6 7 
     *        +-+-+-+-+-+-+-+-+
     * row 0: | | | | | | | | | <-- Byte 0,1
     *        +-+-+-+-+-+-+-+-+
     * row 1: | | | | | | | | | <-- Byte 2,3
     *        +-+-+-+-+-+-+-+-+
     * row 2: | | | | | | | | | <-- Byte 4,5
     *        +-+-+-+-+-+-+-+-+
     * row 3: | | | |2|1| | | | <-- Byte 6,7
     *        +-+-+-+-+-+-+-+-+
     * row 4: | | | |1|2| | | | <-- Byte 8,9
     *        +-+-+-+-+-+-+-+-+
     * row 5: | | | | | | | | | <-- Byte 10,11
     *        +-+-+-+-+-+-+-+-+
     * row 6: | | | | | | | | | <-- Byte 12,13
     *        +-+-+-+-+-+-+-+-+
     * row 7: | | | | | | | | | <-- Byte 14,15
     *        +-+-+-+-+-+-+-+-+
     */

    if(b) {
        uint8_t total_bit = (row * (b->width << 1)) + (column << 1), 
                byte = total_bit >> 3, 
                bit = (total_bit % 4) << 1;

        // (b'11' >> bit & board_value) >> (6 - bit)
        return ((192 >> bit) & b->board[byte]) >> (6 - bit);
    }
    return 3;
}

void board_put(board b, uint8_t row, uint8_t column, uint8_t player) {
    if(b) {
        uint8_t total_bit = (row * (b->width << 1)) + (column << 1), 
                byte = total_bit >> 3, 
                bit = (column % 4) << 1, 
                bph = 192 >> bit;

        // Reset the value of the bits to 0
        // byte (????????) | bph (00011000) = ????11???
        // ???11??? ^ 00011000 = ???00???
        b->board[byte] = (b->board[byte] | bph) ^ bph;

        // Insert your new value
        // byte (????????) | ((10|01)000000 >> bit) = ????(10|01)???
        if(player) b->board[byte] |= ((player == 1) ? 64 : 128) >> bit;
    }
}

uint8_t board_is_legal_move(board b, uint8_t row, uint8_t column) {
    if(b && row >= 0 && row < b->height && column >= 0 && column < b->width) {
        if(!board_get(b, row, column)) {

            // Check each of the 8 directions going out from the requested coordinate
            // Keep track of how many captures we have
            int8_t counts = 0, cr, cc, count, bv;
            for(int8_t rd = -1; rd < 2; rd++) {
                for(int8_t cd = -1; cd < 2; cd++) {
                    // Avoid infinite loop when rd=cd=0
                    if(!rd && !cd) continue;

                    // Take a step in the current direction
                    cr = row + rd;
                    cc = column + cd;

                    count = 0;
                    while(cr >= 0 && cr < b->height && cc >= 0 && cc < b->width) {
                        bv = board_get(b, cr, cc);
                        if(bv && bv != b->player) {
                            // There is a possible capture
                            count++;

                            // Take another step in the current direction
                            cr += rd;
                            cc += cd;

                            if((cr == b->height && rd) ||
                               (cr < 0 && rd == -1) ||
                               (cc == b->width && cd) ||
                               (cc < 0 && cd < 0)) {
                                   // We hit the edge of the board, this is not a capture
                                   count = 0;
                                   break;
                               }
                        }
                        else {
                            if(!bv)
                                // If we had any captures, they are in vain because our color isn't at the other end.
                                count = 0;
                            break;
                        }
                    }
                    counts += count;
                }
            }

            // Return true if we capture at least 1 piece
            return count > 0;
        }
    }

    // Either the board pointer was empty, or the space was already filled.
    return 0;
}

void board_place_piece(board b, uint8_t row, uint8_t column) {
    if(b && row >= 0 && row < b->height && column >= 0 && column < b->width) {
        // Check each of the 8 directions going out from the requested coordinate
        // flip any captures found
        int8_t counts = 0, cr, cc, count, bv;
        for(int8_t rd = -1; rd < 2; rd++) {
            for(int8_t cd = -1; cd < 2; cd++) {
                // Avoid infinite loop when rd=cd=0
                if(!rd && !cd) continue;

                // Take a step in the current direction
                cr = row + rd;
                cc = column + cd;

                count = 0;
                while(cr >= 0 && cr < b->height && cc >= 0 && cc < b->width) {
                    bv = board_get(b, cr, cc);
                    if(bv && bv != b->player) {
                        // There is a possible capture
                        count++;

                        // Take another step in the current direction
                        cr += rd;
                        cc += cd;

                        if((cr == b->height && rd) ||
                            (cr < 0 && rd == -1) ||
                            (cc == b->width && cd) ||
                            (cc < 0 && cd < 0)) {
                                // We hit the edge of the board, this is not a capture
                                count = 0;
                                break;
                            }
                    }
                    else {
                        if(!bv)
                            // If we had any captures, they are in vain because our color isn't at the other end.
                            count = 0;
                        break;
                    }
                }
                
                if(count > 0) {
                    cr = row + rd;
                    cc = column + cd;
                    bv = board_get(b, cr, cc);

                    while(bv && bv != b->player) {
                        board_put(b, cr, cc, b->player);
                        cr += rd;
                        cc += cd;
                        bv = board_get(b, cr, cc);
                    }
                }
            }
        }

        // Flip the player to the opponent
        b->player = (b->player == 1) ? 2 : 1;
    }
}