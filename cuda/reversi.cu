#include "reversi.cuh"

/**
 * A Warp is a logical unit that shares a pc
 * Keep thread blocks as large as possible without increasing divergence
 */

__device__ uint8_t board_get_cuda(board b, uint8_t row, uint8_t column) {
    if(b) {
        uint8_t total_bit = (row * (b->width << 1)) + (column << 1), 
                byte = total_bit >> 3, 
                bit = total_bit % 8;

        return ((192 >> bit) & b->board[byte]) >> (6 - bit);
    }
    return 3;
}

__device__ void board_put_cuda(board b, uint8_t row, uint8_t column, uint8_t player) {
    if(b) {
        uint8_t total_bit = (row * (b->width << 1)) + (column << 1), 
                byte = total_bit >> 3, 
                bit = total_bit % 8, 
                bph = 192 >> bit;

        b->board[byte] = (b->board[byte] | bph) ^ bph;

        if(player) b->board[byte] |= ((player == 1) ? 64 : 128) >> bit;
    }
}

__global__ uint8_t board_is_legal_move_cuda(board b, uint8_t row, uint8_t column) {
    if(b && row >= 0 && row < b->height && column >= 0 && column < b->width) {
        if(!board_get_cuda(b, row, column)) {

            // Check each of the 8 directions going out from the requested coordinate
            // Keep track of how many captures we have
            int8_t counts = 0, cr, cc, count, bv, operating;
            for(int8_t rd = -1; rd < 2; rd++) {
                for(int8_t cd = -1; cd < 2; cd++) {
                    // Avoid infinite loop when rd=cd=0
                    if(!rd && !cd) continue;

                    // Take a step in the current direction
                    cr = row + rd;
                    cc = column + cd;

                    count = 0;
                    while(cr >= 0 && cr < b->height && cc >= 0 && cc < b->width) {
                        bv = board_get_cuda(b, cr, cc);
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
            return counts > 0;
        }
    }

    // Either the board pointer was empty, or the space was already filled.
    return 0;
}