#include "reversi.cuh"
#include "../gameplay/reversi.h"

#include <err.h>
#include <stdio.h>

/**
 * A Warp is a logical unit that shares a pc
 * Keep thread blocks as large as possible without increasing divergence
 */

 __host__ board create_board_cuda(uint8_t starting_player, uint8_t height, uint8_t width) {
    board b;
    cudaMallocManaged(&b, sizeof(board_str));
    if(!b) err(1, "Memory Error while allocating the board\n");
    b->player = starting_player;
    b->height = height;
    b->width = width;

    cudaMallocManaged(&b->board, sizeof(uint8_t) * ((height * width) >> 2));
    memset(b->board, 0, sizeof(uint8_t) * ((height * width) >> 2));
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

 __host__ void destroy_board_cuda(board b) {
     cudaFree(b->board);
     cudaFree(b);
 }

 __host__ __device__ uint8_t board_get_cuda(board b, uint8_t row, uint8_t column) {
    if(b) {
        // printf("Fetching board square\n");
        uint8_t total_bit = (row * (b->width << 1)) + (column << 1), 
                byte = total_bit >> 3, 
                bit = total_bit % 8;

        return ((192 >> bit) & b->board[byte]) >> (6 - bit);
    }
    return 3;
}

__host__ __device__ void board_put_cuda(board b, uint8_t row, uint8_t column, uint8_t player) {
    if(b) {
        uint8_t total_bit = (row * (b->width << 1)) + (column << 1), 
                byte = total_bit >> 3, 
                bit = total_bit % 8, 
                bph = 192 >> bit;

        b->board[byte] = (b->board[byte] | bph) ^ bph;

        if(player) b->board[byte] |= ((player == 1) ? 64 : 128) >> bit;
    }
}

__host__ __device__ uint8_t board_is_legal_move_cuda(board b, uint8_t row, uint8_t column) {
    if(b && row < b->height && column < b->width) {
        if(!board_get_cuda(b, row, column)) {

            // printf("Managed to check a board\n");

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
                    while(cr < b->height && cc < b->width) {
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

__host__ __device__ void clone_into_board_cuda(board src, board dest) {
    if(src && dest) {
        dest->height = src->height;
        dest->width = src->width;
        dest->player = src->player;
        for(uint8_t i = 0; i < ((src->height * src->width) >> 2); i++) dest->board[i] = src->board[i];
    }
}

__host__ __device__ void board_place_piece_cuda(board b, uint8_t row, uint8_t column) {
    if(b && row >= 0 && row < b->height && column >= 0 && column < b->width) {
        board_put_cuda(b, row, column, b->player);
        int8_t cr, cc, bv;
        uint8_t count;
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
                
                if(count > 0) {
                    cr = row + rd;
                    cc = column + cd;
                    bv = board_get_cuda(b, cr, cc);
    
                    while(bv && bv != b->player) {
                        board_put_cuda(b, cr, cc, b->player);
                        cr += rd;
                        cc += cd;
                        bv = board_get_cuda(b, cr, cc);
                    }
                }
            }
        }
    
        // Flip the player to the opponent
        b->player = (b->player == 1) ? 2 : 1;
    }
}