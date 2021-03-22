#pragma once

#include <stdint.h>

#include "../gameplay/reversi_defs.h"

__host__ board create_board_cuda(uint8_t starting_player, uint8_t height, uint8_t width);
__host__ void destroy_board_cuda(board b);

__device__ uint8_t board_get_cuda(board b, uint8_t row, uint8_t column);

__device__ void board_put_cuda(board b, uint8_t row, uint8_t column, uint8_t value);

__host__ __device__ uint8_t board_is_legal_move_cuda(board b, uint8_t row, uint8_t column);

__host__ __device__ void clone_into_board_cuda(board src, board dest);

__host__ __device__ void board_place_piece_cuda(board b, uint8_t row, uint8_t column);