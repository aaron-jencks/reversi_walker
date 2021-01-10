#pragma once

#include <stdint.h>

#include "../gameplay/revers.h"


 __global__ uint8_t board_get_cuda(board b, uint8_t row, uint8_t column);

 __global__ void board_put_cuda(board b, uint8_t row, uint8_t column, uint8_t value);

__global__ uint8_t board_is_legal_move_cuda(board b, uint8_t row, uint8_t column);