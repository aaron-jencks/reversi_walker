#pragma once

#include "../gameplay/reversi_defs.h"

#include <stddef.h>

__global__ void compute_mass_next_cuda(board_str* boards, board_str* result, size_t n, size_t pitch);