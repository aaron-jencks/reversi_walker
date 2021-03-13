#pragma once

#include <stdint.h>

#include "../gameplay/reversi.h"

#ifdef __cplusplus
extern "C" {
#endif

// __uint128_t board_hash(void* brd);

__uint128_t board_spiral_hash(void* brd);

__uint128_t board_fast_hash_6(board b);

#ifdef __cplusplus
}
#endif