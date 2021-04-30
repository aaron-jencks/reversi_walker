#pragma once

#include <stdint.h>

#include "../gameplay/reversi.h"

#ifdef __cplusplus
extern "C" {
#endif

// __uint128_t board_hash(void* brd);

__uint128_t board_spiral_hash(void* brd);

__uint128_t board_fast_hash_6(board b);

board board_unhash_6(__uint128_t key, uint8_t level);

#ifdef __cplusplus
}
#endif