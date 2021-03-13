#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// __uint128_t board_hash(void* brd);

__uint128_t board_spiral_hash(void* brd);

#ifdef __cplusplus
}
#endif