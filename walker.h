#pragma once

#include "reversi.h"

#include <stdint.h>

typedef struct _coord_str {
    uint8_t row;
    uint8_t column;
} coord_str;

/**
 * @brief Represents a coordinate on the reversi board
 * 
 */
typedef coord_str* coord;

/**
 * @brief Finds the next set of boards that can be reached from this one
 * 
 * @param b 
 * @return cache_index* 
 */
 coord* find_next_boards(board b);