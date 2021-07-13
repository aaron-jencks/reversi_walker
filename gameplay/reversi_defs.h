#pragma once

#include <stdint.h>
#include "../project_defs.hpp"

typedef struct _coord_str {
    uint8_t row;
    uint8_t column;
} coord_str;

/**
 * @brief Represents a coordinate on the reversi board
 * 
 */
typedef coord_str* coord;

typedef struct _board_str {
    uint8_t player; // TODO merge with level
    uint8_t width;  // TODO get rid of this
    uint8_t height;
    uint8_t board[(BOARD_HEIGHT * BOARD_WIDTH) >> 2];
    uint8_t level;
} board_str;

typedef board_str* board;

typedef struct _capture_count_str {
    uint8_t* counts;
} capture_count_str;

typedef capture_count_str* capture_count;