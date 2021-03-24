#pragma once

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

typedef struct _board_str {
    uint8_t player;
    uint8_t width;
    uint8_t height;
    uint8_t* board;
} board_str;

typedef board_str* board;

typedef struct _inboard_str {
    uint8_t player;
    uint8_t width;
    uint8_t height;
    uint8_t board[16];
} inboard_t;

typedef struct _capture_count_str {
    uint8_t* counts;
} capture_count_str;

typedef capture_count_str* capture_count;

#define BOARD_WIDTH 6
#define BOARD_HEIGHT 6