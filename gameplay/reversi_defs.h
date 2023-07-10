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

typedef struct {
    uint8_t player;
    uint8_t width;
    uint8_t height;
    uint8_t* board;
} board_t;

typedef board_t* board;

typedef struct {
    uint8_t* counts;
} capture_count_t;

typedef capture_count_t* capture_count;

#define BOARD_WIDTH 6
#define BOARD_HEIGHT 6