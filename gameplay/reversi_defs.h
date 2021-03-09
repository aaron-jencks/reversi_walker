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