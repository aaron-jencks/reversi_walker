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
 * @return coord* Returns a zero-terminated array of coordinates that represent valid moves from this board
 */
coord* find_next_boards(board b);

/**
 * @brief Finds the next set of boards that can be reached from this one,
 * but only checks the 8 cells surrounding coordinate c
 * 
 * @param b 
 * @param c 
 * @return coord* Returns a zero-terminated array of coordinates that represent valid moves from this board
 */
coord* find_next_boards_from_coord(board b, coord c);

coord create_coord(uint8_t row, uint8_t column);
uint16_t coord_to_short(coord c);
uint16_t coord_to_short_ints(uint8_t r, uint8_t c);
coord short_to_coord(uint16_t s);