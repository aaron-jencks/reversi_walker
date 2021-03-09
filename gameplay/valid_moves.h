#pragma once

#include <stdint.h>
#include "reversi.h"
#include "reversi_defs.h"

/**
 * @brief Sets the corresponding bit to 1 in the given board int
 * 
 * @param b
 * @param row 
 * @param column 
 * @return uint64_t returns a copy of the board b with the corresponding bit set high
 */
uint64_t encode_valid_position(uint64_t b, uint8_t row, uint8_t column);

/**
 * @brief Removes the corresponding bit in the given board int, setting it to 0
 * 
 * @param b 
 * @param row 
 * @param column 
 * @return uint64_t returns a copy of the board b with the corresponding bit set low
 */
uint64_t remove_valid_position(uint64_t b, uint8_t row, uint8_t column);

/**
 * @brief Determines if the given position is a valid move position
 * 
 * @param b 
 * @param row 
 * @param column 
 * @return uint8_t Returns a 1 if the position is valid, and 0 otherwise.
 */
uint8_t is_valid_position(uint64_t b, uint8_t row, uint8_t column);

/**
 * @brief Finds the valid moves from a given coordinate, and update a board int accordingly
 * 
 * @param b 
 * @param bi
 * @param row 
 * @param column 
 * @return uint64_t Returns the updated board int
 */
uint64_t find_valid_positions_from_coord(uint64_t bi, board b, uint8_t row, uint8_t column);

/**
 * @brief Retrieves all of the positions in the board b that have 1s and returns their coordinates
 * 
 * @param b 
 * @return coord** Returns a zero-terminated list of coordinates that are in this board, or 0 if there are none
 */
coord* retrieve_all_valid_positions(uint64_t b);