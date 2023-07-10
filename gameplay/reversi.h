#pragma once

#include <stdint.h>
#include <stdbool.h>

#include "reversi_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a board object
 * 
 * @param starting_player 1 for white start, 0 for black start
 * @param height The height of the board in rows
 * @param width The width of the board in columns
 * @return board_t 
 */
board_t create_board(uint8_t starting_player, uint8_t height, uint8_t width);

/**
 * @brief Create an 8x8 board object, unhashing from a uint128, must be an 8x8 board or smaller
 * 
 * @param starting_player 1 for white start, 0 for black start
 * @param key The key to unhash the board from
 * @return board_t 
 */
board_t create_board_unhash_8(uint8_t starting_player, __uint128_t key);

/**
 * @brief Create a 6x6 board object, unhashing from a uint128, must be an 8x8 board or smaller
 * 
 * @param starting_player 1 for white start, 0 for black start
 * @param key The key to unhash the board from
 * @return board_t 
 */
board_t create_board_unhash_6(uint8_t starting_player, __uint128_t key);

/**
 * @brief Create a board object, copying another board
 * 
 * @param board Board to clone
 * @return othelloboard 
 */
board_t clone_board(board_t b);

/**
 * @brief Clones a board into a given already allocated board.
 * 
 * @param src 
 * @param dest 
 */
void clone_into_board(board_t src, board_t* dest);

/**
 * @brief Destroys a given board
 * 
 * @param b 
 */
void destroy_board(board_t b);

// Transfer the boolean board values in and out of the bits in the struct

/**
 * @brief Gets a board space value out of the 16 byte chunk that stores the board
 * 
 * @param b Board to extract the position out of
 * @param row The row of the coordinate to retrieve the value of
 * @param column The column of the coordinate to retrieve the value of
 * @return uint8_t Returns 0 for empty, 1 for white, and 2 for black
 */
uint8_t board_get(board_t b, uint8_t row, uint8_t column);

/**
 * @brief Inserts a board space into the 16 byte chunk that stores the board
 * 
 * @param b The board to place the value into
 * @param row The row of the coordinate to place the value into
 * @param column The column of the coordinate to place the value into
 * @param value 0 for empty, 1 for white, 2 for black
 */
void board_put(board_t b, uint8_t row, uint8_t column, uint8_t value);

/**
 * @brief Determines if the move is legal, 
 * a move is legal if it captures at least 1 piece of the opposing color and the square is unoccupied
 * 
 * @param b Board to check
 * @param row Row of the coordinate to check
 * @param column Column of the coordinate to check
 * @return bool returns true if the move is legal and false otherwise
 */
bool board_is_legal_move(board_t b, uint8_t row, uint8_t column);

/**
 * @brief Places a piece on the board, does not check if it's legal
 * 
 * @param b The board to place the piece on
 * @param row The row to place the piece at
 * @param column The column to place the piece at
 * @return uint64_t Returns a struct representing how many pieces in each direction the move captured
 */
uint64_t board_place_piece(board_t* b, uint8_t row, uint8_t column);

/**
 * @brief Finds the count for the given capture_count struct in the given direction, the directions are:
 * 0: upper-left
 * 1: up
 * 2: upper-right
 * 3: left
 * 4: right
 * 5: lower-left
 * 6: lower
 * 7: lower-right
 * 
 * @param c 
 * @param direction 
 * @return uint8_t Returns the number of pieces captured in the given direction
 */
uint8_t capture_count_get_count(uint64_t c, uint8_t direction);

/**
 * @brief Inserts a capture count into the given capture_count struct for the given direction and count,
 * the directions are:
 * 0: upper-left
 * 1: up
 * 2: upper-right
 * 3: left
 * 4: right
 * 5: lower-left
 * 6: lower
 * 7: lower-right
 * 
 * @param c 
 * @param direction 
 * @param count The number of pieces captured, must be in the interval (0,6)
 */
uint64_t capture_count_put_count(uint64_t c, uint8_t direction, uint8_t count);

#ifdef __cplusplus
}
#endif