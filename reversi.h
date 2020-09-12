#pragma once

#include <stdint.h>

typedef struct _board_str {
    uint8_t player;
    uint8_t width;
    uint8_t height;
    uint8_t* board;
} board_str;

typedef board_str* board;

/**
 * @brief Create a board object
 * 
 * @param starting_player 1 for white start, 0 for black start
 * @param height The height of the board in rows
 * @param width The width of the board in columns
 * @return board 
 */
board create_board(uint8_t starting_player, uint8_t height, uint8_t width);

/**
 * @brief Destroys a given board
 * 
 * @param b 
 */
void destroy_board(board b);

// Transfer the boolean board values in and out of the bits in the struct

/**
 * @brief Gets a board space value out of the 16 byte chunk that stores the board
 * 
 * @param b Board to extract the position out of
 * @param row The row of the coordinate to retrieve the value of
 * @param column The column of the coordinate to retrieve the value of
 * @return uint8_t Returns 0 for empty, 1 for white, and 2 for black
 */
uint8_t board_get(board b, uint8_t row, uint8_t column);

/**
 * @brief Inserts a board space into the 16 byte chunk that stores the board
 * 
 * @param b The board to place the value into
 * @param row The row of the coordinate to place the value into
 * @param column The column of the coordinate to place the value into
 * @param value 0 for empty, 1 for white, 2 for black
 */
void board_put(board b, uint8_t row, uint8_t column, uint8_t value);

/**
 * @brief Determines if the move is legal, 
 * a move is legal if it captures at least 1 piece of the opposing color and the square is unoccupied
 * 
 * @param b Board to check
 * @param row Row of the coordinate to check
 * @param column Column of the coordinate to check
 * @return uint8_t Returns 1 if the move is legal and 0 otherwise
 */
uint8_t board_is_legal_move(board b, uint8_t row, uint8_t column);

/**
 * @brief Places a piece on the board, does not check if it's legal
 * 
 * @param b The board to place the piece on
 * @param row The row to place the piece at
 * @param column The column to place the piece at
 */
void board_place_piece(board b, uint8_t row, uint8_t column);