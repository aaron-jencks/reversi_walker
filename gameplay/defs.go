package gameplay

import "github.com/aaron-jencks/reversi/utils/uint128"

type BoardValue uint8

const (
	BOARD_EMPTY = BoardValue(iota)
	BOARD_WHITE
	BOARD_BLACK
	BOARD_PIECES
)

// Represents a coordinate on the reversi board
type Coord struct {
	Row    uint8
	Column uint8
}

/*
 * Represents the number of captured pieces
 *
 * 0: upper-left
 * 1: up
 * 2: upper-right
 * 3: left
 * 4: right
 * 5: lower-left
 * 6: lower
 * 7: lower-right
 *
 * ---,---,--|-,---,---,-|--,---,---
 *  0   1   2    3   4  5     6   7
 */
type CaptureCounts struct {
	Counts []uint8
}

// Function that transforms a board into a uint128
type BoardHashFunc func(Board) uint128.Uint128

// Function that transforms a uint128 into a board with the given size
type BoardUnhashFunc func(uint8, uint128.Uint128) Board
