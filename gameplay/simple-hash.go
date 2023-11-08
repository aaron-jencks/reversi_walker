package gameplay

import (
	"github.com/aaron-jencks/reversi/utils/uint128"
)

// Hash hashes the board into a 128 bit integer of the format
//
//	    |-|----|---...---|
//	     ^ ^            ^
//	Player Center 4     Rest of the board
//
// we only need 1 bit for the player as well as the center 4 squares
// this is because the center squares will never be empty
//
// The rest of the squares require 2 bits per square
func SimpleHash(b Board) uint128.Uint128 {
	var rc, cc uint8 = b.Size >> 1, b.Size >> 1
	var header uint8 = uint8(b.Player-1) << 4
	header += uint8(b.Get(rc, cc)-1) << 3
	header += uint8(b.Get(rc-1, cc)-1) << 2
	header += uint8(b.Get(rc-1, cc-1)-1) << 1
	header += uint8(b.Get(rc, cc-1) - 1)
	header <<= 2

	result := uint128.Uint128{
		L: uint64(header),
	}

	for i := uint8(0); i < b.Size; i++ {
		for j := uint8(0); j < b.Size; j++ {
			if (i == rc && j == cc) ||
				(i == rc-1 && j == cc-1) ||
				(i == rc-1 && j == cc) ||
				(i == rc && j == cc-1) {
				// if it's one of the center squares, skip it
				continue
			}
			bv := b.Get(i, j)
			result.L += uint64(bv)
			if i == b.Size-1 && j == b.Size-1 {
				// don't need to bitshift
				continue
			}
			result = result.ShiftLeft(2)
		}
	}

	return result
}

// SimpleUnhashBoard unhashes a board from the given hash key and board size
func SimpleUnhashBoard(size uint8, key uint128.Uint128) Board {
	result := Board{
		Size:  size,
		Board: make([]uint8, size*size),
	}

	center := size >> 1

	for i := int(size - 1); i >= 0; i-- {
		for j := int(size - 1); j >= 0; j-- {
			ui := uint8(i)
			uj := uint8(j)
			if (ui == center && uj == center) ||
				(ui == center-1 && uj == center-1) ||
				(ui == center-1 && uj == center) ||
				(ui == center && uj == center-1) {
				// if it's one of the center squares, skip it
				continue
			}
			bv := BoardValue(key.L & 3)
			key = key.ShiftRight(2)
			if bv > BOARD_EMPTY {
				result.Put(uint8(i), uint8(j), bv)
			}
		}
	}

	bv := BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(center, center-1, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(center-1, center-1, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(center-1, center, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(center, center, bv)

	result.Player = BoardValue((key.L & 1) + 1)

	return result
}
