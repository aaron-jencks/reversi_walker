package gameplay

import "github.com/aaron-jencks/reversi/utils/uint128"

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
func LinearHash(b Board) uint128.Uint128 {
	center := b.Size >> 1
	ocenter := center - 1
	var header uint8 = uint8(b.Player-1) << 4
	header += uint8(b.Get(center, ocenter)-1) << 3
	header += uint8(b.Get(ocenter, center)-1) << 2
	header += uint8(b.Get(ocenter, ocenter)-1) << 1
	header += uint8(b.Get(center, center) - 1)
	header <<= 2

	result := uint128.Uint128{
		L: uint64(header),
	}

	ca := int(ocenter*b.Size + ocenter)
	cb := int(ocenter*b.Size + center)
	cd := int(center*b.Size + ocenter)
	ce := int(center*b.Size + center)

	for bi := range b.Board {
		if bi == ca || bi == cb || bi == cd || bi == ce {
			continue
		}
		result.L += uint64(b.Board[bi])
		if bi == len(b.Board)-1 {
			// don't need to bitshift
			continue
		}
		result = result.ShiftLeft(2)
	}

	return result
}

// SimpleUnhashBoard unhashes a board from the given hash key and board size
func LinearUnhashBoard(size uint8, key uint128.Uint128) Board {
	result := Board{
		Size:  size,
		Board: make([]uint8, size*size),
	}

	center := size >> 1
	ocenter := center - 1

	ca := int(ocenter*size + ocenter)
	cb := int(ocenter*size + center)
	cd := int(center*size + ocenter)
	ce := int(center*size + center)

	for bi := int(size*size - 1); bi >= 0; bi-- {
		if bi == ca || bi == cb || bi == cd || bi == ce {
			continue
		}

		bv := BoardValue(key.L & 3)
		key = key.ShiftRight(2)
		if bv > BOARD_EMPTY {
			result.Board[bi] = uint8(bv)
		}
	}

	bv := BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(center, center, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(ocenter, ocenter, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(ocenter, center, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(center, ocenter, bv)

	result.Player = BoardValue((key.L & 1) + 1)

	return result
}
