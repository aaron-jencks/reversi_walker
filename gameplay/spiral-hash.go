package gameplay

import "github.com/aaron-jencks/reversi/utils/uint128"

var spiralRowArr []int8 = []int8{
	-2, -2, -2,
	-1, 0, 1,
	1, 1, 1,
	0, -1, -2, -3,
	-3, -3, -3, -3,
	-2, -1, 0, 1, 2,
	2, 2, 2, 2, 2,
	1, 0, -1, -2, -3, -4,
	-4, -4, -4, -4, -4, -4,
	-3, -2, -1, 0, 1, 2, 3,
	3, 3, 3, 3, 3, 3, 3,
	2, 1, 0, -1, -2, -3, -4,
}

var spiralColArr []int8 = []int8{
	0, -1, -2,
	-2, -2, -2,
	-1, 0, 1,
	1, 1, 1, 1,
	0, -1, -2, -3,
	-3, -3, -3, -3, -3,
	-2, -1, 0, 1, 2,
	2, 2, 2, 2, 2, 2,
	1, 0, -1, -2, -3, -4,
	-4, -4, -4, -4, -4, -4, -4,
	-3, -2, -1, 0, 1, 2, 3,
	3, 3, 3, 3, 3, 3, 3,
}

// Hashes a board in a spiral pattern to create better locality
// 00000000
// 0kjihg00
// 00654f00
// 00703e00
// 00812d00
// 009abc00
// 00000000
// 00000000
func SpiralHash(b Board) uint128.Uint128 {
	var rc, cc uint8 = b.Size >> 1, b.Size >> 1
	var header uint8 = uint8(b.Player-1) << 4
	header += uint8(b.Get(rc-1, cc-1)-1) << 3
	header += uint8(b.Get(rc, cc-1)-1) << 2
	header += uint8(b.Get(rc, cc)-1) << 1
	header += uint8(b.Get(rc-1, cc) - 1)
	header <<= 2

	result := uint128.Uint128{
		L: uint64(header),
	}

	scount := b.Size*b.Size - 4

	for ci := uint8(0); ci < scount; ci++ {
		r := uint8(int8(rc) + spiralRowArr[ci])
		c := uint8(int8(cc) + spiralColArr[ci])
		bv := b.Get(r, c)
		result.L += uint64(bv)
		if ci == scount-1 {
			// Don't need to bitshift
			continue
		}
		result = result.ShiftLeft(2)
	}

	return result
}

// SpiralUnhashBoard unhashes a board from the given hash key and board size
func SpiralUnhashBoard(size uint8, key uint128.Uint128) Board {
	result := Board{
		Size:  size,
		Board: make([]uint8, size*size),
	}

	center := int8(size >> 1)
	scount := size*size - 4

	for ci := uint8(0); ci < scount; ci++ {
		rci := scount - ci - 1
		r := center + spiralRowArr[rci]
		c := center + spiralColArr[rci]
		bv := BoardValue(key.L & 3)
		key = key.ShiftRight(2)
		if bv > BOARD_EMPTY {
			result.Put(uint8(r), uint8(c), bv)
		}
	}

	ucenter := uint8(center)

	bv := BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(ucenter-1, ucenter, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(ucenter, ucenter, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(ucenter, ucenter-1, bv)
	bv = BoardValue((key.L & 1) + 1)
	key = key.ShiftRight(1)
	result.Put(ucenter-1, ucenter-1, bv)

	result.Player = BoardValue((key.L & 1) + 1)

	return result
}
