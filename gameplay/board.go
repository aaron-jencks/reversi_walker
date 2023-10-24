package gameplay

import (
	"fmt"

	"github.com/aaron-jencks/reversi/utils/uint128"
)

// Represents a reversi board
type Board struct {
	Player BoardValue // the current player
	Width  uint8      // the width of the board
	Height uint8      // the height of the board
	Board  []uint8    // a compressed version of the board
}

// Hash hashes the board into a 128 bit integer of the format
//      |-|----|---...---|
//       ^ ^            ^
//  Player Center 4     Rest of the board
// we only need 1 bit for the player as well as the center 4 squares
// this is because the center squares will never be empty
//
// The rest of the squares require 2 bits per square
func (b Board) Hash() uint128.Uint128 {
	var rc, cc uint8 = b.Height >> 1, b.Width >> 1
	var header uint8 = uint8(b.Player-1) << 4
	header += uint8(b.Get(rc, cc)-1) << 3
	header += uint8(b.Get(rc-1, cc)-1) << 2
	header += uint8(b.Get(rc-1, cc-1)-1) << 1
	header += uint8(b.Get(rc, cc-1) - 1)
	header <<= 2

	result := uint128.Uint128{
		L: uint64(header),
	}

	for i := uint8(0); i < b.Height; i++ {
		for j := uint8(0); j < b.Width; j++ {
			if (i == rc && j == cc) ||
				(i == rc-1 && j == cc-1) ||
				(i == rc-1 && j == cc) ||
				(i == rc && j == cc-1) {
				// if it's one of the center squares, skip it
				continue
			}
			bv := b.Get(i, j)
			result.L += uint64(bv)
			if i == b.Height-1 && j == b.Width-1 {
				// don't need to bitshift
				continue
			}
			result = result.ShiftLeft(2)
		}
	}

	return result
}

func (b Board) Equal(o Board) bool {
	if b.Player != o.Player ||
		b.Height != o.Height || b.Width != o.Width {
		return false
	}

	for i := uint8(0); i < b.Height; i++ {
		for j := uint8(0); j < b.Width; j++ {
			if b.Get(i, j) != o.Get(i, j) {
				return false
			}
		}
	}

	return true
}

// Get retrieves the value of the board at the given position
func (b Board) Get(row, column uint8) BoardValue {
	/* column: 0 1 2 3 4 5 6 7
	 *        +-+-+-+-+-+-+-+-+
	 * row 0: | | | | | | | | | <-- Byte 0,1
	 *        +-+-+-+-+-+-+-+-+
	 * row 1: | | | | | | | | | <-- Byte 2,3
	 *        +-+-+-+-+-+-+-+-+
	 * row 2: | | | | | | | | | <-- Byte 4,5
	 *        +-+-+-+-+-+-+-+-+
	 * row 3: | | | |2|1| | | | <-- Byte 6,7
	 *        +-+-+-+-+-+-+-+-+
	 * row 4: | | | |1|2| | | | <-- Byte 8,9
	 *        +-+-+-+-+-+-+-+-+
	 * row 5: | | | | | | | | | <-- Byte 10,11
	 *        +-+-+-+-+-+-+-+-+
	 * row 6: | | | | | | | | | <-- Byte 12,13
	 *        +-+-+-+-+-+-+-+-+
	 * row 7: | | | | | | | | | <-- Byte 14,15
	 *        +-+-+-+-+-+-+-+-+
	 */

	var total_bit uint8 = (row * (b.Width << 1)) + (column << 1)
	var sel_byte uint8 = total_bit >> 3
	var bit uint8 = total_bit & 7

	bv := BoardValue(((192 >> bit) & b.Board[sel_byte]) >> (6 - bit))
	return bv
}

// Put places a piece onto the board at the given row and column
func (b *Board) Put(row, column uint8, value BoardValue) {
	var total_bit uint8 = (row * (b.Width << 1)) + (column << 1)
	var sel_byte uint8 = total_bit >> 3
	var bit uint8 = total_bit & 7
	var bph uint8 = 192 >> bit

	// Reset the value of the bits to 0
	// byte (????????) | bph (00011000) = ????11???
	// ???11??? ^ 00011000 = ???00???
	b.Board[sel_byte] = (b.Board[sel_byte] | bph) ^ bph

	// Insert your new value
	// byte (????????) | ((10|01)000000 >> bit) = ????(10|01)???
	if value > BOARD_EMPTY {
		var nv uint8 = 64
		if value == BOARD_BLACK {
			nv = 128
		}
		b.Board[sel_byte] |= nv >> bit
	}
}

// Clone clones the current board
func (b Board) Clone() Board {
	result := Board{
		Player: b.Player,
		Height: b.Height,
		Width:  b.Width,
		Board:  make([]uint8, len(b.Board)),
	}
	copy(result.Board, b.Board)
	return result
}

func (b Board) CloneInto(bt *Board) {
	bt.Player = b.Player
	bt.Height = b.Height
	bt.Width = b.Width
	copy(bt.Board, b.Board)
}

// IsLegalMove determines if a given position is legal for the current turn
func (b Board) IsLegalMove(row, column uint8) bool {
	if row >= b.Height || column >= b.Width || b.Get(row, column) != BOARD_EMPTY {
		return false
	}

	var counts, cr, cc, count uint8
	var bv BoardValue
	for rd := int8(-1); rd < 2; rd++ {
		for cd := int8(-1); cd < 2; cd++ {
			// avoid infinite loop when rd=cd=0
			if rd == 0 && cd == 0 {
				continue
			}

			// take step in the current direction
			cr = uint8(int8(row) + rd)
			cc = uint8(int8(column) + cd)

			count = 0
			for cr < b.Height && cc < b.Width {
				bv = b.Get(cr, cc)
				if bv > BOARD_EMPTY && bv != b.Player {
					// there is a possible capture
					count++

					// take another step in the current direction
					cr = uint8(int8(cr) + rd)
					cc = uint8(int8(cc) + cd)

					if cr >= b.Height || cc >= b.Width {
						// we hit the edge of the board, this is not a capture
						count = 0
						break
					}
				} else {
					if bv == BOARD_EMPTY {
						// if we had any captures, they are in vain because our color isn't at theother end
						count = 0
					}
					break
				}
			}
			counts += count
		}
	}

	// Return true if we capture at least one piece
	return counts > 0
}

// PlacePiece places a piece at the given position and flips
// the correct pieces
func (b *Board) PlacePiece(row, column uint8) {
	if row >= b.Height || column >= b.Width || b.Get(row, column) != BOARD_EMPTY {
		return
	}

	b.Put(row, column, b.Player)

	var cr, cc uint8
	var bv BoardValue
	for rd := int8(-1); rd < 2; rd++ {
		for cd := int8(-1); cd < 2; cd++ {
			// avoid infinite loop when rd=cd=0
			if rd == 0 && cd == 0 {
				continue
			}

			// take step in the current direction
			cr = uint8(int8(row) + rd)
			cc = uint8(int8(column) + cd)

			if cr >= b.Height || cc >= b.Width {
				continue
			}

			// get the first board value
			bv = b.Get(cr, cc)

			// we do need to count how many pieces there are
			// otherwise we'll end up flipping pieces we don't intend to

			count := 0
			ccr := cr
			ccc := cc

			for bv != b.Player {
				count++

				// take another step in the current direction
				ccr = uint8(int8(ccr) + rd)
				ccc = uint8(int8(ccc) + cd)

				if ccr >= b.Height || ccc >= b.Width {
					count = 0
					break
				}

				// get the next board value
				bv = b.Get(ccr, ccc)

				if bv == BOARD_EMPTY {
					count = 0
					break
				}
			}

			// now we can flip the pieces
			ccr = cr
			ccc = cc

			for c := 0; c < count; c++ {
				// flip the piece
				b.Put(ccr, ccc, b.Player)

				ccr = uint8(int8(ccr) + rd)
				ccc = uint8(int8(ccc) + cd)
			}

		}
	}

	if b.Player == BOARD_WHITE {
		b.Player = BOARD_BLACK
	} else {
		b.Player = BOARD_WHITE
	}
}

func (b Board) Display() {
	fmt.Printf("player: %d\n  ", b.Player)
	for i := uint8(0); i < b.Width; i++ {
		fmt.Printf("%d ", i)
	}
	fmt.Println("")
	for i := uint8(0); i < b.Height; i++ {
		fmt.Printf("%d ", i)
		for j := uint8(0); j < b.Width; j++ {
			fmt.Printf("%d ", b.Get(i, j))
		}
		fmt.Println("")
	}
}

func CreateEmptyBoard(starting_player BoardValue, height, width uint8) Board {
	return Board{
		Player: starting_player,
		Height: height,
		Width:  width,
		Board:  make([]uint8, (height*width)>>2),
	}
}

// CreateBoard creates a board with the given starting player, height, and width
func CreateBoard(starting_player BoardValue, height, width uint8) Board {
	result := CreateEmptyBoard(starting_player, height, width)

	/* <1 byte>
	 * +-+-+-+-+-+-+-+-+
	 * | | | | | | | | | <-- 2 bytes | Byte 0,1
	 * +-+-+-+-+-+-+-+-+
	 * | | | | | | | | | <-- Byte 2,3
	 * +-+-+-+-+-+-+-+-+
	 * | | | | | | | | | <-- Byte 4,5
	 * +-+-+-+-+-+-+-+-+
	 * | | | |2|1| | | | <-- Byte 6,7
	 * +-+-+-+-+-+-+-+-+
	 * | | | |1|2| | | | <-- Byte 8,9
	 * +-+-+-+-+-+-+-+-+
	 * | | | | | | | | | <-- Byte 10,11
	 * +-+-+-+-+-+-+-+-+
	 * | | | | | | | | | <-- Byte 12,13
	 * +-+-+-+-+-+-+-+-+
	 * | | | | | | | | | <-- Byte 14,15
	 * +-+-+-+-+-+-+-+-+
	 */

	/** To create the starting position we need to fill the bits as such:
	 * +--+--+--+--+--+--+--+--+
	 * |00|00|00|10|01|00|00|00| <-- 2,64 for byte 6 and 7
	 * +--+--+--+--+--+--+--+--+
	 * |00|00|00|01|10|00|00|00| <-- 1,128 for byte 8 and 9
	 * +--+--+--+--+--+--+--+--+
	 */

	result.Put((height>>1)-1, (width>>1)-1, BOARD_BLACK)
	result.Put((height>>1)-1, width>>1, BOARD_WHITE)
	result.Put(height>>1, (width>>1)-1, BOARD_WHITE)
	result.Put(height>>1, width>>1, BOARD_BLACK)

	return result
}

// CreateUnhashBoard unhashes a board from the given hash key and board size
func CreateUnhashBoard(size uint8, key uint128.Uint128) Board {
	result := Board{
		Height: size,
		Width:  size,
		Board:  make([]uint8, (size*size)>>2),
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
