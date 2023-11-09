package gameplay

import (
	"fmt"

	"github.com/aaron-jencks/reversi/utils/uint128"
)

var boardHash BoardHashFunc = SpiralHash
var boardUnhash BoardUnhashFunc = SpiralUnhashBoard

func SetBoardHashFunc(hfun string) {
	fmt.Println("changing hash function to", hfun)
	switch hfun {
	case "simple":
		boardHash = SimpleHash
	case "spiral":
		boardHash = SpiralHash
	case "linear":
		boardHash = LinearHash
	}
}

func SetBoardUnHashFunc(hfun string) {
	fmt.Println("changing unhash function to", hfun)
	switch hfun {
	case "simple":
		boardUnhash = SimpleUnhashBoard
	case "spiral":
		boardUnhash = SpiralUnhashBoard
	case "linear":
		boardUnhash = LinearUnhashBoard
	}
}

// Represents a reversi board
type Board struct {
	Player BoardValue // the current player
	Size   uint8      // the width/height of the board
	Board  []uint8    // a compressed version of the board
}

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
func (b Board) Hash() uint128.Uint128 {
	return boardHash(b)
}

func (b Board) Equal(o Board) bool {
	if b.Player != o.Player || b.Size != o.Size {
		return false
	}

	for bi, by := range b.Board {
		if o.Board[bi] != by {
			return false
		}
	}

	return true
}

// Get retrieves the value of the board at the given position
func (b Board) Get(row, column uint8) BoardValue {
	// (0, 0) (0, 1) (0, 2) ... (0, w) (1, 0) ... (h, w)
	index := row*b.Size + column
	bv := BoardValue(b.Board[index])
	return bv
}

// Put places a piece onto the board at the given row and column
func (b *Board) Put(row, column uint8, value BoardValue) {
	index := row*b.Size + column
	b.Board[index] = uint8(value)
}

// Clone clones the current board
func (b Board) Clone() Board {
	result := Board{
		Player: b.Player,
		Size:   b.Size,
		Board:  make([]uint8, len(b.Board)),
	}
	copy(result.Board, b.Board)
	return result
}

func (b Board) CloneInto(bt *Board) {
	bt.Player = b.Player
	bt.Size = b.Size
	copy(bt.Board, b.Board)
}

// IsLegalMove determines if a given position is legal for the current turn
func (b Board) IsLegalMove(row, column uint8) bool {
	if row >= b.Size || column >= b.Size || b.Get(row, column) != BOARD_EMPTY {
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
			for cr < b.Size && cc < b.Size {
				bv = b.Get(cr, cc)
				if bv > BOARD_EMPTY && bv != b.Player {
					// there is a possible capture
					count++

					// take another step in the current direction
					cr = uint8(int8(cr) + rd)
					cc = uint8(int8(cc) + cd)

					if cr >= b.Size || cc >= b.Size {
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
	if row >= b.Size || column >= b.Size || b.Get(row, column) != BOARD_EMPTY {
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

			if cr >= b.Size || cc >= b.Size {
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

				if ccr >= b.Size || ccc >= b.Size {
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
	for i := uint8(0); i < b.Size; i++ {
		fmt.Printf("%d ", i)
	}
	fmt.Println("")
	for i := uint8(0); i < b.Size; i++ {
		fmt.Printf("%d ", i)
		for j := uint8(0); j < b.Size; j++ {
			fmt.Printf("%d ", b.Get(i, j))
		}
		fmt.Println("")
	}
}

func CreateEmptyBoard(starting_player BoardValue, size uint8) Board {
	return Board{
		Player: starting_player,
		Size:   size,
		Board:  make([]uint8, size*size),
	}
}

// CreateBoard creates a board with the given starting player, height, and width
func CreateBoard(starting_player BoardValue, size uint8) Board {
	result := CreateEmptyBoard(starting_player, size)

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

	cc := size >> 1
	occ := cc - 1

	result.Put(occ, occ, BOARD_BLACK)
	result.Put(occ, cc, BOARD_WHITE)
	result.Put(cc, occ, BOARD_WHITE)
	result.Put(cc, cc, BOARD_BLACK)

	return result
}

// CreateUnhashBoard unhashes a board from the given hash key and board size
func CreateUnhashBoard(size uint8, key uint128.Uint128) Board {
	return boardUnhash(size, key)
}
