package gameplay

import (
	"fmt"
	"testing"

	"github.com/aaron-jencks/reversi/utils/uint128"
)

func generatePossibleBoards(size, r, c uint8) []Board {

	var children []Board

	nr := r
	nc := c + 1
	if nc == size {
		nr++
		if nr == size {
			children = []Board{
				CreateEmptyBoard(BOARD_BLACK, size),
				CreateEmptyBoard(BOARD_WHITE, size),
			}
		}
		nc = 0
	}

	if children == nil {
		children = generatePossibleBoards(size, nr, nc)
	}

	var values []BoardValue = []BoardValue{
		BOARD_BLACK,
		BOARD_WHITE,
	}
	center := size >> 1
	if !((r == center || r == center-1) && (c == center || c == center-1)) {
		// we're not in the center 4
		values = append(values, BOARD_EMPTY)
	}

	result := make([]Board, len(children)*len(values))

	for vi, v := range values {
		for bi, b := range children {
			index := vi*len(children) + bi
			result[index] = b.Clone()
			result[index].Put(r, c, v)
		}
	}

	return result
}

func Test4x4SpiralHashUniqueness(t *testing.T) {
	cache := map[uint64]map[uint64]Board{}

	cacheInsert := func(b Board, bh uint128.Uint128) bool {
		inner, ok := cache[bh.H]
		if ok {
			_, ok := inner[bh.L]
			if ok {
				return false
			}
			inner[bh.L] = b
		} else {
			cache[bh.H] = map[uint64]Board{
				bh.L: b,
			}
		}
		return true
	}

	boards := generatePossibleBoards(4, 0, 0)
	fmt.Printf("Generated %d boards\n", len(boards))
	for bi, b := range boards {
		bh := SpiralHash(b)
		if !cacheInsert(b, bh) {
			b.Display()
			fmt.Println("\n and")
			fmt.Println()
			cache[bh.H][bh.L].Display()
			fmt.Println("\n have the same hash value of:")
			fmt.Println(bh)
			fmt.Println()
			t.Fail()
		}
		fmt.Printf("\rTested %d/%d boards", bi+1, len(boards))
	}
}
