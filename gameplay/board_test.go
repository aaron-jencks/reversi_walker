package gameplay

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCreation(t *testing.T) {
	b := CreateBoard(BOARD_WHITE, 8, 8)
	assert.Equal(t, BOARD_BLACK, b.Get(3, 3), "size 8: expected player two in upper left")
	assert.Equal(t, BOARD_BLACK, b.Get(4, 4), "size 8: expected player two in lower right")
	assert.Equal(t, BOARD_WHITE, b.Get(3, 4), "size 8: expected player one in upper right")
	assert.Equal(t, BOARD_WHITE, b.Get(4, 3), "size 8: expected player one in lower left")
	b = CreateBoard(BOARD_WHITE, 6, 6)
	assert.Equal(t, BOARD_BLACK, b.Get(2, 2), "size 6: expected player two in upper left")
	assert.Equal(t, BOARD_BLACK, b.Get(3, 3), "size 6: expected player two in lower right")
	assert.Equal(t, BOARD_WHITE, b.Get(2, 3), "size 6: expected player one in upper right")
	assert.Equal(t, BOARD_WHITE, b.Get(3, 2), "size 6: expected player one in lower left")
	b = CreateBoard(BOARD_WHITE, 4, 4)
	assert.Equal(t, BOARD_BLACK, b.Get(1, 1), "size 4: expected player two in upper left")
	assert.Equal(t, BOARD_BLACK, b.Get(2, 2), "size 4: expected player two in lower right")
	assert.Equal(t, BOARD_WHITE, b.Get(1, 2), "size 4: expected player one in upper right")
	assert.Equal(t, BOARD_WHITE, b.Get(2, 1), "size 4: expected player one in lower left")
}

func TestBoardPutGet(t *testing.T) {
	for _, size := range []uint8{4, 6, 8} {
		b := CreateBoard(BOARD_WHITE, size, size)
		for i := uint8(0); i < size; i++ {
			for j := uint8(0); j < size; j++ {
				b.Put(i, j, BOARD_WHITE)
				assert.Equal(t, BOARD_WHITE, b.Get(i, j), "expected correct value on the board after placement")
				b.Put(i, j, BOARD_BLACK)
				assert.Equal(t, BOARD_BLACK, b.Get(i, j), "expected correct value on the board after placement")
				b.Put(i, j, BOARD_EMPTY)
				assert.Equal(t, BOARD_EMPTY, b.Get(i, j), "expected correct value on the board after placement")
			}
		}
	}
}

type pieceCoordinate struct {
	Row    uint8
	Column uint8
	Value  BoardValue
}

func TestLegality(t *testing.T) {
	tcs := []struct {
		name   string
		pieces []pieceCoordinate
		row    uint8
		column uint8
		player BoardValue
		legal  bool
	}{
		{
			name: "vertical down legal",
			pieces: []pieceCoordinate{
				{2, 4, BOARD_WHITE},
				{1, 4, BOARD_WHITE},
			},
			row:    0,
			column: 4,
			player: BOARD_BLACK,
			legal:  true,
		},
		{
			name: "vertical illegal edge",
			pieces: []pieceCoordinate{
				{0, 0, BOARD_WHITE},
				{0, 1, BOARD_WHITE},
				{0, 2, BOARD_WHITE},
			},
			row:    0,
			column: 3,
			player: BOARD_BLACK,
		},
		{
			name: "illegal not empty",
			pieces: []pieceCoordinate{
				{2, 4, BOARD_WHITE},
				{1, 4, BOARD_WHITE},
				{0, 4, BOARD_BLACK},
			},
			row:    0,
			column: 4,
			player: BOARD_BLACK,
		},
		{
			name: "vertical up legal",
			pieces: []pieceCoordinate{
				{5, 3, BOARD_WHITE},
				{6, 3, BOARD_WHITE},
			},
			row:    7,
			column: 3,
			player: BOARD_BLACK,
			legal:  true,
		},
		{
			name: "horizontal right legal",
			pieces: []pieceCoordinate{
				{3, 2, BOARD_WHITE},
				{3, 1, BOARD_WHITE},
			},
			row:    3,
			column: 0,
			player: BOARD_BLACK,
			legal:  true,
		},
		{
			name: "horizontal left legal",
			pieces: []pieceCoordinate{
				{4, 5, BOARD_WHITE},
				{4, 6, BOARD_WHITE},
			},
			row:    4,
			column: 7,
			player: BOARD_BLACK,
			legal:  true,
		},
		{
			name: "diagonal -x down legal",
			pieces: []pieceCoordinate{
				{2, 2, BOARD_WHITE},
				{1, 1, BOARD_WHITE},
			},
			row:    0,
			column: 0,
			player: BOARD_BLACK,
			legal:  true,
		},
		{
			name: "diagonal -x up legal",
			pieces: []pieceCoordinate{
				{5, 5, BOARD_WHITE},
				{6, 6, BOARD_WHITE},
			},
			row:    7,
			column: 7,
			player: BOARD_BLACK,
			legal:  true,
		},
		{
			name: "diagonal x down legal",
			pieces: []pieceCoordinate{
				{2, 5, BOARD_BLACK},
				{1, 6, BOARD_BLACK},
			},
			row:    0,
			column: 7,
			player: BOARD_WHITE,
			legal:  true,
		},
		{
			name: "diagonal x up legal",
			pieces: []pieceCoordinate{
				{5, 2, BOARD_BLACK},
				{6, 1, BOARD_BLACK},
			},
			row:    7,
			column: 0,
			player: BOARD_WHITE,
			legal:  true,
		},
	}

	for _, tc := range tcs {
		t.Run(fmt.Sprintf("TestLegality: %s", tc.name), func(tt *testing.T) {
			b := CreateBoard(tc.player, 8, 8)
			for _, p := range tc.pieces {
				b.Put(p.Row, p.Column, p.Value)
			}
			av := b.IsLegalMove(tc.row, tc.column)
			assert.Equal(tt, tc.legal, av, "legality mismatch")
		})
	}
}

func TestPlacePiece(t *testing.T) {
	b := CreateBoard(BOARD_WHITE, 8, 8)
	b.Put(3, 3, BOARD_EMPTY)
	b.Put(3, 4, BOARD_EMPTY)
	b.Put(4, 3, BOARD_EMPTY)
	b.Put(4, 4, BOARD_EMPTY)

	pieces := []pieceCoordinate{
		{3, 2, BOARD_BLACK},
		{3, 1, BOARD_WHITE},
		{2, 3, BOARD_BLACK},
		{1, 3, BOARD_BLACK},
		{0, 3, BOARD_WHITE},
		{2, 2, BOARD_BLACK},
		{1, 1, BOARD_BLACK},
		{0, 0, BOARD_WHITE},
		{3, 4, BOARD_BLACK},
		{3, 5, BOARD_BLACK},
		{3, 6, BOARD_BLACK},
		{3, 7, BOARD_WHITE},
		{2, 4, BOARD_BLACK},
		{1, 5, BOARD_WHITE},
		{4, 4, BOARD_BLACK},
		{5, 5, BOARD_BLACK},
		{6, 6, BOARD_WHITE},
		{4, 3, BOARD_BLACK},
		{5, 3, BOARD_BLACK},
		{6, 3, BOARD_BLACK},
		{7, 3, BOARD_WHITE},
		{4, 2, BOARD_BLACK},
		{5, 1, BOARD_BLACK},
		{6, 0, BOARD_WHITE},
	}

	for _, piece := range pieces {
		b.Put(piece.Row, piece.Column, piece.Value)
	}

	assert.True(t, b.IsLegalMove(3, 3), "expected move to be legal")
	b.PlacePiece(3, 3)

	for _, piece := range pieces {
		assert.Equal(t, BOARD_WHITE, b.Get(piece.Row, piece.Column), "expected all pieces to get flipped")
	}

	assert.Equal(t, BOARD_WHITE, b.Get(3, 3), "expected piece to be placed")
	assert.Equal(t, BOARD_BLACK, b.Player, "player should have been toggled")
}

func TestBoardPlaceFlipping(t *testing.T) {
	createBoardFromPieces := func(player BoardValue, height, width uint8,
		white []Coord, black []Coord) Board {
		b := CreateEmptyBoard(player, height, width)
		for _, w := range white {
			b.Put(w.Row, w.Column, BOARD_WHITE)
		}
		for _, blk := range black {
			b.Put(blk.Row, blk.Column, BOARD_BLACK)
		}
		return b
	}

	tcs := []struct {
		name string
		pre  Board
		post Board
		row  uint8
		col  uint8
	}{
		{
			pre: createBoardFromPieces(BOARD_BLACK, 8, 8,
				[]Coord{
					{3, 4},
					{4, 3},
					{5, 4},
					{4, 4},
				}, []Coord{
					{3, 3},
				}),
			post: createBoardFromPieces(BOARD_WHITE, 8, 8,
				[]Coord{
					{3, 4},
					{4, 3},
					{5, 4},
				}, []Coord{
					{3, 3},
					{4, 4},
					{5, 5},
				}),
			row: 5,
			col: 5,
		},
	}

	for tci, tc := range tcs {
		t.Run(fmt.Sprintf("%d: %s", tci, tc.name), func(tt *testing.T) {
			tc.pre.PlacePiece(tc.row, tc.col)
			fmt.Println("Expected:")
			tc.post.Display()
			fmt.Println("Actual:")
			tc.pre.Display()
			assert.True(tt, tc.pre.Equal(tc.post), "expected pre and post board to be equal")
		})
	}
}

func TestHashing(t *testing.T) {
	const iterCount = 10000
	const bsize uint8 = 8
	const bcenter uint8 = bsize >> 1
	for i := 0; i < iterCount; i++ {
		pb := CreateBoard(BoardValue(rand.Intn(2))+1, bsize, bsize)
		for r := uint8(0); r < 4; r++ {
			for c := uint8(0); c < 4; c++ {
				if (r == bcenter && c == bcenter) ||
					(r == bcenter-1 && c == bcenter-1) ||
					(r == bcenter-1 && c == bcenter) ||
					(r == bcenter && c == bcenter-1) {
					// if it's one of the center squares, skip it for now
					continue
				}
				pb.Put(r, c, BoardValue(rand.Intn(3)))
			}
		}

		pb.Put(bcenter, bcenter, BoardValue(rand.Intn(2)+1))
		pb.Put(bcenter-1, bcenter-1, BoardValue(rand.Intn(2)+1))
		pb.Put(bcenter, bcenter-1, BoardValue(rand.Intn(2)+1))
		pb.Put(bcenter-1, bcenter, BoardValue(rand.Intn(2)+1))

		bh := pb.Hash()

		nb := CreateUnhashBoard(bsize, bh)

		if !assert.Equal(t, pb, nb, "expected pre-hash and post-hash boards to be the same") {
			break
		}
	}
}

func TestHashing4(t *testing.T) {
	const iterCount = 10000
	const bsize uint8 = 4
	const bcenter uint8 = bsize >> 1
	for i := 0; i < iterCount; i++ {
		pb := CreateBoard(BoardValue(rand.Intn(2))+1, bsize, bsize)
		for r := uint8(0); r < 4; r++ {
			for c := uint8(0); c < 4; c++ {
				if (r == bcenter && c == bcenter) ||
					(r == bcenter-1 && c == bcenter-1) ||
					(r == bcenter-1 && c == bcenter) ||
					(r == bcenter && c == bcenter-1) {
					// if it's one of the center squares, skip it for now
					continue
				}
				pb.Put(r, c, BoardValue(rand.Intn(3)))
			}
		}

		pb.Put(bcenter, bcenter, BoardValue(rand.Intn(2)+1))
		pb.Put(bcenter-1, bcenter-1, BoardValue(rand.Intn(2)+1))
		pb.Put(bcenter, bcenter-1, BoardValue(rand.Intn(2)+1))
		pb.Put(bcenter-1, bcenter, BoardValue(rand.Intn(2)+1))

		bh := pb.Hash()

		nb := CreateUnhashBoard(bsize, bh)

		if !assert.Equal(t, pb, nb, "expected pre-hash and post-hash boards to be the same") {
			break
		}
	}
}
