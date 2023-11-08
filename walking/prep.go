package walking

import (
	"github.com/aaron-jencks/reversi/caching"
	"github.com/aaron-jencks/reversi/gameplay"
)

// TODO if this finds final boards, which is likely the case in a 4x4, then it should insert them into the cache

// Performs BFS on an initialized board of the given size
// and returns a minimum of count boards
func FindInitialBoards(count uint, size uint8) []gameplay.Board {
	neighbor_stack := caching.CreateArrayStack[gameplay.Coord](100)
	q := make([]gameplay.Board, 0, size)
	q = append(q, gameplay.CreateBoard(gameplay.BOARD_WHITE, size))
	for len(q) < int(count) {
		qe := q[0]
		q = q[1:]
		findNextBoards(qe, &neighbor_stack)
		for neighbor_stack.Len() > 0 {
			m := neighbor_stack.Pop()
			bc := qe.Clone()
			bc.PlacePiece(m.Row, m.Column)
			q = append(q, bc)
		}
	}
	return q
}
