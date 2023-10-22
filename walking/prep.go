package walking

import "github.com/aaron-jencks/reversi/gameplay"

// Performs BFS on an initialized board of the given size
// and returns a minimum of count boards
func FindInitialBoards(count uint, size uint8) []gameplay.Board {
	q := make([]gameplay.Board, 0, size)
	q = append(q, gameplay.CreateBoard(gameplay.BOARD_WHITE, size, size))
	for len(q) < int(count) {
		qe := q[0]
		q = q[1:]
		moves := findNextBoards(qe)
		for _, m := range moves {
			bc := qe.Clone()
			bc.PlacePiece(m.Row, m.Column)
			q = append(q, bc)
		}
	}
	return q
}
