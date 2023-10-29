package walking

import (
	"github.com/aaron-jencks/reversi/caching"
	"github.com/aaron-jencks/reversi/gameplay"
)

func findNextBoards(b gameplay.Board, coord_out *caching.ArrayStack[gameplay.Coord]) {
	for i := uint8(0); i < b.Height; i++ {
		for j := uint8(0); j < b.Width; j++ {
			if b.IsLegalMove(i, j) {
				coord_out.Push(gameplay.Coord{
					Row:    i,
					Column: j,
				})
			}
		}
	}
}
