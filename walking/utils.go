package walking

import "github.com/aaron-jencks/reversi/gameplay"

func findNextBoards(b gameplay.Board) []gameplay.Coord {
	result := make([]gameplay.Coord, 0, b.Height*b.Width)

	for i := uint8(0); i < b.Height; i++ {
		for j := uint8(0); j < b.Width; j++ {
			if b.IsLegalMove(i, j) {
				result = append(result, gameplay.Coord{
					Row:    i,
					Column: j,
				})
			}
		}
	}

	return result
}
