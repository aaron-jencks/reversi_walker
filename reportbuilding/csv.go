package reportbuilding

import (
	"fmt"
	"io"

	"github.com/aaron-jencks/reversi/gameplay"
	"github.com/aaron-jencks/reversi/visiting"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func boardValueToString(bv gameplay.BoardValue) string {
	switch bv {
	case gameplay.BOARD_BLACK:
		return "black"
	case gameplay.BOARD_WHITE:
		return "white"
	case gameplay.BOARD_EMPTY:
		return "empty"
	}
	return "invalid"
}

func BuildCSVReport(writer io.Writer, cache visiting.VisitedCache, size uint8) error {
	p := message.NewPrinter(language.English)
	p.Printf("Creating a csv report for boards of size %d\n", size)

	header := "player"
	for hi := uint8(0); hi < size*size; hi++ {
		row := hi / size
		col := hi % size
		header += fmt.Sprintf(",%c%d", 'a'+col, row)
	}
	header += "\n"
	_, err := writer.Write([]byte(header))
	if err != nil {
		return err
	}

	keys := cache.Keys()
	for ki, k := range keys {
		p.Printf("\rProcessing entry %d/%d", ki, len(keys))
		b := gameplay.CreateUnhashBoard(size, k)
		entry := fmt.Sprintf("\"%s\"", boardValueToString(b.Player))
		for r := uint8(0); r < size; r++ {
			for c := uint8(0); c < size; c++ {
				entry += fmt.Sprintf(",%d", b.Get(r, c))
			}
		}
		entry += "\n"
		_, err = writer.Write([]byte(entry))
		if err != nil {
			p.Println()
			return err
		}
	}
	p.Println()
	return nil
}
