package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/aaron-jencks/reversi/gameplay"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

var texTemplate string = `\documentclass{article}

\usepackage{othelloboard}

\begin{document}

%s

\end{document}
`

func main() {
	p := message.NewPrinter(language.English)
	var csv_file string = "../../final.csv"
	var out_file string = "./final.tex"
	flag.StringVar(&csv_file, "file", csv_file, "specifies the file to parse")
	flag.StringVar(&out_file, "out", out_file, "specifies the file to output latex to")
	flag.Parse()

	data, err := os.ReadFile(csv_file)
	if err != nil {
		p.Printf("failed to read file %s: %s\n", csv_file, err.Error())
		return
	}

	sdata := string(data)

	entries := strings.Split(sdata, "\n")
	header := entries[0]
	bsize := uint8(math.Sqrt(float64(len(strings.Split(header, ",")) - 1)))

	boards := make([]string, len(entries)-1)

	for ei, entry := range entries[1:] {
		p.Printf("\rProcessing board %d/%d", ei+1, len(entries[1:]))
		columns := strings.Split(entry, ",")
		player := strings.Trim(columns[0], "\"")
		pbv := gameplay.BOARD_BLACK
		if player == "white" {
			pbv = gameplay.BOARD_WHITE
		}

		// parse board
		b := gameplay.CreateEmptyBoard(pbv, bsize, bsize)
		for ci := range columns[1:] {
			row := uint8(ci / int(bsize))
			col := uint8(ci % int(bsize))
			var v uint8
			_, err = fmt.Sscan(columns[ci+1], &v)
			if err != nil {
				p.Printf("\nfailed to parse (%d, %d) of board with value %s: %s\n", row, col, columns[ci], err.Error())
				return
			}
			b.Put(row, col, gameplay.BoardValue(v))
		}

		// create othelloboard representation
		bstring := "\\begin{othelloboard}{1}\n\\dotmarkings\n"

		offset := 0
		if bsize < 8 {
			offset = (8 - int(bsize)) >> 1
		}

		for i := 0; i < offset; i++ {
			bstring += generateEmptyOthelloRowCommand(i)
		}

		for r := uint8(0); r < bsize; r++ {
			arr := []int{}
			for i := 0; i < offset; i++ {
				arr = append(arr, 0)
			}
			for c := uint8(0); c < bsize; c++ {
				arr = append(arr, int(b.Get(r, c)))
			}
			for i := 0; i < offset; i++ {
				arr = append(arr, 0)
			}
			bstring += generateOthelloRowCommand(offset+int(r), arr)
		}

		for i := 0; i < offset; i++ {
			bstring += generateEmptyOthelloRowCommand(offset + int(bsize) + i)
		}

		bstring += "\\end{othelloboard}"

		boards[ei] = bstring
	}

	p.Println("\nFinished processing boards")

	fbstring := strings.Join(boards, "\n")

	err = os.WriteFile(out_file, []byte(fmt.Sprintf(texTemplate, fbstring)), 0777)
	if err != nil {
		p.Printf("failed to write out latex file to %s: %s\n", out_file, err.Error())
	}

	p.Printf("Generated latex information to %s\n", out_file)
}

func generateEmptyOthelloRowCommand(row int) string {
	return generateOthelloRowCommand(row, []int{0, 0, 0, 0, 0, 0, 0, 0})
}

func generateOthelloRowCommand(row int, entries []int) string {
	return fmt.Sprintf("\\othelloarray%srow {%d}{%d}{%d}{%d}{%d}{%d}{%d}{%d}\n", generateOthelloRowName(row),
		entries[0], entries[1], entries[2], entries[3], entries[4], entries[5], entries[6], entries[7])
}

func generateOthelloRowName(row int) string {
	switch row {
	case 0:
		return "first"
	case 1:
		return "second"
	case 2:
		return "third"
	case 3:
		return "fourth"
	case 4:
		return "fifth"
	case 5:
		return "sixth"
	case 6:
		return "seventh"
	case 7:
		return "eighth"
	default:
		return "invalid"
	}
}
