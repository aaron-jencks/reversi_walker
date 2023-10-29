package main

import (
	"flag"
	"os"

	"github.com/aaron-jencks/reversi/checkpoints"
	"github.com/aaron-jencks/reversi/gameplay"
	"github.com/aaron-jencks/reversi/utils"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func main() {
	p := message.NewPrinter(language.English)
	var restore_file string = "../../checkpoint.bin"
	var hash_string string = "simple"
	var unhash_string string = "simple"
	var bsize uint = 8
	flag.StringVar(&restore_file, "file", restore_file, "specifies the file to parse, defaults to ../../checkpoint.bin")
	flag.StringVar(&hash_string, "hash", hash_string, "specifies which hash function to use for rehashing the boards, must be within (simple, spiral), defaults to simple")
	flag.StringVar(&unhash_string, "unhash", unhash_string, "specifies which hash function to use for unhashing the boards, must be within (simple, spiral), defaults to simple")
	flag.UintVar(&bsize, "size", bsize, "specifies the size of the board, defaults to 8")
	flag.Parse()

	var hash gameplay.BoardHashFunc
	var unhash gameplay.BoardUnhashFunc

	switch hash_string {
	case "simple":
		hash = gameplay.SimpleHash
	case "spiral":
		hash = gameplay.SpiralHash
	default:
		p.Printf("Unknown hash method: %s\n", hash_string)
		return
	}

	switch unhash_string {
	case "simple":
		unhash = gameplay.SimpleUnhashBoard
	case "spiral":
		unhash = gameplay.SpiralUnhashBoard
	default:
		p.Printf("Unknown unhash method: %s\n", unhash_string)
		return
	}

	stats, err := checkpoints.CheckpointStats(restore_file)
	if err != nil {
		p.Printf("Failed to read %s: %s\n", restore_file, err.Error())
		return
	}
	p.Println(stats)

	bcount := stats.CacheSize + stats.WalkerBoards

	p.Printf("Converting %d boards\n", bcount)
	fp, err := os.OpenFile(restore_file, os.O_RDWR, 0777)
	if err != nil {
		p.Printf("Failed to open checkpoint %s for reading and writing: %s\n", restore_file, err.Error())
		return
	}
	defer fp.Close()

	// reusable buffer for reading uint64 data
	i128buff := make([]byte, 16)

	// the file offset where the boards begin
	bstart := int64(stats.VLen) + int64(stats.TimeSize) + 48

	// move to start of boards
	_, err = fp.Seek(bstart, 0)
	if err != nil {
		p.Printf("Failed to reposition fp to %d: %s\n", bstart, err.Error())
		return
	}

	// read in all of the boards
	boards := make([]gameplay.Board, bcount)
	for bi := range boards {
		_, err := fp.Read(i128buff)
		if err != nil {
			p.Printf("Failed to read in board %d: %s\n", bi, err.Error())
			return
		}
		boards[bi] = unhash(uint8(bsize), utils.Uint128FromBytes(i128buff))
		p.Printf("\rUnhashed %d/%d boards", bi+1, bcount)
	}

	p.Println("\nFinished reading boards, rehashing")

	// move back to start of boards
	_, err = fp.Seek(bstart, 0)
	if err != nil {
		p.Printf("Failed to reposition fp to %d: %s\n", bstart, err.Error())
		return
	}

	for bi, b := range boards {
		bh := hash(b)
		_, err := fp.Write(utils.Uint128ToBytes(bh))
		if err != nil {
			p.Printf("Failed to rewrite board %d: %s\n", bi, err.Error())
			return
		}
		p.Printf("\rRehashed %d/%d boards", bi+1, bcount)
	}

	p.Println("\nFinished rehashing checkpoint file")
}
