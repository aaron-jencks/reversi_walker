package main

import (
	"flag"

	"github.com/aaron-jencks/reversi/checkpoints"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func main() {
	p := message.NewPrinter(language.English)
	restore_file := flag.String("file", "../../checkpoint.bin", "specifies the file to parse, defaults to ../../checkpoint.bin")
	flag.Parse()
	stats, err := checkpoints.CheckpointStats(*restore_file)
	if err != nil {
		p.Printf("Failed to read %s: %s\n", *restore_file, err.Error())
		return
	}
	p.Println(stats)
}
