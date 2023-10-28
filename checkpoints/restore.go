package checkpoints

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/aaron-jencks/reversi/caching"
	"github.com/aaron-jencks/reversi/gameplay"
	"github.com/aaron-jencks/reversi/utils"
	"github.com/aaron-jencks/reversi/walking"
)

type WalkerWChan struct {
	Walker walking.BoardWalker
	Fchan  chan *os.File
	Rchan  chan bool
}

func RestoreSimulation(ctx context.Context, filename string, size uint8, procs uint, meta walking.WalkerMetaData, tstart *time.Time) ([]WalkerWChan, error) {
	fmt.Printf("Starting restore from %s\n", filename)
	fp, err := os.OpenFile(filename, os.O_RDONLY, 0777)
	if err != nil {
		return nil, err
	}
	defer fp.Close()

	// reusable buffer for reading uint64 data
	i64buff := make([]byte, 8)

	_, err = fp.Read(i64buff)
	if err != nil {
		return nil, err
	}
	vlen := utils.Uint64FromBytes(i64buff)
	vbuff := make([]byte, vlen)
	_, err = fp.Read(vbuff)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Checkpoint version %s\n", string(vbuff))

	_, err = fp.Read(i64buff)
	if err != nil {
		return nil, err
	}
	*meta.Counter = utils.Uint64FromBytes(i64buff)

	_, err = fp.Read(i64buff)
	if err != nil {
		return nil, err
	}
	*meta.Explored = utils.Uint64FromBytes(i64buff)

	_, err = fp.Read(i64buff)
	if err != nil {
		return nil, err
	}
	*meta.Repeated = utils.Uint64FromBytes(i64buff)

	// read start time
	_, err = fp.Read(i64buff)
	if err != nil {
		return nil, err
	}
	tbinlen := utils.Uint64FromBytes(i64buff)
	tbinbuff := make([]byte, tbinlen)
	_, err = fp.Read(tbinbuff)
	if err != nil {
		return nil, err
	}
	err = tstart.UnmarshalBinary(tbinbuff)
	if err != nil {
		return nil, err
	}

	err = meta.Visited.FromFile(fp)
	if err != nil {
		return nil, err
	}

	fmt.Println("Restored counters, start time and cache, reading boards...")
	// the rest of the file is boards

	// first figure out how many boards there are
	noff, err := fp.Seek(0, 1)
	if err != nil {
		return nil, err
	}
	stats, err := fp.Stat()
	if err != nil {
		return nil, err
	}
	bsize := stats.Size()
	board_count := (bsize - noff) / 16

	var boards []gameplay.Board = make([]gameplay.Board, board_count)
	i128buff := make([]byte, 16)
	_, err = fp.Read(i128buff)
	for bi := int64(0); bi < board_count && err == nil; bi++ {
		bh := utils.Uint128FromBytes(i128buff)
		b := gameplay.CreateUnhashBoard(size, bh)
		boards[bi] = b
		fmt.Printf("\rLoaded %d/%d boards", bi+1, board_count)
	}

	fmt.Println("\nFinished restoring boards, creating walkers")
	walkers := make([]WalkerWChan, procs)

	for wi := range walkers {
		fc := make(chan *os.File)
		rc := make(chan bool)
		walkers[wi].Walker = walking.CreateWalkerFromMeta(uint32(wi), fc, rc, meta)
		walkers[wi].Fchan = fc
		walkers[wi].Rchan = rc
	}

	fmt.Println("Created walkers, distributing boards")

	bpw := uint(len(boards)) / procs
	eb := uint(len(boards)) % procs
	fmt.Printf("With %d walkers and %d boards, that's %d boards per walker with %d extra", procs, len(boards), bpw, eb)

	boff := uint(0)
	for wi := range walkers {
		board_cache := caching.CreatePointerCache[gameplay.Board](5000, func() gameplay.Board {
			return gameplay.CreateBoard(gameplay.BOARD_BLACK, size, size)
		})

		stack := caching.CreateArrayStack[walking.WalkerBoardWIndex](1830)

		for wb := uint(0); wb < bpw; wb++ {
			sbi, sb := board_cache.Get()
			boards[boff+wb].CloneInto(sb)
			stack.Push(walking.WalkerBoardWIndex{
				Board: sb,
				Index: sbi,
			})
		}

		boff += bpw
		if uint(wi) < eb {
			sbi, sb := board_cache.Get()
			boards[boff].CloneInto(sb)
			stack.Push(walking.WalkerBoardWIndex{
				Board: sb,
				Index: sbi,
			})
			boff++
		}

		go walkers[wi].Walker.WalkPrestacked(ctx, &board_cache, &stack, size)
	}

	fmt.Println("Finished restoring state")
	return walkers, nil
}
