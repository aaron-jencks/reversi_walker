package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"

	"github.com/aaron-jencks/reversi/visiting"
	"github.com/aaron-jencks/reversi/walking"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func main() {
	p := message.NewPrinter(language.English)

	// TODO implement saving daemon
	var checkpoint_path string = "./checkpoint.bin"
	var procs uint = uint(runtime.NumCPU()) << 1
	var ubsize uint = 8
	var display_poll time.Duration = 100 * time.Millisecond
	var walker_update_interval uint = 100000
	var save_interval time.Duration = 5 * time.Minute

	flag.StringVar(&checkpoint_path, "check", checkpoint_path, "indicates where to save checkpoints, defaults to ./checkpoint.bin")
	flag.UintVar(&procs, "procs", procs, "specifies how many threads to use for processing, defaults to 2*cpu cores")
	flag.UintVar(&ubsize, "size", ubsize, "specifies the size of the board to run the program on, defaults to 8")
	flag.DurationVar(&display_poll, "display", display_poll, "specifies how often to update the statistics in the terminal, defaults to 100 ms")
	flag.UintVar(&walker_update_interval, "walkupdate", walker_update_interval, "specifies how often the walkers should update the counters, defaults to every 100000 boards")
	flag.DurationVar(&save_interval, "save", save_interval, "specifies how often to save the state of the walker, defaults to 5m")
	flag.Parse()

	if ubsize > 255 {
		p.Printf("board size must be small enough to fit within one byte (<= 255), received: %d\n", ubsize)
		return
	}

	bsize := uint8(ubsize)

	var counter, explored, repeated, finished uint64 = 0, 0, 0, 0
	var clock, finlock sync.RWMutex = sync.RWMutex{}, sync.RWMutex{}

	ctx, can := context.WithCancel(context.Background())

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	initialBoards := walking.FindInitialBoards(procs, bsize)

	p.Printf("Created %d boards for %d processors\n", len(initialBoards), procs)

	cache := visiting.CreateSimpleVisitedCache()

	walkers := make([]*walking.BoardWalker, len(initialBoards))
	fchans := make([]chan *os.File, len(initialBoards))
	rchans := make([]chan bool, len(initialBoards))

	for wi, ib := range initialBoards {
		fchans[wi] = make(chan *os.File)
		rchans[wi] = make(chan bool)
		walkers[wi] = &walking.BoardWalker{
			Identifier:     uint32(wi),
			Counter:        &counter,
			Counter_lock:   &clock,
			Explored:       &explored,
			Repeated:       &repeated,
			Visited:        cache,
			Finished_count: &finished,
			Finished_lock:  &finlock,
			File_chan:      fchans[wi],
			Ready_chan:     rchans[wi],
		}
		go walkers[wi].Walk(ctx, walker_update_interval, ib)
	}

	prev_explored := explored
	tstart := time.Now()
	save_ticker := time.NewTicker(save_interval)
	for {
		select {
		case <-sigs:
			can()
		case <-ctx.Done():
			// execution ended
			clock.RLock()
			p.Printf("\r[%s] final counts %d found %d explored %d repeated\n", time.Since(tstart), counter, explored, repeated)
			clock.RUnlock()
			err := save_state(checkpoint_path, fchans, rchans)
			if err != nil {
				p.Printf("failed to save state: %s\n", err.Error())
			}
			return
		case <-save_ticker.C:
			// save progress
			err := save_state(checkpoint_path, fchans, rchans)
			if err != nil {
				p.Printf("failed to save state: %s\n", err.Error())
			}
		default:
			clock.RLock()
			finlock.RLock()
			erate := uint64(float64(explored-prev_explored) / display_poll.Seconds())
			tfinished := finished
			p.Printf("\r[%s] %d found %d explored %d repeated %d finished @ %d boards/sec",
				time.Since(tstart), counter, explored, repeated, finished, erate)
			prev_explored = explored
			clock.RUnlock()
			finlock.RUnlock()
			if uint(tfinished) == procs {
				p.Printf("\nall walkers exited, quitting\n")
				can()
			}
			time.Sleep(display_poll)
		}
	}
}

func save_state(fname string, fchans []chan *os.File, rchans []chan bool) error {
	fp, err := os.OpenFile(fname, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0777)
	if err != nil {
		return err
	}
	for wi, fchan := range fchans {
		fchan <- fp
		v := <-rchans[wi]
		if !v {
			return fmt.Errorf("failed to save writer %d", wi)
		}
	}
	return nil
}
