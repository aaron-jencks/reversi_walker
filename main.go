package main

import (
	"context"
	"flag"
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
	var walker_update_interval uint = 10000

	flag.StringVar(&checkpoint_path, "check", checkpoint_path, "indicates where to save checkpoints, defaults to ./checkpoint.bin")
	flag.UintVar(&procs, "procs", procs, "specifies how many threads to use for processing, defaults to 2*cpu cores")
	flag.UintVar(&ubsize, "size", ubsize, "specifies the size of the board to run the program on, defaults to 8")
	flag.DurationVar(&display_poll, "display", display_poll, "specifies how often to update the statistics in the terminal, defaults to 100 ms")
	flag.UintVar(&walker_update_interval, "walkupdate", walker_update_interval, "specifies how often the walkers should update the counters, defaults to every 10000 boards")
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

	for wi, ib := range initialBoards {
		bw := walking.BoardWalker{
			Identifier:     uint32(wi),
			Counter:        &counter,
			Counter_lock:   &clock,
			Explored:       &explored,
			Repeated:       &repeated,
			Visited:        visiting.CreateSimpleVisitedCache(),
			Finished_count: &finished,
			Finished_lock:  &finlock,
		}
		go bw.Walk(ctx, walker_update_interval, ib)
	}

	prev_explored := explored
	for {
		select {
		case <-sigs:
			can()
		case <-ctx.Done():
			clock.RLock()
			p.Printf("\rfinal counts %d found %d explored %d repeated\n", counter, explored, repeated)
			clock.RUnlock()
			return
		default:
			clock.RLock()
			finlock.RLock()
			erate := uint64(float64(explored-prev_explored) / display_poll.Seconds())
			tfinished := finished
			p.Printf("\r%d found %d explored %d repeated %d finished @ %d boards/sec", counter, explored, repeated, finished, erate)
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
