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

	var checkpoint_path string = "./checkpoint.bin"
	flag.StringVar(&checkpoint_path, "check", checkpoint_path, "indicates where to save checkpoints")
	flag.Parse()

	var counter, explored, repeated uint64 = 0, 0, 0
	var clock, elock, rlock sync.RWMutex = sync.RWMutex{}, sync.RWMutex{}, sync.RWMutex{}

	ctx, can := context.WithCancel(context.Background())

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	initialBoards := walking.FindInitialBoards(uint(runtime.NumCPU()), 8)

	p.Printf("Created %d boards for %d processors\n", len(initialBoards), runtime.NumCPU())

	save_chan := make(chan bool)

	for wi, ib := range initialBoards {
		bw := walking.BoardWalker{
			Identifier:    uint32(wi),
			Counter:       &counter,
			Counter_lock:  &clock,
			Explored:      &explored,
			Explored_lock: &elock,
			Repeated:      &repeated,
			Repeated_lock: &rlock,
			Visited:       visiting.CreateSimpleVisitedCache(),
		}
		go bw.Walk(ctx, save_chan, ib)
	}

	prev_explored := explored
	for {
		select {
		case <-sigs:
			can()
		case <-ctx.Done():
			clock.RLock()
			elock.RLock()
			rlock.RLock()
			p.Printf("\rfinal counts %d found %d explored %d repeated\n", counter, explored, repeated)
			clock.RUnlock()
			elock.RUnlock()
			rlock.RUnlock()
			return
		default:
			clock.RLock()
			elock.RLock()
			rlock.RLock()
			erate := uint64(float64(explored-prev_explored) / 0.1)
			p.Printf("\r%d found %d explored %d repeated @ %d boards/sec", counter, explored, repeated, erate)
			prev_explored = explored
			clock.RUnlock()
			elock.RUnlock()
			rlock.RUnlock()
			time.Sleep(100 * time.Millisecond)
		}
	}
}
