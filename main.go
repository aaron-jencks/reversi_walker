package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"sync"
	"syscall"
	"time"

	"github.com/aaron-jencks/reversi/checkpoints"
	"github.com/aaron-jencks/reversi/visiting"
	"github.com/aaron-jencks/reversi/walking"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func main() {
	p := message.NewPrinter(language.English)

	var checkpoint_path string = "./checkpoint.bin"
	var procs uint = uint(runtime.NumCPU())
	var ubsize uint = 8
	var display_poll time.Duration = 2 * time.Second
	var save_interval time.Duration = 30 * time.Minute
	var cache_update_interval time.Duration = time.Second
	var cpu_profile_file string = ""
	var mem_profile_file string = ""
	var restore_file string = ""

	flag.StringVar(&checkpoint_path, "check", checkpoint_path, "indicates where to save checkpoints, defaults to ./checkpoint.bin")
	flag.UintVar(&procs, "procs", procs, "specifies how many threads to use for processing, defaults to the number of cpu cores")
	flag.UintVar(&ubsize, "size", ubsize, "specifies the size of the board to run the program on, defaults to 8")
	flag.DurationVar(&display_poll, "display", display_poll, "specifies how often to update the statistics in the terminal, defaults to 2s")
	flag.DurationVar(&save_interval, "save", save_interval, "specifies how often to save the state of the walker, defaults to 30m")
	flag.DurationVar(&cache_update_interval, "walkupdate", cache_update_interval, "specifies how often the walkers should take their cached data and sync it with the cache, defaults to 1s")
	flag.StringVar(&cpu_profile_file, "cpuprofile", cpu_profile_file, "specifies where to save pprof data to if supplied, leave empty to disable")
	flag.StringVar(&mem_profile_file, "memprofile", mem_profile_file, "specifies where to save the pprof memory data to if supplied, leave empty to disable")
	flag.StringVar(&restore_file, "restore", restore_file, "specifies where to restore the simulation from if supplied, leave empty to start fresh")
	flag.Parse()

	if ubsize > 255 {
		p.Printf("board size must be small enough to fit within one byte (<= 255), received: %d\n", ubsize)
		return
	}

	if cpu_profile_file != "" && mem_profile_file == cpu_profile_file {
		p.Printf("cpu profile and memory profile cannot be written to the same file!\n")
		return
	}

	if cpu_profile_file != "" {
		f, err := os.Create(cpu_profile_file)
		if err != nil {
			p.Printf("failed to start cpu profiling: %s\n", err.Error())
			return
		}
		pprof.StartCPUProfile(f)
		defer func() {
			if mem_profile_file != "" {
				f, err := os.Create(mem_profile_file)
				if err != nil {
					p.Printf("failed to start memory profiling: %s\n", err.Error())
					return
				}
				pprof.WriteHeapProfile(f)
				err = f.Close()
				if err != nil {
					p.Printf("failed to close memory profile file: %s\n", err.Error())
				}
			}
			pprof.StopCPUProfile()
		}()
	}

	bsize := uint8(ubsize)

	var counter, explored, repeated, finished uint64 = 0, 0, 0, 0
	var finlock sync.RWMutex = sync.RWMutex{}

	ctx, can := context.WithCancel(context.Background())

	tstart := time.Now()

	cache := visiting.CreateSimpleVisitedCache()

	var walkers []*walking.BoardWalker
	var fchans []chan *os.File
	var rchans []chan bool

	// check if we need to restore the state and do so now
	if restore_file != "" {
		p.Printf("Restoring from checkpoint %s", restore_file)

		meta := walking.WalkerMetaData{
			Visited:         cache,
			Counter:         &counter,
			Explored:        &explored,
			Repeated:        &repeated,
			Finished_count:  &finished,
			Finished_lock:   &finlock,
			Update_interval: cache_update_interval,
		}
		walker_data, err := checkpoints.RestoreSimulation(ctx, restore_file, bsize, procs, meta, &tstart)
		if err != nil {
			p.Printf("Failed to restore state from checkpoint %s: %s\n", restore_file, err.Error())
			return
		}

		walkers = make([]*walking.BoardWalker, len(walker_data))
		fchans = make([]chan *os.File, len(walker_data))
		rchans = make([]chan bool, len(walker_data))

		for wi, w := range walker_data {
			walkers[wi] = &w.Walker
			fchans[wi] = w.Fchan
			rchans[wi] = w.Rchan
		}

		p.Printf("Restoring from checkpoint %s complete", restore_file)
	} else {
		// no restore point supplied, begin as normal

		initialBoards := walking.FindInitialBoards(procs, bsize)

		p.Printf("Created %d boards for %d processors\n", len(initialBoards), procs)

		walkers = make([]*walking.BoardWalker, len(initialBoards))
		fchans = make([]chan *os.File, len(initialBoards))
		rchans = make([]chan bool, len(initialBoards))

		for wi, ib := range initialBoards {
			fchans[wi] = make(chan *os.File)
			rchans[wi] = make(chan bool)
			walkers[wi] = &walking.BoardWalker{
				Identifier:      uint32(wi),
				Counter:         &counter,
				Explored:        &explored,
				Repeated:        &repeated,
				Visited:         cache,
				Finished_count:  &finished,
				Finished_lock:   &finlock,
				File_chan:       fchans[wi],
				Ready_chan:      rchans[wi],
				Update_interval: cache_update_interval,
			}
			go walkers[wi].Walk(ctx, ib)
		}
	}

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	prev_explored := explored
	save_ticker := time.NewTicker(save_interval)
	for {
		select {
		case <-sigs:
			can()
		case <-ctx.Done():
			// execution ended
			p.Printf("\nTermination signal received\n")
			walking.PauseWalkers(len(walkers))
			cache.RLock()
			p.Printf("\r[%s] final counts %d found %d explored %d repeated\n", time.Since(tstart), counter, explored, repeated)
			vcounter := counter
			vexplored := explored
			vrepeated := repeated
			cache.RUnlock()
			p.Printf("\nSaving walkers\n")
			err := checkpoints.SaveSimulation(checkpoint_path, fchans, rchans, cache, vcounter, vexplored, vrepeated, tstart)
			if err != nil {
				p.Printf("failed to save state: %s\n", err.Error())
			}
			return
		case <-save_ticker.C:
			// save progress
			walking.PauseWalkers(len(walkers))
			p.Printf("\nSaving walkers\n")
			cache.RLock()
			vcounter := counter
			vexplored := explored
			vrepeated := repeated
			cache.RUnlock()
			err := checkpoints.SaveSimulation(checkpoint_path, fchans, rchans, cache, vcounter, vexplored, vrepeated, tstart)
			if err != nil {
				p.Printf("failed to save state: %s\n", err.Error())
			}
			walking.UnpauseWalkers()
		default:
			cache.RLock()
			finlock.RLock()
			erate := uint64(float64(explored-prev_explored) / display_poll.Seconds())
			tfinished := finished
			p.Printf("\r[%s] %d found %d explored %d repeated %d finished @ %d boards/sec",
				time.Since(tstart), counter, explored, repeated, finished, erate)
			prev_explored = explored
			cache.RUnlock()
			finlock.RUnlock()
			if uint(tfinished) == procs {
				p.Printf("\nall walkers exited, quitting\n")
				can()
			}
			time.Sleep(display_poll)
		}
	}
}
