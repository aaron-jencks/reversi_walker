package main

import (
	"context"
	"flag"
	"math/rand"
	"os"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/aaron-jencks/reversi/gameplay"
	"github.com/aaron-jencks/reversi/utils/uint128"
	"github.com/aaron-jencks/reversi/visiting"
	"github.com/aaron-jencks/reversi/walking"

	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func main() {
	p := message.NewPrinter(language.English)

	var hashtype string = "spiral"
	var cachetype string = "simple"
	var lcachetype string = "simple"
	var cpu_profile_file string = "cpuprofile.prof"
	var mem_profile_file string = "memprofile.mprof"
	var walk_time time.Duration = 30 * time.Second
	var hash_count int = 30000000
	var purge_interval time.Duration = 0
	var procs int = 1

	flag.StringVar(&hashtype, "hash", hashtype, "specifies the hash function to use for the test, must be one of (simple, spiral, linear)")
	flag.StringVar(&cachetype, "cache", cachetype, "specifies the type of cache to use for the global cache, must be one of (simple)")
	flag.StringVar(&lcachetype, "lcache", lcachetype, "specifies the type of cache to use for the local walker, must be one of (simple)")
	flag.StringVar(&cpu_profile_file, "cpuprofile", cpu_profile_file, "specifies where to save the cpu profile to")
	flag.StringVar(&mem_profile_file, "memprofile", mem_profile_file, "specifies where to save the memory profile to")
	flag.DurationVar(&walk_time, "walktime", walk_time, "specifies how long to run the walker for for statistics evaluation")
	flag.IntVar(&hash_count, "hashcount", hash_count, "specifies how many boards to hash/unhash for the hashing portion of the test")
	flag.DurationVar(&purge_interval, "purge", purge_interval, "specifies how often to purge the local cache during the test, if 0 then it's set to the walk time +1s")
	flag.IntVar(&procs, "procs", procs, "specifies how many walkers to launch for the test")
	flag.Parse()

	if uint64(purge_interval) == 0 {
		purge_interval = walk_time + 1*time.Second
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
	} else if mem_profile_file != "" {
		defer func() {
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
		}()
	}

	// TODO final board state rate
	// TODO overall board state rate
	// TODO average hash and unhash rate

	gameplay.SetBoardHashFunc(hashtype)
	gameplay.SetBoardUnHashFunc(hashtype)

	p.Println("starting walking test")
	frate, erate, rrate := runWalkingTest(cachetype, lcachetype, procs, purge_interval, walk_time)
	p.Printf("walking test results: (final: %f b/s, explored: %f b/s, repeated: %f b/s)\n", frate, erate, rrate)

	p.Println("starting hash throughput test")
	boards := make([]gameplay.Board, hash_count)
	for bi := range boards {
		boards[bi] = generateRandomBoard()
	}

	tstart := time.Now()
	hashes := make([]uint128.Uint128, hash_count)
	for bi, b := range boards {
		hashes[bi] = b.Hash()
	}
	elapsed := time.Since(tstart)
	p.Printf("hash throughput: elapsed: %s, rate: %f h/s\n", elapsed, float64(hash_count)/elapsed.Seconds())

	tstart = time.Now()
	for hi, h := range hashes {
		boards[hi] = gameplay.CreateUnhashBoard(6, h)
	}
	elapsed = time.Since(tstart)
	p.Printf("unhash throughput: elapsed: %s, rate: %f h/s\n", elapsed, float64(hash_count)/elapsed.Seconds())
}

func runWalkingTest(cachetype, lcachetype string, procs int, purge_interval, walk_time time.Duration) (float64, float64, float64) {
	p := message.NewPrinter(language.English)

	var final, explored, repeated, finished uint64
	finished_lock := sync.RWMutex{}

	var cache visiting.VisitedCache
	switch cachetype {
	case "simple":
		cache = visiting.CreateSimpleVisitedCache()
	default:
		p.Println("unrecognized cache type:", cachetype)
		return -1, -1, -1
	}

	wmeta := walking.WalkerMetaData{
		Visited:         cache,
		Counter:         &final,
		Explored:        &explored,
		Repeated:        &repeated,
		Finished_count:  &finished,
		Finished_lock:   &finished_lock,
		Update_interval: 50 * time.Millisecond,
		Purge_interval:  purge_interval,
	}

	initialBoards := walking.FindInitialBoards(uint(procs), 6)

	p.Printf("created %d boards for %d processors\n", len(initialBoards), procs)

	ctx, can := context.WithCancel(context.Background())

	var frate, erate, rrate float64
	monitchan := make(chan bool)

	go func() {
		var frates, erates, rrates []float64
		var prevfinal, prevexplored, prevrepeated uint64
		poll_interval := 100 * time.Millisecond

		avg := func(rates []float64) float64 {
			var result float64
			for _, r := range rates {
				result += r
			}
			return result / float64(len(rates))
		}

		ticker := time.NewTicker(poll_interval)
		should_exit := false
		for {
			select {
			case <-ctx.Done():
				should_exit = true
			case <-ticker.C:
				cache.RLock()
				fdiff := final - prevfinal
				ediff := explored - prevexplored
				rdiff := repeated - prevrepeated
				cache.RUnlock()
				frates = append(frates, float64(fdiff)/poll_interval.Seconds())
				erates = append(erates, float64(ediff)/poll_interval.Seconds())
				rrates = append(rrates, float64(rdiff)/poll_interval.Seconds())
				if should_exit {
					frate = avg(frates)
					erate = avg(erates)
					rrate = avg(rrates)
					close(monitchan)
					return
				}
			}
		}
	}()

	walkers := make([]*walking.BoardWalker, len(initialBoards))
	fchans := make([]chan *os.File, len(initialBoards))
	rchans := make([]chan bool, len(initialBoards))

	for wi, ib := range initialBoards {
		if wi >= procs {
			break
		}

		fchans[wi] = make(chan *os.File)
		rchans[wi] = make(chan bool)
		walker := walking.CreateWalkerFromMeta(uint32(wi), fchans[wi], rchans[wi], wmeta)
		walker.Enable_saving = false
		walkers[wi] = &walker
		go walkers[wi].Walk(ctx, ib)
	}

	time.Sleep(walk_time)
	can()
	<-monitchan
	return frate, erate, rrate
}

func generateRandomBoard() gameplay.Board {
	pb := gameplay.CreateBoard(gameplay.BoardValue(rand.Intn(2))+1, 6)
	bcenter := uint8(3)
	for r := uint8(0); r < 4; r++ {
		for c := uint8(0); c < 4; c++ {
			if (r == bcenter && c == bcenter) ||
				(r == bcenter-1 && c == bcenter-1) ||
				(r == bcenter-1 && c == bcenter) ||
				(r == bcenter && c == bcenter-1) {
				// if it's one of the center squares, skip it for now
				continue
			}
			pb.Put(r, c, gameplay.BoardValue(rand.Intn(3)))
		}
	}

	pb.Put(bcenter, bcenter, gameplay.BoardValue(rand.Intn(2)+1))
	pb.Put(bcenter-1, bcenter-1, gameplay.BoardValue(rand.Intn(2)+1))
	pb.Put(bcenter, bcenter-1, gameplay.BoardValue(rand.Intn(2)+1))
	pb.Put(bcenter-1, bcenter, gameplay.BoardValue(rand.Intn(2)+1))

	return pb
}
