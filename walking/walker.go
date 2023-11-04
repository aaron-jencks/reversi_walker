package walking

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/aaron-jencks/reversi/caching"
	"github.com/aaron-jencks/reversi/gameplay"
	"github.com/aaron-jencks/reversi/utils/uint128"
	"github.com/aaron-jencks/reversi/visiting"
)

type WalkerMetaData struct {
	Visited         visiting.VisitedCache
	Counter         *uint64
	Explored        *uint64
	Repeated        *uint64
	Finished_count  *uint64
	Finished_lock   *sync.RWMutex
	Update_interval time.Duration
	Purge_interval  time.Duration
}

// represents a processor that walks the game tree
type BoardWalker struct {
	Identifier      uint32
	Visited         visiting.VisitedCache
	Counter         *uint64
	Explored        *uint64
	Repeated        *uint64
	File_chan       <-chan *os.File
	Ready_chan      chan<- bool
	Finished_count  *uint64
	Finished_lock   *sync.RWMutex
	Update_interval time.Duration
	Purge_interval  time.Duration
}

func CreateWalkerFromMeta(identifier uint32, file_chan <-chan *os.File, ready_chan chan<- bool, meta WalkerMetaData) BoardWalker {
	return BoardWalker{
		Identifier:      identifier,
		Visited:         meta.Visited,
		Counter:         meta.Counter,
		Explored:        meta.Explored,
		Repeated:        meta.Repeated,
		File_chan:       file_chan,
		Ready_chan:      ready_chan,
		Finished_count:  meta.Finished_count,
		Finished_lock:   meta.Finished_lock,
		Update_interval: meta.Update_interval,
		Purge_interval:  meta.Purge_interval,
	}
}

var SAVING bool = false
var PAUSED_COUNT int = 0
var SAVING_LOCK sync.RWMutex = sync.RWMutex{}

func PauseWalkers(wc int) {
	fmt.Println("Pausing walkers")
	SAVING_LOCK.Lock()
	SAVING = true
	PAUSED_COUNT = 0
	SAVING_LOCK.Unlock()
	fmt.Println("Set saving flag")
	for {
		SAVING_LOCK.RLock()
		pc := PAUSED_COUNT
		SAVING_LOCK.RUnlock()
		fmt.Printf("\rWaiting for walkers to pause %d/%d", pc, wc)
		if pc == wc {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Println("\nFinished pausing walkers")
}

func UnpauseWalkers() {
	fmt.Println("Unpausing walkers")
	SAVING_LOCK.Lock()
	SAVING = false
	PAUSED_COUNT = 0
	SAVING_LOCK.Unlock()
	fmt.Println("Finished unpausing walkers")
}

type WalkerBoardWIndex struct {
	Board *gameplay.Board
	Index int
}

func (bw BoardWalker) Walk(ctx context.Context, starting_board gameplay.Board) {
	board_cache := caching.CreatePointerCache[gameplay.Board](5000, func() gameplay.Board {
		return gameplay.CreateBoard(gameplay.BOARD_BLACK, starting_board.Height, starting_board.Width)
	})

	// TODO can add a local visited cache to speed up repeated finds
	// will make search much more efficient

	sbi, sb := board_cache.Get()
	starting_board.CloneInto(sb)

	stack := caching.CreateArrayStack[WalkerBoardWIndex](1830)
	stack.Push(WalkerBoardWIndex{
		Board: sb,
		Index: sbi,
	})

	bw.WalkPrestacked(ctx, &board_cache, &stack, starting_board.Height)
}

func (bw BoardWalker) WalkPrestacked(ctx context.Context, board_cache *caching.PointerCache[gameplay.Board], stack *caching.ArrayStack[WalkerBoardWIndex], bsize uint8) {
	// TODO can add a local visited cache to speed up repeated finds
	// will make search much more efficient

	var explored uint64 = 0
	SAVING_LOCK.RLock()
	saving := SAVING
	SAVING_LOCK.RUnlock()

	updater := time.NewTicker(bw.Update_interval)
	purger := time.NewTicker(bw.Purge_interval)
	update_buffer := caching.CreateArrayStack[uint128.Uint128](int(bw.Update_interval.Seconds() * 8000000))

	neighbor_stack := caching.CreateArrayStack[gameplay.Coord](100)

	local_cache := visiting.CreateSimpleVisitedCache()
	local_final_cache := visiting.CreateSimpleVisitedCache()
	var local_repeated uint64 = 0

	exit_on_save := false

	if stack.Len() > 0 {
		fmt.Printf("processor %d has started\n", bw.Identifier)

	SearchLoop:
		for stack.Len() > 0 {
			select {
			case <-updater.C:
				// putting these channels here reduces lock contention
				select {
				case <-purger.C:
					if !saving {
						fmt.Printf("purging local cache from walker %d\n", bw.Identifier)
						local_cache.Clear()
						local_final_cache.Clear()
					}
				case <-ctx.Done():
					exit_on_save = true
				case fp := <-bw.File_chan:
					err := bw.ToFile(fp, stack)
					if err != nil {
						fmt.Printf("failed to save walker %d: %s\n", bw.Identifier, err.Error())
						bw.Ready_chan <- false
						break SearchLoop
					}
					fmt.Printf("saved processor %d\n", bw.Identifier)
					bw.Ready_chan <- true
					if exit_on_save {
						break SearchLoop
					}
				default:
				}

				psaving := saving
				SAVING_LOCK.RLock()
				saving = SAVING
				SAVING_LOCK.RUnlock()
				if saving != psaving {
					SAVING_LOCK.Lock()
					PAUSED_COUNT++
					SAVING_LOCK.Unlock()
				}

				bw.Visited.Lock()
				for update_buffer.Len() > 0 {
					bh := update_buffer.Pop()
					if bw.Visited.TryInsert(bh) {
						// new board state was found
						*bw.Counter += 1
					} else {
						*bw.Repeated += 1
					}
				}
				*bw.Explored += explored
				*bw.Repeated += local_repeated
				bw.Visited.Unlock()
				explored = 0
				local_repeated = 0
				updater.Reset(bw.Update_interval)
			default:
				if saving {
					time.Sleep(100 * time.Millisecond)
					continue
				}
				sb := stack.Pop()

				findNextBoards(*sb.Board, &neighbor_stack)

				if neighbor_stack.Len() > 0 {
					// If move is legal, then append it to the search stack
					for neighbor_stack.Len() > 0 {
						mm := neighbor_stack.Pop()
						bci, bc := board_cache.Get()
						sb.Board.CloneInto(bc)
						bc.PlacePiece(mm.Row, mm.Column)
						bh := bc.Hash()
						if local_cache.TryInsert(bh) {
							stack.Push(WalkerBoardWIndex{
								Board: bc,
								Index: bci,
							})
						} else {
							board_cache.Free(bci)
						}
					}
				} else {
					// there are no legal moves, try the other player
					if sb.Board.Player == gameplay.BOARD_WHITE {
						sb.Board.Player = gameplay.BOARD_BLACK
					} else {
						sb.Board.Player = gameplay.BOARD_WHITE
					}

					findNextBoards(*sb.Board, &neighbor_stack)

					if neighbor_stack.Len() > 0 {
						for neighbor_stack.Len() > 0 {
							mm := neighbor_stack.Pop()
							bci, bc := board_cache.Get()
							sb.Board.CloneInto(bc)
							bc.PlacePiece(mm.Row, mm.Column)
							bh := bc.Hash()
							if local_cache.TryInsert(bh) {
								stack.Push(WalkerBoardWIndex{
									Board: bc,
									Index: bci,
								})
							} else {
								board_cache.Free(bci)
							}
						}
					} else {
						// there are no moves for anybody
						bh := sb.Board.Hash()

						// the local cache reduces the overall speed by 3mil/sec
						// but the repeated speed decreases by 19mil/update

						// because we check for visitation before we push the boards onto the stack
						// we can't use the same local cache here, otherwise boards that were neighbors
						// to previous boards would already be in the cache and thus, would not get counted.
						if local_final_cache.TryInsert(bh) {
							update_buffer.Push(bh)
						} else {
							local_repeated++
						}
					}
				}

				board_cache.Free(sb.Index)

				explored += 1
			}
		}
	}

	fmt.Printf("processor %d is exiting\n", bw.Identifier)

	// empty remaining boards
	bw.Visited.Lock()
	for update_buffer.Len() > 0 {
		bh := update_buffer.Pop()
		if bw.Visited.TryInsert(bh) {
			// new board state was found
			*bw.Counter += 1
		} else {
			*bw.Repeated += 1
		}
	}
	*bw.Explored += explored
	*bw.Repeated += local_repeated
	bw.Visited.Unlock()

	bw.Finished_lock.Lock()
	defer bw.Finished_lock.Unlock()
	*bw.Finished_count++
	fmt.Printf("processor %d has exited\n", bw.Identifier)
}

func (bw BoardWalker) ToFile(fp *os.File, stack *caching.ArrayStack[WalkerBoardWIndex]) error {
	barr := make([]byte, 16)
	for bi := 0; bi < stack.Len(); bi++ {
		b := stack.Index(bi)
		bh := b.Board.Hash()
		barr[0] = byte(bh.H >> 56)
		barr[1] = byte(bh.H >> 48)
		barr[2] = byte(bh.H >> 40)
		barr[3] = byte(bh.H >> 32)
		barr[4] = byte(bh.H >> 24)
		barr[5] = byte(bh.H >> 16)
		barr[6] = byte(bh.H >> 8)
		barr[7] = byte(bh.H)
		barr[8] = byte(bh.L >> 56)
		barr[9] = byte(bh.L >> 48)
		barr[10] = byte(bh.L >> 40)
		barr[11] = byte(bh.L >> 32)
		barr[12] = byte(bh.L >> 24)
		barr[13] = byte(bh.L >> 16)
		barr[14] = byte(bh.L >> 8)
		barr[15] = byte(bh.L)
		_, err := fp.Write(barr)
		if err != nil {
			return err
		}
	}
	return nil
}
