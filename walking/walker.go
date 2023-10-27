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

type walkerBoardWIndex struct {
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

	stack := caching.CreateArrayStack[walkerBoardWIndex](1830)
	stack.Push(walkerBoardWIndex{
		Board: sb,
		Index: sbi,
	})

	// TODO find a way to cache boards so that we don't have to reallocate the array every time we clone one

	var explored uint64 = 0
	SAVING_LOCK.RLock()
	saving := SAVING
	SAVING_LOCK.RUnlock()

	updater := time.NewTicker(bw.Update_interval)
	update_buffer := caching.CreateArrayStack[uint128.Uint128](int(bw.Update_interval.Seconds() * 8000000))

	neighbor_stack := caching.CreateArrayStack[gameplay.Coord](100)

	exit_on_save := false

	if stack.Len() > 0 {
		fmt.Printf("processor %d has started\n", bw.Identifier)

		// TODO reduce the number of channels to get rid of lock contention
		// or put channels into a different loop so that they aren't checked every iteration
	SearchLoop:
		for stack.Len() > 0 {
			select {
			case <-updater.C:
				// putting these channels here reduces lock contention
				select {
				case <-ctx.Done():
					exit_on_save = true
				case fp := <-bw.File_chan:
					err := bw.ToFile(fp, &stack)
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
				bw.Visited.Unlock()
				explored = 0
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
						stack.Push(walkerBoardWIndex{
							Board: bc,
							Index: bci,
						})
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
							stack.Push(walkerBoardWIndex{
								Board: bc,
								Index: bci,
							})
						}
					} else {
						// there are no moves for anybody
						bh := sb.Board.Hash()
						update_buffer.Push(bh)
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
			*bw.Counter += 1
		} else {
			*bw.Repeated += 1
		}
	}
	*bw.Explored += explored
	bw.Visited.Unlock()

	bw.Finished_lock.Lock()
	defer bw.Finished_lock.Unlock()
	*bw.Finished_count++
	fmt.Printf("processor %d has exited\n", bw.Identifier)
}

func (bw BoardWalker) ToFile(fp *os.File, stack *caching.ArrayStack[walkerBoardWIndex]) error {
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
