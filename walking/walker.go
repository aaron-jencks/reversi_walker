package walking

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/aaron-jencks/reversi/gameplay"
	"github.com/aaron-jencks/reversi/visiting"
)

// represents a processor that walks the game tree
type BoardWalker struct {
	Identifier     uint32
	Visited        visiting.VisitedCache
	Counter        *uint64
	Counter_lock   *sync.RWMutex
	Explored       *uint64
	Explored_lock  *sync.RWMutex
	Repeated       *uint64
	Repeated_lock  *sync.RWMutex
	Saving_counter *uint64
	File_pointer   **os.File
	File_lock      *sync.Mutex
	Finished_count *uint64
	Finished_lock  *sync.RWMutex
}

// when set, causes the processors to save their work
var SAVING_FLAG bool
var saving_lock sync.Mutex

func (bw BoardWalker) Walk(ctx context.Context, save_chan chan bool, starting_board gameplay.Board) {
	// TODO make it so that we don't update the counters every board, but on an interval of boards
	// this will reduce lock contention
	stack := make([]gameplay.Board, 1, 1000)
	stack[0] = starting_board

	if len(stack) > 0 {
		fmt.Printf("processor %d has started\n", bw.Identifier)

	SearchLoop:
		for len(stack) > 0 {
			select {
			case <-ctx.Done():
				break SearchLoop
			case <-save_chan:
				saving_lock.Lock()

				saving_lock.Unlock()
			default:
				sb := stack[len(stack)-1]
				stack = stack[:len(stack)-1]

				next_coords := findNextBoards(sb)

				if len(next_coords) > 0 {
					// If move is legal, then append it to the search stack
					for _, mm := range next_coords {
						bc := sb.Clone()
						bc.PlacePiece(mm.Row, mm.Column)
						stack = append(stack, bc)
					}
				} else {
					// there are no legal moves, try the other player
					if sb.Player == gameplay.BOARD_WHITE {
						sb.Player = gameplay.BOARD_BLACK
					} else {
						sb.Player = gameplay.BOARD_WHITE
					}

					next_coords = findNextBoards(sb)

					if len(next_coords) > 0 {
						for _, mm := range next_coords {
							bc := sb.Clone()
							bc.PlacePiece(mm.Row, mm.Column)
							stack = append(stack, bc)
						}
					} else {
						// there are no moves for anybody
						bh := sb.Hash()
						if bw.Visited.TryInsert(bh) {
							// new board state was found
							bw.Counter_lock.Lock()
							*bw.Counter += 1
							bw.Counter_lock.Unlock()
						} else {
							bw.Repeated_lock.Lock()
							*bw.Repeated += 1
							bw.Repeated_lock.Unlock()
						}
					}
				}

				bw.Explored_lock.Lock()
				*bw.Explored += 1
				bw.Explored_lock.Unlock()
			}
		}
	}

	fmt.Printf("processor %d has exited\n", bw.Identifier)
	bw.Finished_lock.Lock()
	defer bw.Finished_lock.Unlock()
	*bw.Finished_count++
}

func (bw BoardWalker) ToFile(stack []gameplay.Board) error {
	saving_lock.Lock()
	defer saving_lock.Unlock()
	barr := make([]byte, 16)
	for _, b := range stack {
		bh := b.Hash()
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
		_, err := (*bw.File_pointer).Write(barr)
		if err != nil {
			return err
		}
	}
	return nil
}
