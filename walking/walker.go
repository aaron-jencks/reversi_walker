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

func (bw BoardWalker) Walk(ctx context.Context, starting_board gameplay.Board) {
	stack := caching.CreateArrayStack[gameplay.Board](1830)
	stack.Push(starting_board)

	// TODO find a way to cache boards so that we don't have to reallocate the array every time we clone one

	var explored uint64 = 0

	updater := time.NewTicker(bw.Update_interval)
	update_buffer := caching.CreateArrayStack[uint128.Uint128](int(bw.Update_interval.Seconds() * 1400000))

	if stack.Len() > 0 {
		fmt.Printf("processor %d has started\n", bw.Identifier)

	SearchLoop:
		for stack.Len() > 0 {
			select {
			case <-ctx.Done():
				break SearchLoop
			case fp := <-bw.File_chan:
				err := bw.ToFile(fp, &stack)
				if err != nil {
					fmt.Printf("failed to save walker %d: %s\n", bw.Identifier, err.Error())
					bw.Ready_chan <- false
					break SearchLoop
				}
				fmt.Printf("saved processor %d\n", bw.Identifier)
				bw.Ready_chan <- true
			case <-updater.C:
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
				sb := stack.Pop()

				next_coords := findNextBoards(sb)

				if len(next_coords) > 0 {
					// If move is legal, then append it to the search stack
					for _, mm := range next_coords {
						bc := sb.Clone()
						bc.PlacePiece(mm.Row, mm.Column)
						stack.Push(bc)
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
							stack.Push(bc)
						}
					} else {
						// there are no moves for anybody
						bh := sb.Hash()
						update_buffer.Push(bh)
					}
				}

				explored += 1
			}
		}
	}

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
	explored = 0

	fmt.Printf("processor %d has exited\n", bw.Identifier)
	bw.Finished_lock.Lock()
	defer bw.Finished_lock.Unlock()
	*bw.Finished_count++
}

func (bw BoardWalker) ToFile(fp *os.File, stack *caching.ArrayStack[gameplay.Board]) error {
	barr := make([]byte, 16)
	for bi := 0; bi < stack.Len(); bi++ {
		b := stack.Index(bi)
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
		_, err := fp.Write(barr)
		if err != nil {
			return err
		}
	}
	return nil
}
