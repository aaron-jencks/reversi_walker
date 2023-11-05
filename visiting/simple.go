package visiting

import (
	"fmt"
	"os"
	"sync"

	"github.com/aaron-jencks/reversi/utils"
	"github.com/aaron-jencks/reversi/utils/uint128"
)

type SimpleVisitedCache struct {
	sync.RWMutex
	m map[uint64]map[uint64]bool
}

func CreateSimpleVisitedCache() *SimpleVisitedCache {
	return &SimpleVisitedCache{
		m: make(map[uint64]map[uint64]bool),
	}
}

func (sc *SimpleVisitedCache) TryInsert(k uint128.Uint128) bool {
	im, ok := sc.m[k.H]
	if !ok {
		sc.m[k.H] = map[uint64]bool{
			k.L: true,
		}
		return true
	}
	_, ok = im[k.L]
	if !ok {
		im[k.L] = true
		return true
	}
	return false
}

// Len returns the total number of boards stored in the map
func (sc *SimpleVisitedCache) Len() int {
	result := 0
	for _, v := range sc.m {
		result += len(v)
	}
	return result
}

func (sc *SimpleVisitedCache) Clear() {
	sc.m = map[uint64]map[uint64]bool{}
}

func (sc *SimpleVisitedCache) ToFile(fp *os.File) error {
	mlen := sc.Len()

	fmt.Printf("Saving cache with %d entries\n", mlen)

	// write the length
	_, err := fp.Write(utils.Uint64ToBytes(uint64(mlen)))
	if err != nil {
		return err
	}

	// write the boards
	count := 0
	for h, ls := range sc.m {
		for l := range ls {
			_, err = fp.Write(utils.Uint128ToBytes(uint128.Uint128{
				H: h,
				L: l,
			}))
			if err != nil {
				return err
			}
			count++
			fmt.Printf("\rSaving %d/%d", count, mlen)
		}
	}

	fmt.Println("\nFinished saving cache")

	return nil
}

func (sc *SimpleVisitedCache) FromFile(fp *os.File) error {
	fmt.Println("Loading cache from file...")

	// reusable uint64 buffer
	i64buff := make([]byte, 8)

	_, err := fp.Read(i64buff)
	if err != nil {
		return err
	}

	count := utils.Uint64FromBytes(i64buff)

	sc.Clear()

	// reusable uint128 buff
	i128buff := make([]byte, 16)
	for bi := uint64(0); bi < count; bi++ {
		_, err := fp.Read(i128buff)
		if err != nil {
			return err
		}
		sc.TryInsert(utils.Uint128FromBytes(i128buff))
		fmt.Printf("\rLoaded %d/%d", bi+1, count)
	}

	fmt.Println("\nFinished loading cache")

	return nil
}

func (sc *SimpleVisitedCache) Keys() []uint128.Uint128 {
	result := make([]uint128.Uint128, sc.Len())
	ri := 0
	for h, hm := range sc.m {
		for l := range hm {
			result[ri].H = h
			result[ri].L = l
			ri++
		}
	}
	return result
}
