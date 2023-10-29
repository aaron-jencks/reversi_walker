package visiting

import (
	"math/rand"
	"testing"

	"github.com/aaron-jencks/reversi/utils/uint128"
	"github.com/stretchr/testify/assert"
)

func TestSimple(t *testing.T) {
	c := CreateSimpleVisitedCache()

	genRandKey := func() uint128.Uint128 {
		return uint128.Uint128{
			H: rand.Uint64(),
			L: rand.Uint64(),
		}
	}

	nk := genRandKey()
	snk := genRandKey()
	for nk.Equal(snk) {
		snk = genRandKey()
	}

	assert.True(t, c.TryInsert(nk), "expected first insert to be successful")
	assert.False(t, c.TryInsert(nk), "expected second insert to be unsuccessful")
	assert.True(t, c.TryInsert(snk), "expected first insert to be successful")
	assert.False(t, c.TryInsert(snk), "expected second insert to be unsuccessful")
}

func TestMultithreadedSimple(t *testing.T) {
	c := CreateSimpleVisitedCache()

	genRandKey := func() uint128.Uint128 {
		return uint128.Uint128{
			H: rand.Uint64(),
			L: rand.Uint64(),
		}
	}

	// generate unique keys
	keys := make([]uint128.Uint128, 100000)
	for ki := range keys {
		var rk uint128.Uint128
		found := true
		for found {
			rk = genRandKey()
			found = false
			for _, pk := range keys[:ki] {
				if rk.Equal(pk) {
					found = true
					break
				}
			}
		}
		keys[ki] = rk
	}

	startproc := make(chan bool)
	procs := make([]chan bool, 10)
	count := 0
	for pi := range procs {
		procs[pi] = make(chan bool)
		go func(out chan bool) {
			<-startproc
			for _, k := range keys {
				c.Lock()
				if c.TryInsert(k) {
					count++
				}
				c.Unlock()
			}
			out <- true
		}(procs[pi])
	}

	close(startproc) // start the processors
	for _, p := range procs {
		<-p
	}

	assert.Equal(t, 100000, count, "expected output count to be equal to the number of insertions")
	assert.Equal(t, 100000, c.Len(), "expected size of the cache to be equal to the number of unique insertions")
}
