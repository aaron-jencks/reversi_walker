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
