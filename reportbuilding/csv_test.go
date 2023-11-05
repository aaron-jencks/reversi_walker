package reportbuilding

import (
	"fmt"
	"testing"

	"github.com/aaron-jencks/reversi/gameplay"
	"github.com/aaron-jencks/reversi/visiting"
	"github.com/stretchr/testify/assert"
)

type testWriter struct {
	Arr []byte
}

func (tw *testWriter) Write(p []byte) (n int, err error) {
	tw.Arr = append(tw.Arr, p...)
	return len(p), nil
}

func TestCSVOutput(t *testing.T) {
	b := gameplay.CreateBoard(gameplay.BOARD_BLACK, 4, 4)
	values := []uint8{0, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 0, 0, 1, 2}

	for r := uint8(0); r < 4; r++ {
		for c := uint8(0); c < 4; c++ {
			b.Put(r, c, gameplay.BoardValue(values[4*r+c]))
		}
	}

	output := "player,a0,b0,c0,d0,a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3\n\"black\""
	for _, v := range values {
		output += fmt.Sprintf(",%d", v)
	}
	output += "\n"

	cache := visiting.CreateSimpleVisitedCache()
	bh := b.Hash()
	nb := gameplay.CreateUnhashBoard(4, bh)
	assert.Equal(t, b, nb, "expected hashed and unhashed boards to be equal")
	assert.True(t, cache.TryInsert(b.Hash()), "failed to insert board into the test cache")

	writer := testWriter{}
	err := BuildCSVReport(&writer, cache, 4)
	assert.NoError(t, err, "unexpected writing error occured")

	assert.Equal(t, output, string(writer.Arr), "expected csv output to be equal")
}
