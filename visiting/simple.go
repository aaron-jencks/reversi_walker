package visiting

import (
	"sync"

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
