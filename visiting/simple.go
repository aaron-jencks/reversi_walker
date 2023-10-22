package visiting

import (
	"sync"

	"github.com/aaron-jencks/reversi/utils/uint128"
)

type SimpleVisitedCache struct {
	m map[uint64]map[uint64]bool
	l sync.RWMutex
}

func CreateSimpleVisitedCache() *SimpleVisitedCache {
	return &SimpleVisitedCache{
		m: make(map[uint64]map[uint64]bool),
	}
}

func (sc *SimpleVisitedCache) TryInsert(k uint128.Uint128) bool {
	sc.l.RLock()
	im, ok := sc.m[k.H]
	if !ok {
		sc.l.RUnlock()
		sc.l.Lock()
		sc.m[k.H] = map[uint64]bool{
			k.L: true,
		}
		sc.l.Unlock()
		return true
	}
	_, ok = im[k.L]
	sc.l.RUnlock()
	if !ok {
		sc.l.Lock()
		im[k.L] = true
		sc.l.Unlock()
		return true
	}
	return false
}
