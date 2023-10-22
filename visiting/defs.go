package visiting

import "github.com/aaron-jencks/reversi/utils/uint128"

// represents a cache for final board states
//
// implementers must be careful to ensure it is thread-safe
type VisitedCache interface {
	// TryInsert attempts to insert the given key
	//
	// returns if the key was successfully inserted
	TryInsert(uint128.Uint128) bool
}
