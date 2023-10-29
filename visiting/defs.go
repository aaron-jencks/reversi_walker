package visiting

import (
	"os"

	"github.com/aaron-jencks/reversi/utils/uint128"
)

// represents a cache for final board states
//
// implementers must be careful to ensure it is thread-safe
type VisitedCache interface {
	// TryInsert attempts to insert the given key
	//
	// returns if the key was successfully inserted
	TryInsert(uint128.Uint128) bool

	// Lock locks the cache for the current user
	//
	// asynchronous behavior is undefined if TryInsert is called
	// before this is called
	Lock()
	Unlock()

	// ToFile stores the cache to the given file pointer
	ToFile(*os.File) error

	// FromFile Overwrites the current struct with data from the given file
	FromFile(*os.File) error
}
