# Reversi Walker
This program traverses the game space of Othello/Reversi and determines an accurate count of the number of unique ways the game can end.

# Installation and Running
You'll need a golang installation on your system a minimum of 1.18. Then download the repo, cd into the base directory of the repo and run `go build .`. This will build the executable. Below is a description of command line arguments:

```
Usage of ./reversi:
  -check string
        indicates where to save checkpoints (default "./checkpoint.bin")
  -cpuprofile string
        specifies where to save pprof data to if supplied, leave empty to disable
  -display duration
        specifies how often to update the statistics in the terminal (default 2s)
  -memprofile string
        specifies where to save the pprof memory data to if supplied, leave empty to disable
  -procs uint
        specifies how many threads to use for processing (default cpu count)
  -restore string
        specifies where to restore the simulation from if supplied, leave empty to start fresh
  -save duration
        specifies how often to save the state of the walker (default 30m0s)
  -size uint
        specifies the size of the board to run the program on (default 8)
  -visitpurge duration
        specifies how often to purge the thread local visited cache for DFS (default 2m0s)
  -walkupdate duration
        specifies how often the walkers should take their cached data and sync it with the cache (default 1s)
```

**Note** about memory usage: With the implementation of the local thread cache this programs consumes **LOTS** of memory.
You'll want to tweak `-visitpurge` until you find a balance with the garbage collector. If this interval is too high, then it'll run out of memory.
Too low and the walking speed and efficiency will be hindered.

**Note** on durations and updates: To aleviate lock contention, this program uses the `-walkupdate` to limit how often channels are checked.
This slows down responsiveness but massively increases performance. This also means that all other durations should be in increments of `-walkupdate`
to maximize responsiveness.

## Utility Functions

You can find a couple of utility functions that can help with checkpoint files:

- [stats](./cmd/checkpoint-stats/) Parses and reports some useful information about the contents of a checkpoint file.
- [convert](./cmd/checkpoint-convert/) Parses and converts the hashes within a checkpoint file. Be warned, this takes a **LONG** time.

# Documentation
There is a pdf document that contains a report I wrote on this topic in [report](./report). This includes the final counts I've found for board sizes.

# Player
You can play reversi/othello by using the python script I wrote inside of [othello_player](./othello_player).
