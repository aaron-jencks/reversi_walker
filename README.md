# Reversi Walker
C implementation of the game reversi, that counts how many game states there are in the game. It does this by performing a depth-first-search (DFS) on every possible move in the game, counting all of the unique endings that it finds.

# Installation
You can run this program by calling `make`. There are define commands that you can update by editting the line in [makefile](./makefile) marked `cflags`. Below is a list of defines that you can use to modify the program:

- **limitprocs**: Limits the program to a single thread.
- **debug**: Displays verbose data about walker processing.
- **filedebug**: Displays verbose data about checkpoint file IO.
- **checkpointdebug**: Displays verbose data about checkpoint processing.
- **swapdebug**: Displays verbose data about page swapping.
- **mempagedebug**: Displays verbose data about mempage management.
- **heirdebug**: Displays verbose data about heirarchy management.
- **hashdebug**: Displays verbose data about hashtable management.

To use any of these, simply add `-Dname` to `cflags`.

To turn on the debugging symbols, add `-g` to `cflags`

# Uninstallation
You can remove all compiled objects and the executable by calling `make clean`

# Documentation
There is a pdf document that contains a report I wrote on this topic in [report](./report).

# Player
You can play reversi/othello by using the python script I wrote inside of [othello_player](./othello_player).
