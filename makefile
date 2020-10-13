cc=gcc
cflags= # -Ddebug -g
objects=reversi.o ll.o walker.o arraylist.o hashtable.o lookup.o valid_moves.o
test_objects=capturecounts_test.o legal_moves_test.o board_placement_test.o

all: main;

main: main.o $(objects)
	$(cc) $(cflags) -o $@ $< $(objects)

tester: tester.o $(objects) $(test_objects)
	$(cc) $(cflags) -o $@ $< $(objects) $(test_objects)

main.o: main.c $(objects)
	$(cc) $(cflags) -o $@ -c $<

tester.o: tester.c $(objects) $(test_objects)
	$(cc) $(cflags) -o $@ -c $<

reversi.o: reversi.c reversi.h
	$(cc) $(cflags) -o $@ -c $<

valid_moves.o: valid_moves.c valid_moves.h walker.o arraylist.o
	$(cc) $(cflags) -o $@ -c $<

cache.o: cache.c cache.h arraylist.o
	$(cc) $(cflags) -o $@ -c $<

lookup.o: lookup3.c lookup3.h
	$(cc) $(cflags) -o $@ -c $<

ll.o: ll.c ll.h
	$(cc) $(cflags) -o $@ -c $<

hashtable.o: hashtable.c hashtable.h ll.o
	$(cc) $(cflags) -o $@ -c $<

arraylist.o: arraylist.c arraylist.h
	 $(cc) $(cflags) -o $@ -c $<

walker.o: walker.c walker.h reversi.o ll.o arraylist.o
	$(cc) $(cflags) -o $@ -c $<

# Tests

capturecounts_test.o: tests/capturecounts_test.c tests/capturecounts_test.h reversi.o
	$(cc) $(cflags) -o $@ -c $<

legal_moves_test.o: tests/legal_moves_test.c tests/legal_moves_test.h reversi.o walker.o
	$(cc) $(cflags) -o $@ -c $<

board_placement_test.o: tests/board_placement.c tests/board_placement.h reversi.o
	$(cc) $(cflags) -o $@ -c $<

.PHONY : clean
clean:
	rm main main.o $(objects)

.PHONY : tests
tests: tester $(objects) $(test_objects)
	./tester
	rm tester tester.o $(objects) $(test_objects)