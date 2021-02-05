cc=gcc
# cflags=$(cflags)
objects=reversi.o ll.o walker.o arraylist.o hashtable.o lookup.o valid_moves.o mempage.o fileio.o mmap_man.o hash_functions.o heir.o path_util.o saving_algorithms.o heir_swapper.o
cuda_objects=
test_objects=capturecounts_test.o legal_moves_test.o board_placement_test.o mempage_test.o fileio_test.o mmap_test.o

CHECKPOINT_PATH="$(HOME)/reversi_checkpoint.bin"
export CHECKPOINT_PATH

all: main;

main: main.o $(objects)
	$(cc) $(cflags) -o $@ $< $(objects) -pthread

gmain: gmain.cu $(cuda_objects)
	nvcc -arch=sm_61 $(cflags) -o $@ $< $(cuda_objects) -lpthread

tester: tester.o $(objects) $(test_objects)
	$(cc) $(cflags) -o $@ $< $(objects) $(test_objects) -pthread

main.o: main.c $(objects)
	$(cc) $(cflags) -o $@ -c $<

tester.o: tester.c $(objects) $(test_objects)
	$(cc) -g $(cflags) -o $@ -c $<

reversi.o: ./gameplay/reversi.c ./gameplay/reversi.h
	$(cc) $(cflags) -o $@ -c $<

valid_moves.o: ./gameplay/valid_moves.c ./gameplay/valid_moves.h walker.o arraylist.o
	$(cc) $(cflags) -o $@ -c $<

cache.o: ./hashing/cache.c ./hashing/cache.h arraylist.o
	$(cc) $(cflags) -o $@ -c $<

lookup.o: ./hashing/lookup3.c ./hashing/lookup3.h
	$(cc) $(cflags) -o $@ -c $<

ll.o: ./utils/ll.c ./utils/ll.h
	$(cc) $(cflags) -o $@ -c $<

heir.o: ./mem_man/heir.c ./mem_man/heir.h hash_functions.o reversi.o mmap_man.o
	$(cc) $(cflags) -o $@ -c $<

heir_swapper.o: ./mem_man/heir_swapper.c ./mem_man/heir_swapper.h heir.o  mmap_man.o
	$(cc) $(cflags) -o $@ -c $<

hashtable.o: ./hashing/hashtable.c ./hashing/hashtable.h mempage.o arraylist.o
	$(cc) $(cflags) -o $@ -c $<

hash_functions.o: ./hashing/hash_functions.c ./hashing/hash_functions.h reversi.o lookup.o
	$(cc) $(cflags) -o $@ -c $<

arraylist.o: ./utils/arraylist.c ./utils/arraylist.h
	$(cc) $(cflags) -o $@ -c $<

mmap_man.o: ./mem_man/mmap_man.c ./mem_man/mmap_man.h path_util.o
	$(cc) $(cflags) -o $@ -c $<

mempage.o: ./mem_man/mempage.c ./mem_man/mempage.h path_util.o
	$(cc) $(cflags) -o $@ -c $<

walker.o: ./gameplay/walker.c ./gameplay/walker.h reversi.o ll.o arraylist.o
	$(cc) $(cflags) -o $@ -c $<

fileio.o: ./utils/fileio.c ./utils/fileio.h walker.o arraylist.o hashtable.o  path_util.o saving_algorithms.o
	$(cc) $(cflags) -o $@ -c $<

path_util.o: ./utils/path_util.c ./utils/path_util.h
	$(cc) $(cflags) -o $@ -c $<

saving_algorithms.o: ./utils/saving_algorithms.c ./utils/saving_algorithms.h
	$(cc) $(cflags) -o $@ -c $<

# Tests

capturecounts_test.o: tests/capturecounts_test.c tests/capturecounts_test.h reversi.o
	$(cc) $(cflags) -o $@ -c $<

legal_moves_test.o: tests/legal_moves_test.c tests/legal_moves_test.h reversi.o walker.o
	$(cc) $(cflags) -o $@ -c $<

board_placement_test.o: tests/board_placement.c tests/board_placement.h reversi.o
	$(cc) $(cflags) -o $@ -c $<

mempage_test.o: tests/mempage_test.c tests/mempage_test.h mempage.o
	$(cc) $(cflags) -o $@ -c $<

fileio_test.o: tests/fileio_test.c tests/fileio_test.h hashtable.o fileio.o
	$(cc) $(cflags) -o $@ -c $<

mmap_test.o: tests/mmap_test.c tests/mmap_test.h heir.o reversi.o hash_functions.o 
	$(cc) $(cflags) -o $@ -c $<

.PHONY : clean
clean:
	rm main main.o $(objects) gmain $(cuda_objects)

.PHONY : tests
tests: tester $(objects) $(test_objects)
	./tester
	rm tester tester.o $(objects) $(test_objects)
