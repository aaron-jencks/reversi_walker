cc=gcc
pp=g++
cflags+=-O3 -Wall
objects=reversi.o mmap_man.o hash_functions.o path_util.o heapsort.o csv.o dmempage.o 
cpp_objects=fdict.o hdict.o heir.o fileio.o walker.o semaphore.o tarraylist.o 
cuda_objects=
test_objects=capturecounts_test.o legal_moves_test.o board_placement_test.o mempage_test.o mmap_test.o dict_test.o heapsort_test.o arraylist_test.o 

CHECKPOINT_PATH="$(HOME)/reversi_checkpoint.bin"
export CHECKPOINT_PATH

all: main;

main: main.o $(objects) $(cpp_objects)
	$(pp) $(cflags) -o $@ $< $(objects) $(cpp_objects) -pthread

gmain: gmain.cu $(cuda_objects)
	nvcc -arch=sm_61 $(cppflags) -o $@ $< $(cuda_objects) -lpthread

tester: tester.o $(objects) $(cpp_objects) $(test_objects)
	$(pp) $(cflags) -o $@ $< $(objects) $(cpp_objects) $(test_objects) -pthread

experimenter: experimenter.cpp
	$(pp) $(cflags) -o $@ $< -pthread

main.o: main.cpp $(objects) $(cpp_objects)
	$(pp) $(cflags) -o $@ -c $<

tester.o: tester.cpp $(objects) $(test_objects)
	$(pp) -g $(cflags) -o $@ -c $<

reversi.o: ./gameplay/reversi.c ./gameplay/reversi.h
	$(cc) $(cflags) -o $@ -c $<

valid_moves.o: ./gameplay/valid_moves.c ./gameplay/valid_moves.h walker.o tarraylist.o 
	$(cc) $(cflags) -o $@ -c $<

cache.o: ./hashing/cache.c ./hashing/cache.h tarraylist.o
	$(cc) $(cflags) -o $@ -c $<

lookup.o: ./hashing/lookup3.c ./hashing/lookup3.h
	$(cc) $(cflags) -o $@ -c $<

ll.o: ./utils/ll.c ./utils/ll.h
	$(cc) $(cflags) -o $@ -c $<

heir.o: ./mem_man/heir.cpp ./mem_man/heir.hpp hash_functions.o reversi.o mmap_man.o hdict.o fdict.o semaphore.o 
	$(pp) $(cflags) -o $@ -c $<

heir_swapper.o: ./mem_man/heir_swapper.c ./mem_man/heir_swapper.h mmap_man.o heapsort.o hashtable.o 
	$(cc) $(cflags) -o $@ -c $<

hashtable.o: ./hashing/hashtable.c ./hashing/hashtable.h mempage.o utils/tarraylist.hpp
	$(cc) $(cflags) -o $@ -c $<

hash_functions.o: ./hashing/hash_functions.c ./hashing/hash_functions.h reversi.o lookup.o
	$(cc) $(cflags) -o $@ -c $<

mmap_man.o: ./mem_man/mmap_man.c ./mem_man/mmap_man.h path_util.o
	$(cc) $(cflags) -o $@ -c $<

mempage.o: ./mem_man/mempage.c ./mem_man/mempage.h path_util.o
	$(cc) $(cflags) -o $@ -c $<

walker.o: ./gameplay/walker.cpp ./gameplay/walker.hpp reversi.o tarraylist.o heir.o 
	$(pp) $(cflags) -o $@ -c $<

fileio.o: ./utils/fileio.cpp ./utils/fileio.hpp walker.o tarraylist.o path_util.o heir.o 
	$(pp) $(cflags) -o $@ -c $<

path_util.o: ./utils/path_util.c ./utils/path_util.h
	$(cc) $(cflags) -o $@ -c $<

heapsort.o: ./utils/heapsort.c ./utils/heapsort.h 
	$(cc) $(cflags) -o $@ -c $<

csv.o: ./utils/csv.c ./utils/csv.h 
	$(cc) $(cflags) -o $@ -c $<

dmempage.o: ./utils/dictionary/dmempage.c ./utils/dictionary/dmempage.h ./utils/dictionary/dict_def.h 
	$(cc) $(cflags) -o $@ -c $<

fdict.o: ./utils/dictionary/fdict.cpp ./utils/dictionary/fdict.hpp ./utils/dictionary/dict_def.h heapsort.o tarraylist.o 
	$(pp) $(cflags) -o $@ -c $<

hdict.o: ./utils/dictionary/hdict.cpp ./utils/dictionary/hdict.hpp dmempage.o 
	$(pp) $(cflags) -o $@ -c $<

tarraylist.o: ./utils/tarraylist.cpp ./utils/tarraylist.hpp;
	$(pp) $(cflags) -o $@ -c $<

semaphore.o: ./utils/semaphore.cpp ./utils/semaphore.hpp
	$(pp) $(cflags) -o $@ -c $<

# Tests

capturecounts_test.o: tests/capturecounts_test.c tests/capturecounts_test.h reversi.o
	$(cc) $(cflags) -o $@ -c $<

legal_moves_test.o: tests/legal_moves_test.cpp tests/legal_moves_test.hpp reversi.o walker.o tarraylist.o 
	$(pp) $(cflags) -o $@ -c $<

board_placement_test.o: tests/board_placement.c tests/board_placement.h reversi.o
	$(cc) $(cflags) -o $@ -c $<

mempage_test.o: tests/mempage_test.c tests/mempage_test.h mempage.o
	$(cc) $(cflags) -o $@ -c $<

fileio_test.o: tests/fileio_test.c tests/fileio_test.h fileio.o
	$(cc) $(cflags) -o $@ -c $<

mmap_test.o: tests/mmap_test.cpp tests/mmap_test.hpp heir.o reversi.o hash_functions.o 
	$(pp) $(cflags) -o $@ -c $<

dict_test.o: tests/dict_test.cpp tests/dict_test.hpp utils/dictionary/dict_def.h fdict.o hdict.o
	$(pp) $(cflags) -o $@ -c $<

heapsort_test.o: tests/heapsort_test.c tests/heapsort_test.h heapsort.o
	$(cc) $(cflags) -o $@ -c $<

arraylist_test.o: tests/arraylist_test.cpp tests/arraylist_test.hpp tarraylist.o 
	$(pp) $(cflags) -o $@ -c $<

.PHONY : clean
clean:
	rm main main.o $(objects) gmain $(cuda_objects) tester tester.o $(test_objects) $(cpp_objects) experimenter

.PHONY : tests
tests: tester $(objects) $(cpp_objects) $(test_objects)
	./tester
	rm tester tester.o $(objects) $(test_objects)
