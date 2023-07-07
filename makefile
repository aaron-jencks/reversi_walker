CC=gcc
PP=g++
CFLAGS=-Wall
LDFLAGS=-pthread
DEBUG=-g -Ddebug
OBJS=gameplay/reversi.o mem_man/mmap_man.o hashing/hash_functions.o utils/path_util.o utils/heapsort.o utils/csv.o utils/dictionary/dmempage.o 
CPP_OBJS=utils/dictionary/fdict.opp utils/dictionary/hdict.o mem_man/heir.opp utils/fileio.opp gameplay/walker.opp utils/semaphore.opp utils/tarraylist.opp
TEST_OBJS=tests/capturecounts_test.o tests/legal_moves_test.opp tests/board_placement_test.o tests/mempage_test.o tests/mmap_test.opp tests/dict_test.opp tests/heapsort_test.o tests/arraylist_test.opp

CHECKPOINT_PATH="./reversi_checkpoint.bin"
export CHECKPOINT_PATH

CC_COMPILE=$(CC) $(CFLAGS) -o $(1) -c $(2)
PP_COMPILE=$(CC) $(CFLAGS) -o $(1) -c $(2)

ifeq ($(MAKECMDGOALS),debug)
	CFLAGS+= $(DEBUG)
else
	CFLAGS+= -O3
endif

all: main;
debug: main;

main: main.o $(OBJS) $(CPP_OBJS)
	$(PP) $(CFLAGS) -o $@ $< $(OBJS) $(CPP_OBJS) $(LDFLAGS)

tester: tester.o $(OBJS) $(CPP_OBJS) $(TEST_OBJS)
	$(PP) $(CFLAGS) -o $@ $< $(OBJS) $(CPP_OBJS) $(TEST_OBJS) $(LDFLAGS)

experimenter: experimenter.cpp utils/tarraylist.hpp
	$(PP) $(CFLAGS) -o $@ $< $(LDFLAGS)

main.o: main.cpp $(OBJS) $(CPP_OBJS)
	$(PP) $(CFLAGS) -o $@ -c $<

tester.o: tester.cpp $(OBJS) $(CPP_OBJS) $(TEST_OBJS)
	$(PP) -g $(CFLAGS) -o $@ -c $<

gameplay/valid_moves.o: gameplay/walker.h utils/tarraylist.hpp
hashing/cache.o: utils/tarraylist.hpp
mem_man/heir_swapper.o: mem_man/mmap_man.h utils/heapsort.h hashing/hashtable.h
hashing/hashtable.o: mem_man/mempage.h utils/tarraylist.hpp
hashing/hash_functions.o: gameplay/reversi.h hashing/lookup3.h
mem_man/mmap_man.o: utils/path_util.h
mem_man/mempage.o: utils/path_util.h
utils/dictionary/dmempage.o: utils/dictionary/dict_def.h
tests/capturecounts_test.o: gameplay/reversi.h
tests/board_placement_test.o: gameplay/reversi.h
tests/mempage_test.o: mem_man/mempage.h
tests/fileio_test.o: utils/fileio.h
tests/heapsort_test.o: utils/heapsort.h
%.o: %.c %.h
	$(call CC_COMPILE,$@,$<)

mem_man/heir.o: hashing/hash_functions.h gameplay/reversi.h mem_man/mmap_man.h utils/dictionary/hdict.hpp utils/dictionary/fdict.hpp utils/semaphore.hpp
gameplay/walker.o: gameplay/reversi.h utils/tarraylist.hpp mem_man/heir.hpp
utils/fileio.o: gameplay/walker.h utils/tarraylist.hpp utils/path_util.h mem_man/heir.hpp
utils/dictionary/fdict.opp: ./utils/dictionary/dict_def.h utils/heapsort.h utils/tarraylist.hpp
utils/dictionary/hdict.opp: utils/dictionary/dmempage.h
tests/legal_moves_test.opp: gameplay/reversi.h gameplay/walker.h utils/tarraylist.hpp
tests/mmap_test.opp: mem_man/heir.hpp gameplay/reversi.h hashing/hash_functions.h
tests/dict_test.opp: utils/dictionary/dict_def.h utils/dictionary/fdict.hpp utils/dictionary/hdict.hpp
tests/arraylist_test.opp: utils/tarraylist.hpp
%.opp: %.cpp %.hpp
	$(call PP_COMPILE,$@,$<)

.PHONY : clean
clean:
	rm main main.o $(OBJS) gmain $(cuda_objects) tester tester.o $(TEST_OBJS) $(CPP_OBJS) experimenter

.PHONY : tests
tests: tester $(OBJS) $(CPP_OBJS) $(TEST_OBJS)
	./tester
	rm tester tester.o $(OBJS) $(TEST_OBJS)
