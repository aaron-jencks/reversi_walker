CC=gcc
PP=g++
CFLAGS=-Wall
LDFLAGS=-pthread
DEBUG=-g -Ddebug

C_SRC=gameplay/reversi.c mem_man/mmap_man.c mem_man/mempage.c hashing/hash_functions.c hashing/hashtable.c hashing/lookup3.c utils/path_util.c utils/csv.c utils/dictionary/dmempage.c utils/heapsort.c
CPP_SRC=mem_man/heir.cpp gameplay/walker.cpp utils/tarraylist.cpp utils/fileio.cpp utils/dictionary/fdict.cpp utils/dictionary/hdict.cpp
C_TEST_SRC=tests/capturecounts_test.c tests/board_placement_test.c tests/fileio_test.c tests/heapsort_test.c tests/mempage_test.c
CPP_TEST_SRC=tests/arraylist_test.cpp tests/dict_test.cpp tests/legal_moves_test.cpp tests/mmap_test.cpp

C_HEADERS=$(C_SRC:.c=.h)
CPP_HEADERS=$(CPP_SRC:.cpp=.hpp)
C_TEST_HEADERS=$(C_TEST_SRC:.c=.h)
CPP_TEST_HEADERS=$(CPP_TEST_SRC:.cpp=.hpp)

SRC=$(C_SRC) $(CPP_SRC)
HEADERS=$(C_HEADERS) $(CPP_HEADERS)
TEST_SRC=$(SRC) $(C_TEST_SRC) $(CPP_TEST_SRC)
TEST_HEADERS=$(HEADERS) $(C_TEST_HEADERS) $(CPP_TEST_HEADERS)

CHECKPOINT_PATH="./reversi_checkpoint.bin"
export CHECKPOINT_PATH

CC_COMPILE=$(CC) $(CFLAGS) -o $(1) -c $(2)
PP_COMPILE=$(CC) $(CFLAGS) -o $(1) -c $(2)

ifeq ($(MAKECMDGOALS),debug)
	CFLAGS+= $(DEBUG)
else
	CFLAGS+= -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-result -Wno-strict-aliasing -O3
endif

all: main;
debug: main;

main: main.cpp $(SRC) $(HEADERS)
	$(PP) $(CFLAGS) -o $@ $< $(SRC) $(LDFLAGS)

tester: tester.cpp $(SRC) $(HEADERS) $(TEST_SRC) $(TEST_HEADERS)
	$(PP) $(CFLAGS) -o $@ $< $(SRC) $(TEST_SRC) $(LDFLAGS)

.PHONY : clean
clean:
	rm main tester

.PHONY : tests
tests: tester
	./tester
	rm tester
