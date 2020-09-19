cc=gcc
cflags=-Ddebug -g
objects=reversi.o cache.o ll.o walker.o arraylist.o hashtable.o lookup.o

all: main;

main: main.o $(objects)
	$(cc) $(cflags) -o $@ $< $(objects)

main.o: main.c $(objects)
	$(cc) $(cflags) -o $@ -c $<

reversi.o: reversi.c reversi.h
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

walker.o: walker.c walker.h reversi.o ll.o
	$(cc) $(cflags) -o $@ -c $<

.PHONY : clean
clean:
	rm $(objects)