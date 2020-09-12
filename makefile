cc=gcc
cflags=-Ddebug -g
objects=reversi.o 

reversi.o: reversi.c reversi.h
	$(cc) $(cflags) -o $@ -c $<