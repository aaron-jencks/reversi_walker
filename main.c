#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <err.h>
#include <pthread.h>
#include <sys/sysinfo.h>

#include "reversi.h"
#include "hashtable.h"
#include "lookup3.h"
#include "walker.h"
#include "ll.h"
#include "arraylist.h"
#include "valid_moves.h"

// TODO you can use the previous two board states to predict the next set of valid moves.

/**
 * I had to get rid of the stacks because I had no way to keep track of when a move generates new moves for BOTH colors .W.
 * or when a capture generates new moves
 * 
 * ie.
 * 
 * 00000000
 * 00000000
 * 00012000
 * 00021000
 * 00000000
 * 00000000
 * 
 * --->
 * 
 * 00000000
 * 00001000
 * 00011000
 * 00021000
 * 00x00000 <-- Where x is a new move generated for white by a captured piece -___-
 * 00000000
 * 
 * I need to optimize the linkedlists in the hashtable, and optimize the memory usage of the DFS stack by getting rid of the pointers.
 * 
 * Re work the hashtable so that it doesn't error out when the size gets too big.
 *  - TODO Add mempage system to speed up search in bins
 *  - TODO integrate mempage into hashtable.c
 * 
 * [DONE] make hashtable so that it uses arraylists for the bins, until they get too big, then switch the bins over to arraylists
 * 
 * Use pthread library to multithread project
 * mutex locks: pthread_mutex_t
 * use about 2 processes per core
 * Use pthread_yield instead of sleeping
 * 
 * put locks on the bins in the hashtable
 * 
 * do bfs to find the number of moves necessary to fill all the workers, then switch to dfs with n workers for each entry in the search queue
 * 
 * use 'ulimit -c unlimited' to make core dump
 * 
 */


void display_board(board b) {
    if(b) {
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c++) {
                printf("%c", board_get(b, r, c) + '0');
            }
            printf("\n");
        }
        printf("\n");
    }
}


void display_capture_counts(uint64_t cc) {
    /*
    * 0: upper-left
    * 1: up
    * 2: upper-right
    * 3: left
    * 4: right
    * 5: lower-left
    * 6: lower
    * 7: lower-right
    */
    printf("Capture Counts:\n");
    uint8_t c;
    for(uint8_t i = 0; i < 8; i++) {
        c = capture_count_get_count(cc, i);
        switch(i) {
            case 0:
                printf("\tNorthwest: ");
                break;
            case 1:
                printf("\tNorth: ");
                break;
            case 2:
                printf("\tNortheast: ");
                break;
            case 3:
                printf("\tWest: ");
                break;
            case 4:
                printf("\tEast: ");
                break;
            case 5:
                printf("\tSouthwest: ");
                break;
            case 6:
                printf("\tSouth: ");
                break;
            case 7:
                printf("\tSoutheast: ");
                break;
        }
        printf("%u\n", c);
    }
    printf("\n");
}


__uint128_t board_hash(void* brd) {
    if(brd) {
        board b = (board)brd;

        __uint128_t result = 0;

        result += b->player;
        result = result << 2;

        // You can fit 2 spaces in 3 bits if you really try,
        // so on an 8x8 board, 
        // we end up only using 96 bits instead of the entire 128.
        // well, 98 since we are including the player now
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c += 2) {
                uint8_t s1 = board_get(b, r, c), 
                        s2 = board_get(b, r, c + 1);

                result += (!s1 && !s2) ? 4 : 
                          (s1 == 1 && s2 == 1) ? 3 : 
                          (s1 == 2 && s2 == 2) ? 0 : 
                          (s1 == 2) ? 1 : 2;

                result = result << 3;
            }
        }

        // we still need to use the entire 128 bits though,
        // because the hashing algorithm works best in powers of 2
        uint32_t upperupper = 0, upperlower = 0, lowerupper = 0, lowerlower = 0;
        hashlittle2(&result, 8, &upperupper, &upperlower);
        hashlittle2(((char*)&result) + 8, 8, &lowerupper, &lowerlower);
        result = 0;
        result += upperupper;
        result = result << 32;
        result += upperlower;
        result = result << 32;
        result += lowerupper;
        result = result << 32;
        result += lowerlower;

        return result;
    }
    return 0;
}


// TODO make work on a 6x6
// Modify to copy boards, instead of 
int main() {
    hashtable cache = create_hashtable(1000000, &board_hash);
    uint64_t count = 0;
    pthread_mutex_t counter_lock;

    if(pthread_mutex_init(&counter_lock, 0)) err(4, "Initialization of counter mutex failed\n");

    printf("Starting walk...\n");

    uint32_t procs = get_nprocs();
    printf("Running on %d processors, using %d threads\n", procs, procs << 1);
    procs = procs << 1;

    // Perform BFS to get the desired number of initial states
    board b = create_board(1, 6, 6);
    
    // Setup the queue
    ptr_arraylist search_queue = create_ptr_arraylist(procs + 1);

    // Account for reflections and symmetry by using 1 of the 4 possible starting moves
    coord* next_moves = find_next_boards(b);

    for(char im = 0; next_moves[im]; im++) {
        coord m = next_moves[im];
        uint16_t sm = coord_to_short(m);
        board cb = clone_board(b);
        board_place_piece(cb, m->row, m->column);
        append_pal(search_queue, cb);
        free(m);
        break;
    }

    for(char im = 1; next_moves[im]; im++) {
        coord m = next_moves[im];
        free(m);
    }

    free(next_moves);
    destroy_board(b);

    // Perform the BFS
    while(search_queue->pointer < procs) {
        b = pop_front_pal(search_queue);
        next_moves = find_next_boards(b);

        for(char im = 0; next_moves[im]; im++) {
            coord m = next_moves[im];
            uint16_t sm = coord_to_short(m);
            board cb = clone_board(b);
            board_place_piece(cb, m->row, m->column);
            append_pal(search_queue, cb);
            free(m);
        }

        free(next_moves);
        destroy_board(b);
    }

    printf("Found %ld initial board states\n", search_queue->pointer);

    // Distribute the initial states to a set of new pthreads.
    ptr_arraylist threads = create_ptr_arraylist(search_queue->pointer + 1);

    for(uint64_t t = 0; t < search_queue->pointer; t++) {
        pthread_t* thread_id = (pthread_t*)malloc(sizeof(pthread_t));
        if(!thread_id) err(1, "Memory error while allocating thread id\n");

        processor_args args = create_processor_args(t, search_queue->data[t], cache, &count, &counter_lock);

        // walker_processor(args);
        pthread_create(thread_id, 0, walker_processor, (void*)args);
        append_pal(threads, thread_id);
    }

    // for(uint64_t t = 0; t < threads->pointer; t++) pthread_join(*(pthread_t*)(threads->data[0]), 0);
    while(1) {
        printf("\rFound %ld boards", count);
        sched_yield();
    }

    printf("\nThere are %ld possible board states\n", count);
}