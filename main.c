#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <err.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <string.h>

#include "reversi.h"
#include "hashtable.h"
#include "lookup3.h"
#include "walker.h"
#include "ll.h"
#include "arraylist.h"
#include "valid_moves.h"
#include "fileio.h"

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
 * TODO add ability to make checkpoints to save progress
 *  - Interpret Ctrl+C interrupt signal and cause a save
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

        // NO YOU CAN'T, but luckily, I don't actually need the player
        // You can fit 2 spaces in 3 bits if you really try,
        // so on an 8x8 board, 
        // we end up only using 96 bits instead of the entire 128.
        // well
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c++) {
                uint8_t s1 = board_get(b, r, c);

                result += s1;

                if(c < b->width - 1) result = result << 2;
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
    // TODO create gui for deciding whether or not to restore from a checkpoint
    // ui.o
    // 
    char d;
    while(1) {
        printf("Would you like to restore from a checkpoint?(y/N): ");
        d = getc(stdin);
        if(d == '\n' || d == 'n' || d == 'N') {
            d = 'n';
            break;
        }
        else if(d == 'y' || d == 'Y') {
            d = 'y';
            break;
        }
    }

    getc(stdin);    // Read the extra \n character

    // Allocate all of the stack parameters necessary for file restoration
    hashtable cache;
    uint64_t count = 0, explored_count = 1;
    char* checkpoint_filename;

    // Calculate the number of processors to use
    uint32_t procs = get_nprocs();
    #ifndef limitprocs
        procs = procs << 1;
    #endif

    // Setup the locks
    pthread_mutex_t counter_lock, explored_lock, file_lock;

    // Setup the checkpoint saving system
    FILE** checkpoint_file = malloc(sizeof(FILE*));
    if(!checkpoint_file) err(1, "Memory error while allocating checkpoint file pointer\n");
    uint64_t saving_counter;

    // Initialize the locks
    if(pthread_mutex_init(&counter_lock, 0) || pthread_mutex_init(&explored_lock, 0) || 
       pthread_mutex_init(&file_lock, 0) || pthread_mutex_init(&saving_lock, 0)) 
        err(4, "Initialization of counter mutex failed\n");

    #pragma region Round nprocs to the correct number
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

        explored_count++;

        free(next_moves);
        destroy_board(b);
    }

    procs = search_queue->pointer;

    printf("Rounded nprocs to %d threads\n", procs);
    #pragma endregion

    ptr_arraylist threads;

    if(d == 'y') {
        char** restore_filename;
        scanf("Please enter a file to restore from: %ms", restore_filename);
        processed_file pf = restore_progress(*restore_filename, &board_hash);

        while(1) {
            printf("Would you like to continue saving to this checkpoint?(y/N): ");
            d = getc(stdin);
            if(d == '\n' || d == 'n' || d == 'N') {
                checkpoint_filename = find_temp_filename("checkpoint.bin\0");
                break;
            }
            else if(d == 'y' || d == 'Y') {
                checkpoint_filename = malloc(sizeof(char) * strlen(*restore_filename));
                strcpy(checkpoint_filename, *restore_filename);
                break;
            }
        }

        free(*restore_filename);

        // De-allocate the stuff we did to round procs, because we don't need it now.
        while(search_queue->pointer) destroy_board(pop_front_pal(search_queue));
        destroy_ptr_arraylist(search_queue);

        // Begin restore
        count = pf->found_counter;
        explored_count = pf->explored_counter;

        ptr_arraylist stacks = pf->processor_stacks;
        if(procs != pf->num_processors) {
            // Redistribute workload
            ptr_arraylist all_current_boards = create_ptr_arraylist(procs * 1000), important_boards = create_ptr_arraylist(pf->num_processors + 1);
            for(ptr_arraylist* p = (ptr_arraylist*)pf->processor_stacks->data; *p; p++) {
                for(uint64_t pb = 0; pb < (*p)->pointer; pb++) append_pal((pb) ? all_current_boards : important_boards, (*p)->data[pb]);
                destroy_ptr_arraylist(*p);
            }

            while(stacks->pointer) pop_back_pal(stacks);
            
            for(uint64_t p = 0; p < procs; p++) append_pal(stacks, create_ptr_arraylist(1000));
            uint64_t p_ptr = 0;

            while(important_boards->pointer) {
                append_pal(stacks->data[p_ptr++], pop_back_pal(important_boards));
                if(p_ptr == procs) p_ptr = 0;
            }

            while(all_current_boards->pointer) {
                append_pal(stacks->data[p_ptr++], pop_back_pal(all_current_boards));
                if(p_ptr == procs) p_ptr = 0;
            }
        }

        // Create threads

        // Distribute the initial states to a set of new pthreads.
        threads = create_ptr_arraylist(procs + 1);

        for(uint64_t t = 0; t < procs; t++) {
            pthread_t* thread_id = (pthread_t*)malloc(sizeof(pthread_t));
            if(!thread_id) err(1, "Memory error while allocating thread id\n");

            processor_args args = create_processor_args(t, stacks->data[t], cache, 
                                                        &count, &counter_lock,
                                                        &explored_count, &explored_lock,
                                                        &saving_counter, checkpoint_file, &file_lock);

            // walker_processor(args);
            pthread_create(thread_id, 0, walker_processor_pre_stacked, (void*)args);
            append_pal(threads, thread_id);
        }
    }
    else {
        #ifdef smallcache
            cache = create_hashtable(10, &board_hash);
        #else
            cache = create_hashtable(1000000, &board_hash);
        #endif

        // Distribute the initial states to a set of new pthreads.
        threads = create_ptr_arraylist(procs + 1);

        for(uint64_t t = 0; t < procs; t++) {
            pthread_t* thread_id = (pthread_t*)malloc(sizeof(pthread_t));
            if(!thread_id) err(1, "Memory error while allocating thread id\n");

            processor_args args = create_processor_args(t, search_queue->data[t], cache, 
                                                        &count, &counter_lock,
                                                        &explored_count, &explored_lock,
                                                        &saving_counter, checkpoint_file, &file_lock);

            // walker_processor(args);
            pthread_create(thread_id, 0, walker_processor, (void*)args);
            append_pal(threads, thread_id);
        }
    }

    printf("Starting walk...\n");
    printf("Running on %d threads\n", procs);

    // for(uint64_t t = 0; t < threads->pointer; t++) pthread_join(*(pthread_t*)(threads->data[0]), 0);
    time_t start = time(0), current, save_timer = time(0);
    clock_t cstart = clock();
    uint32_t cpu_time, cpu_days, cpu_hours, cpu_minutes, cpu_seconds,
             run_time, run_days, run_hours, run_minutes, run_seconds, save_time, previous_run_time = start;
    uint64_t previous_board_count = 0;
    while(1) {
        current = time(0);
        run_time = current - start;
        save_time = (current - save_timer) / 3600;
        run_days = run_time / 86400;
        run_hours = (run_time / 3600) % 24;
        run_minutes = (run_time / 60) % 60;
        run_seconds = run_time % 60;
        cpu_time = (uint32_t)(((double)(clock() - cstart)) / CLOCKS_PER_SEC);
        cpu_days = cpu_time / 86400;
        cpu_hours = (cpu_time / 3600) % 24;
        cpu_minutes = (cpu_time / 60) % 60;
        cpu_seconds = cpu_time % 60;
        printf("\rFound %ld final board states. Explored %ld boards @ %f boards/sec. Runtime: %0d:%02d:%02d:%02d CPU Time: %0d:%02d:%02d:%02d %s", 
               count, explored_count, ((double)(explored_count - previous_board_count)) / ((double)(run_time - previous_run_time)),
               run_days, run_hours, run_minutes, run_seconds,
               cpu_days, cpu_hours, cpu_minutes, cpu_seconds,
               (save_time) ? "Saving..." : "");
        fflush(stdout);

        previous_board_count = explored_count;
        previous_run_time = run_time;

        if(save_time) {
            save_progress(checkpoint_file, &file_lock, checkpoint_filename, &saving_counter, cache, count, explored_count, procs);
            save_timer = time(0);
        }
        sched_yield();
    }

    printf("\nThere are %ld possible board states\n", count);
}
