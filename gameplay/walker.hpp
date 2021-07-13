#pragma once

#include "reversi.h"
#include "../hashing/hashtable.h"
#include "../utils/tarraylist.hpp"
#include "../utils/pqueue.hpp"
#include "../mem_man/heir.hpp"
#include "../mem_man/dheap.hpp"
#include "reversi_defs.h"  

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

typedef struct _processor_args_str {
    uint32_t identifier;
    LockedRingBuffer<board>* inputq;
    MMappedLockedPriorityQueue<board>* outputq;
    heirarchy cache; 
    Arraylist<board>* board_cache;
    Arraylist<coord>* coord_cache;
    uint64_t* counter;
    pthread_mutex_t* counter_lock;
    uint64_t* explored_counter;
    pthread_mutex_t* explored_lock;
    uint64_t* repeated_counter;
    pthread_mutex_t* repeated_lock;
    uint64_t* saving_counter;
    FILE** checkpoint_file;
    pthread_mutex_t* file_lock;
    size_t* finished_count;
    pthread_mutex_t* finished_lock;
} processor_args_str;

/**
 * @brief Represents a wrapper for passing arguments to the walker processor using pthreads
 * 
 */
typedef processor_args_str* processor_args;

/**
 * @brief Represents a wrapper for passing arguments to the walker task scheduler
 * 
 */
typedef struct _processor_scheduler_args_t {
    size_t nprocs;
    heirarchy cache;
    MMappedLockedPriorityQueue<board>* inputq;
    uint64_t* counter;
    pthread_mutex_t* counter_lock;
    uint64_t* explored_counter;
    pthread_mutex_t* explored_lock;
    uint64_t* repeated_counter;
    pthread_mutex_t* repeated_lock;
    uint64_t* saving_counter;
    char* checkpoint_file;
    pthread_mutex_t* file_lock;
    size_t* finished_count;
    pthread_mutex_t* finished_lock;
} processor_scheduler_args_t;

/**
 * @brief when set to > 0, will cause all walker_processor calls to save their work.
 * 
 */
extern uint8_t SAVING_FLAG;

/**
 * @brief when set to > 0, will cause all the walker_processor calls to exit.
 * 
 */
extern uint8_t WALKER_KILL_FLAG;
extern pthread_mutex_t saving_lock;

/**
 * @brief Finds the next set of boards that can be reached from this one
 * 
 * @param b 
 * @return coord* Returns a zero-terminated array of coordinates that represent valid moves from this board
 */
void find_next_boards(board b, Arraylist<void*>* coord_buff, Arraylist<void*>* coord_cache);

// /**
//  * @brief Finds the next set of boards that can be reached from this one,
//  * but only checks the 8 cells surrounding coordinate c
//  * 
//  * @param b 
//  * @param c 
//  * @return coord* Returns a zero-terminated array of coordinates that represent valid moves from this board
//  */
// coord* find_next_boards_from_coord(board b, coord c);

/**
 * @brief Finds the next set of boards that can be reached from this one,
 * but only checks the 8 cells surrounding coordinate c
 * 
 * @param b 
 * @param c 
 * @return coord* Returns a zero-terminated array of coordinates that represent valid moves from this board
 */
coord* find_next_boards_from_coord_opposing_player(board b, coord c);

coord create_coord(uint8_t row, uint8_t column);
uint16_t coord_to_short(coord c);
uint16_t coord_to_short_ints(uint8_t r, uint8_t c);
coord short_to_coord(uint16_t s);

processor_args create_processor_args(uint32_t identifier, LockedRingBuffer<board>* inputq, MMappedLockedPriorityQueue<board>* outputq, 
                                     heirarchy cache, Arraylist<board>* board_cache, Arraylist<coord>* coord_cache, 
                                     uint64_t* counter, pthread_mutex_t* counter_lock,
                                     uint64_t* explored_counter, pthread_mutex_t* explored_lock,
                                     uint64_t* repeated_counter, pthread_mutex_t* repeated_lock,
                                     uint64_t* saving_counter, FILE** checkpoint_file, pthread_mutex_t* file_lock,
                                     size_t* finished_count, pthread_mutex_t* finished_lock);

processor_scheduler_args_t* create_processor_scheduler_args(heirarchy cache, MMappedLockedPriorityQueue<board>* inputq, size_t nprocs, 
                                                            uint64_t* counter, pthread_mutex_t* counter_lock,
                                                            uint64_t* explored_counter, pthread_mutex_t* explored_lock,
                                                            uint64_t* repeated_counter, pthread_mutex_t* repeated_lock,
                                                            uint64_t* saving_counter, char* checkpoint_file, pthread_mutex_t* file_lock,
                                                            size_t* finished_count, pthread_mutex_t* finished_lock);

/**
 * @brief Used to launch child threads, accepts a board to start from and a pointer to the cache to use
 * 
 * @param args The args are a struct containing the following information:
 * 
 * @param identifier the identifier for this processor
 * @param starting_board Board to start from
 * @param cache The existence cache to use
 * @param counter a pointer to the current board counter
 * @param counter_lock a lock for the board counter
 * 
 * @return uint64_t Returns the number of boards that this thread counted.
 */
void* walker_processor(void* args);

/**
 * @brief Used to launch child threads, accepts a board to start from and a pointer to the cache to use
 * 
 * @param args The args are a struct containing the following information:
 * 
 * @param identifier the identifier for this processor
 * @param starting_board The pre-initialized stack for the walker to start its search from
 * @param cache The existence cache to use
 * @param counter a pointer to the current board counter
 * @param counter_lock a lock for the board counter
 * 
 * @return uint64_t Returns the number of boards that this thread counted.
 */
// void* walker_processor_pre_stacked(void* args);

/**
 * @brief Writes a thread to file, saves the search stack
 * 
 * @param fp 
 * @param search_stack 
 */
void walker_to_file(FILE* fp, Arraylist<void*>* search_stack);

/**
 * @brief Manages the thread launching as maintains knowledge of how many boards are left in each level that need to be processed
 * Also manages clearing previous levels from the cache.
 * 
 * @param cache The existence cache to use
 * @param inputq The queue of boards that need to be processed
 * 
 * @return void* 
 */
void* walker_task_scheduler(void* args);