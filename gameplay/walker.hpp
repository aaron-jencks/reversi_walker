#pragma once

#include "reversi.h"
#include "../hashing/hashtable.h"
#include "../utils/tarraylist.hpp"
#include "../mem_man/heir.hpp"
#include "reversi_defs.h"  

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

typedef struct _processor_args_str {
    uint32_t identifier;
    board_t starting_board;
    heirarchy cache; 
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
    Arraylist<board_t>* stack;
} processor_args_str;

/**
 * @brief Represents a wrapper for passing arguments to the walker processor using pthreads
 * 
 */
typedef processor_args_str* processor_args;

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
 * @return coord* Returns a zero-terminated array of coordinates that represent valid moves from this board_t
 */
void find_next_boards(board_t b, Arraylist<void*>* coord_buff, Arraylist<void*>* coord_cache);

// /**
//  * @brief Finds the next set of boards that can be reached from this one,
//  * but only checks the 8 cells surrounding coordinate c
//  * 
//  * @param b 
//  * @param c 
//  * @return coord* Returns a zero-terminated array of coordinates that represent valid moves from this board_t
//  */
// coord* find_next_boards_from_coord(board_t b, coord c);

/**
 * @brief Finds the next set of boards that can be reached from this one,
 * but only checks the 8 cells surrounding coordinate c
 * 
 * @param b 
 * @param c 
 * @return Arraylist<coord_t> returns an array of coordinates that represent valid moves from this board_t
 */
Arraylist<coord_t> find_next_boards_from_coord_opposing_player(board_t b, coord_t c);

coord_t create_coord(uint8_t row, uint8_t column);
uint16_t coord_to_short(coord_t c);
uint16_t coord_to_short_ints(uint8_t r, uint8_t c);
coord_t short_to_coord(uint16_t s);

processor_args create_processor_args(uint32_t identifier, board_t starting_board, heirarchy cache, 
                                     uint64_t* counter, pthread_mutex_t* counter_lock,
                                     uint64_t* explored_counter, pthread_mutex_t* explored_lock,
                                     uint64_t* repeated_counter, pthread_mutex_t* repeated_lock,
                                     uint64_t* saving_counter, FILE** checkpoint_file, pthread_mutex_t* file_lock,
                                     size_t* finished_count, pthread_mutex_t* finished_lock);

/**
 * @brief Used to launch child threads, accepts a board_t to start from and a pointer to the cache to use
 * 
 * @param args The args are a struct containing the following information:
 * 
 * @param identifier the identifier for this processor
 * @param starting_board board_t to start from
 * @param cache The existence cache to use
 * @param counter a pointer to the current board_t counter
 * @param counter_lock a lock for the board_t counter
 * 
 * @return uint64_t Returns the number of boards that this thread counted.
 */
void* walker_processor(void* args);

/**
 * @brief Used to launch child threads, accepts a board_t to start from and a pointer to the cache to use
 * 
 * @param args The args are a struct containing the following information:
 * 
 * @param identifier the identifier for this processor
 * @param starting_board The pre-initialized stack for the walker to start its search from
 * @param cache The existence cache to use
 * @param counter a pointer to the current board_t counter
 * @param counter_lock a lock for the board_t counter
 * 
 * @return uint64_t Returns the number of boards that this thread counted.
 */
void* walker_processor_pre_stacked(void* args);

/**
 * @brief Writes a thread to file, saves the search stack
 * 
 * @param fp 
 * @param search_stack 
 */
void walker_to_file(FILE* fp, Arraylist<board_t>* search_stack);