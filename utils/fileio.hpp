#pragma once

#include <stdio.h>
#include <stddef.h>
#include <pthread.h>
#include <stdint.h>

#include "tarraylist.hpp"
#include "../mem_man/heir.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void clear_file_cache();

#ifdef __cplusplus
}
#endif

#pragma region checkpoint saving and restoring

/**
 * - Cache
 * - Current final board count/visited board count
 * - The search queue for each thread
 * 
 * write how many threads there are
 * send a signal to the threads to save
 * TODO make sure to have a mutex lock for the threads when writing to file
 * wait for the threads to save their queues, finishing the last board they were working on
 * save the cache
 * save the counts
 * exit
 * 
 * set fp to end of file
 * use 'ab+' mode
 * 
 */

typedef struct __processed_file_str {
    uint64_t found_counter;
    uint64_t explored_counter;
    uint64_t num_processors;
    heirarchy cache;
    Arraylist<void*>* processor_stacks;
} processed_file_str;

/**
 * @brief Represents a processed file that can be used to restore the walker to a previous point
 * 
 */
typedef processed_file_str* processed_file;

typedef struct __processed_file_3_str {
    uint64_t found_counter;
    uint64_t explored_counter;
    heirarchy cache;
    uint8_t level;
    board* last_level;
    size_t level_n;
} processed_file_3_str;

/**
 * @brief Represents a processed file that can be used to restore the walker to a previous point
 * 
 */
typedef processed_file_3_str* processed_file_3;

#pragma region version 1

// /**
//  * @brief Called by the main thread to cause the system to save itself to a checkpoint file
//  * 
//  * @param checkpoint_file A FILE pointer pointer that is populated and used by the processors to save themselves once the file is open.
//  * @param file_lock The lock used to stop simultaneous file access
//  * @param filename The name of the file to save the checkpoint to
//  * @param found_counter The number of final board states found so far
//  * @param explored_counter The number of boards that have been explored so far
//  * @param num_processors The number of processors currently running
//  */
// void save_progress(FILE** checkpoint_file, pthread_mutex_t* file_lock, char* filename, 
//                    uint64_t* saving_counter, hashtable cache,
//                    uint64_t found_counter, uint64_t explored_counter, uint64_t num_processors);

// /**
//  * @brief Reads and populates the fields of the processed file from the given file, used to restore progress
//  * 
//  * @param filename The file name to restore progress from
//  * @param hash The hash function to use for the cache, because it can't be saved.
//  * @return processed_file Returns a processed file containing all of the arguments required to restore progress, must be free'd by the user.
//  */
// processed_file restore_progress(char* filename, __uint128_t (*hash)(void*));

#pragma endregion
#pragma region version 2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Called by the main thread to cause the system to save itself to a checkpoint file
 * 
 * @param checkpoint_file A FILE pointer pointer that is populated and used by the processors to save themselves once the file is open.
 * @param file_lock The lock used to stop simultaneous file access
 * @param filename The name of the file to save the checkpoint to
 * @param found_counter The number of final board states found so far
 * @param explored_counter The number of boards that have been explored so far
 * @param num_processors The number of processors currently running
 */
void save_progress_v2(FILE** checkpoint_file, pthread_mutex_t* file_lock, char* filename, 
                   uint64_t* saving_counter, heirarchy cache,
                   uint64_t found_counter, uint64_t explored_counter, uint64_t repeated_counter, uint64_t num_processors);

/**
 * @brief Reads and populates the fields of the processed file from the given file, used to restore progress
 * 
 * @param filename The file name to restore progress from
 * @return processed_file Returns a processed file containing all of the arguments required to restore progress, must be free'd by the user.
 */
processed_file restore_progress_v2(char* filename);

#ifdef __cplusplus
}
#endif

#pragma endregion

#pragma region version 3

/**
 * @brief Called by the main thread to cause the system to save itself to a checkpoint file
 * 
 * @param checkpoint_file A FILE pointer pointer that is populated and used by the processors to save themselves once the file is open.
 * @param file_lock The lock used to stop simultaneous file access
 * @param filename The name of the file to save the checkpoint to
 * @param found_counter The number of final board states found so far
 * @param explored_counter The number of boards that have been explored so far
 * @param level The number of the level that was last completed
 * @param last_completed_level The boards that were in the last completed level
 * @param level_n The number of boards in the last completed level
 */
void save_progress_v3(FILE** checkpoint_file, pthread_mutex_t* file_lock, char* filename, 
                   heirarchy cache, uint8_t level, __uint128_t* last_completed_level, size_t level_n, 
                   uint64_t found_counter, uint64_t explored_counter, uint64_t repeated_counter);

processed_file_3 restore_progress_v3(char* filename);

#pragma endregion
#pragma endregion