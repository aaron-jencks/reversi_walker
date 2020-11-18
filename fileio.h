#pragma once

#include <stdio.h>
#include <pthread.h>
#include <stdint.h>

#include "hashtable.h"
#include "arraylist.h"

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
    hashtable cache;
    ptr_arraylist processor_stacks;
} processed_file_str;

/**
 * @brief Represents a processed file that can be used to restore the walker to a previous point
 * 
 */
typedef processed_file_str* processed_file;

/**
 * @brief Generates a unique filename in the /tmp directory for checkpoint saving
 * 
 * @param filename The name of the file to save the data to in the /tmp directory (ex. "checkpoint.bin")
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary filename.
 */
char* find_temp_filename(const char* filename);

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
void save_progress(FILE** checkpoint_file, pthread_mutex_t* file_lock, char* filename, 
                   uint64_t* saving_counter, hashtable cache,
                   uint64_t found_counter, uint64_t explored_counter, uint64_t num_processors);

/**
 * @brief Reads and populates the fields of the processed file from the given file, used to restore progress
 * 
 * @param filename The file name to restore progress from
 * @param hash The hash function to use for the cache, because it can't be saved.
 * @return processed_file Returns a processed file containing all of the arguments required to restore progress, must be free'd by the user.
 */
processed_file restore_progress(char* filename, __uint128_t (*hash)(void*));