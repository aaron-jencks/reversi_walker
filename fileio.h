#pragma once

#include <stdio.h>
#include <pthread.h>
#include <stdint.h>

#include "hashtable.h"

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