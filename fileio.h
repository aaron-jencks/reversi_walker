#pragma once

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