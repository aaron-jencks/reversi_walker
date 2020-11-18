#include "fileio.h"
#include "walker.h"

#include <err.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Generates a unique filename in the /tmp directory for checkpoint saving
 * 
 * @param filename The name of the file to save the data to in the /tmp directory (ex. "checkpoint.bin"), must be null-terminated
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary filename. Must be free'd by the user
 */
char* find_temp_filename(const char* filename) {
    // Setup the checkpoint saving system
    char temp_str[] = "/tmp/reversi.XXXXXX\0", *final_result = calloc(35, sizeof(char));
    char* dir_ptr = mkdtemp(temp_str);
    strncat(final_result, dir_ptr, 20);
    final_result[19] = '/';
    strncat(final_result + 20, filename, 14);
    #ifdef checkpointdebug
        printf("Saving to %s\n", final_result);
    #endif
    return final_result;
}

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
                   uint64_t found_counter, uint64_t explored_counter, uint64_t num_processors) {
    printf("\nStarting save...\n");
    *checkpoint_file = fopen(filename, "wb+");
    if(!*checkpoint_file) err(7, "Unable to open or create file %s\n", filename);
    // printf("Saving child thread search queues\n");
    *saving_counter = 0;

    // Save the counts
    printf("Saving the walk counters\n");
    while(pthread_mutex_trylock(file_lock)) sched_yield();
    fwrite(&found_counter, sizeof(found_counter), 1, *checkpoint_file);
    fwrite(&explored_counter, sizeof(explored_counter), 1, *checkpoint_file);
    fwrite(&num_processors, sizeof(num_processors), 1, *checkpoint_file);
    pthread_mutex_unlock(file_lock);

    // Save the threads
    printf("Saving child thread search queues\n");
    while(pthread_mutex_trylock(&saving_lock)) sched_yield();
    SAVING_FLAG = 1;
    pthread_mutex_unlock(&saving_lock);

    uint64_t temp_saving_counter = 0;
    while(temp_saving_counter < num_processors) {
        while(pthread_mutex_trylock(file_lock)) sched_yield();
        printf("\r%0lu/%ld", *saving_counter, num_processors);
        temp_saving_counter = *saving_counter;
        pthread_mutex_unlock(file_lock);
        sched_yield();
    }

    // Save the hashtable
    printf("\nSaving the hashtable\n");
    while(pthread_mutex_trylock(file_lock)) sched_yield();
    to_file_hs(*checkpoint_file, cache);
    pthread_mutex_unlock(file_lock);

    fclose(*checkpoint_file);

    while(pthread_mutex_trylock(&saving_lock)) sched_yield();
    SAVING_FLAG = 0;
    pthread_mutex_unlock(&saving_lock);
}

processed_file restore_progress(char* filename, __uint128_t (*hash)(void*)) {
    FILE* fp = fopen(filename, "wb+");
    if(!fp) err(7, "Cannot find/open given restore file %s\n", filename);

    printf("Restoring from file %s\n", filename);

    processed_file result = malloc(sizeof(processed_file_str));
    if(!result) err(1, "Memory error while allocating processed file\n");

    // Read the counters
    fscanf(fp, "%lu%lu%lu", &result->found_counter, &result->explored_counter, &result->num_processors);
    result->processor_stacks = create_ptr_arraylist(result->num_processors + 1);
    printf("Saved progress, %lu final boards found, %lu boards explored, %lu processors\n", 
           result->found_counter, result->explored_counter, result->num_processors);

    // Read the processors
    uint8_t player;
    uint64_t board_upper, board_lower;
    __uint128_t board_key;
    for(uint64_t p = 0; p < result->num_processors; p++) {
        ptr_arraylist stack = create_ptr_arraylist(1000);

        while(1) {
            fscanf(fp, "%c%lu%lu", &player, &board_upper, &board_lower);

            // Combine the upper and lower keys
            board_key = board_upper;
            board_key = board_key << 64;
            board_key += board_lower;

            if(player || board_key) {
                board b = create_board_unhash_6(player, board_key);
                append_pal(stack, b);
            }
            else {
                break;
            }
        }

        printf("Read a processor with a stack of %lu elements\n", stack->pointer);
        append_pal(result->processor_stacks, stack);
    }

    // Read in the hashtable
    result->cache = from_file_hs(fp, hash);

    return result;
}