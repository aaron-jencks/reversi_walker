#include "fileio.hpp"
#include "../gameplay/walker.hpp"
#include "../utils/path_util.h"

#include <err.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <sys/stat.h>
#include <fcntl.h>

void clear_file_cache() {
    printf("Clearing filesystem cache\n");
    while(pthread_mutex_trylock(&heirarchy_lock)) sched_yield();
    sync();
    int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
    write(fd, "3", sizeof(char));
    close(fd);
    pthread_mutex_unlock(&heirarchy_lock);
}

#pragma region checkpoing saving and restoring

#ifdef filedebug
void display_board_f(board b) {
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
#endif

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
//                    uint64_t found_counter, uint64_t explored_counter, uint64_t num_processors) {
//     printf("\nStarting save...\n");
//     *checkpoint_file = fopen(filename, "wb+");
//     if(!*checkpoint_file) err(7, "Unable to open or create file %s\n", filename);
//     // printf("Saving child thread search queues\n");
//     *saving_counter = 0;

//     // Save the counts
//     printf("Saving the walk counters\n");
//     while(pthread_mutex_trylock(file_lock)) sched_yield();
//     fwrite(&found_counter, sizeof(found_counter), 1, *checkpoint_file);
//     fwrite(&explored_counter, sizeof(explored_counter), 1, *checkpoint_file);
//     fwrite(&num_processors, sizeof(num_processors), 1, *checkpoint_file);
//     pthread_mutex_unlock(file_lock);

//     // Save the threads
//     printf("Saving child thread search queues\n");
//     while(pthread_mutex_trylock(&saving_lock)) sched_yield();
//     SAVING_FLAG = 1;
//     pthread_mutex_unlock(&saving_lock);

//     uint64_t temp_saving_counter = 0;
//     while(temp_saving_counter < num_processors) {
//         while(pthread_mutex_trylock(file_lock)) sched_yield();
//         printf("\r%0lu/%ld", *saving_counter, num_processors);
//         temp_saving_counter = *saving_counter;
//         pthread_mutex_unlock(file_lock);
//         sched_yield();
//     }

//     // Save the hashtable
//     printf("\nSaving the hashtable\n");
//     while(pthread_mutex_trylock(file_lock)) sched_yield();
//     to_file_hs(*checkpoint_file, cache);
//     pthread_mutex_unlock(file_lock);

//     fclose(*checkpoint_file);

//     while(pthread_mutex_trylock(&saving_lock)) sched_yield();
//     SAVING_FLAG = 0;
//     pthread_mutex_unlock(&saving_lock);
// }

// processed_file restore_progress(char* filename, __uint128_t (*hash)(void*)) {
//     FILE* fp = fopen(filename, "rb+");
//     if(!fp) err(7, "Cannot find/open given restore file %s\n", filename);

//     printf("Restoring from file %s\n", filename);

//     processed_file result = calloc(1, sizeof(processed_file_str));
//     if(!result) err(1, "Memory error while allocating processed file\n");

//     // Read the counters
//     fread(&result->found_counter, sizeof(result->found_counter), 1, fp);
//     fread(&result->explored_counter, sizeof(result->explored_counter), 1, fp);
//     fread(&result->num_processors, sizeof(result->num_processors), 1, fp);
//     result->processor_stacks = new Arraylist<void*>(result->num_processors + 1);
//     printf("Restored progress, %lu final boards found, %lu boards explored, %lu processors\n", 
//            result->found_counter, result->explored_counter, result->num_processors);

//     // Read the processors
//     uint8_t player;
//     uint64_t board_upper, board_lower;
//     __uint128_t board_key;
//     for(uint64_t p = 0; p < result->num_processors; p++) {
//         Arraylist<void*>* stack = new Arraylist<void*>(1000);

//         while(1) {
//             player = 0;
//             board_key = 0;

//             fread(&player, sizeof(uint8_t), 1, fp);
//             fread(&board_key, sizeof(__uint128_t), 1, fp);
//             // fscanf(fp, "%c%lu%lu", &player, &board_upper, &board_lower);

//             #ifdef filedebug
//                 printf("Found board %u: %lu %lu\n", player, ((uint64_t*)(&board_key))[1], ((uint64_t*)(&board_key))[0]);
//             #endif

//             // // Combine the upper and lower keys
//             // board_key = board_upper;
//             // board_key = board_key << 64;
//             // board_key += board_lower;

//             if(player || board_key) {
//                 board b = create_board_unhash_6(player, board_key);

//                 #ifdef filedebug
//                     display_board_f(b);
//                 #endif

//                 stack->append(b);
//             }
//             else {
//                 break;
//             }
//         }

//         printf("Read a processor with a stack of %lu elements\n", stack->pointer);
//         result->processor_stacks->append(stack);
//     }

//     // Read in the hashtable
//     printf("Restoring cache\n");
//     result->cache = (heirarchy)from_file_hs(fp, hash);

//     return result;
// }

#pragma endregion
#pragma region version 2

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
                   uint64_t found_counter, uint64_t explored_counter, uint64_t repeated_counter, uint64_t num_processors) {
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
    to_file_heir(*checkpoint_file, cache);
    pthread_mutex_unlock(file_lock);

    fclose(*checkpoint_file);

    clear_file_cache();

    while(pthread_mutex_trylock(&saving_lock)) sched_yield();
    SAVING_FLAG = 0;
    pthread_mutex_unlock(&saving_lock);
}

processed_file restore_progress_v2(char* filename) {
    FILE* fp = fopen(filename, "rb+");
    if(!fp) err(7, "Cannot find/open given restore file %s\n", filename);

    printf("Restoring from file %s\n", filename);

    processed_file result = (processed_file)calloc(1, sizeof(processed_file_str));
    if(!result) err(1, "Memory error while allocating processed file\n");

    // Read the counters
    fread(&result->found_counter, sizeof(result->found_counter), 1, fp);
    fread(&result->explored_counter, sizeof(result->explored_counter), 1, fp);
    fread(&result->num_processors, sizeof(result->num_processors), 1, fp);
    result->processor_stacks = new Arraylist<void*>(result->num_processors + 1);
    printf("Restored progress, %lu final boards found, %lu boards explored, %lu processors\n", 
           result->found_counter, result->explored_counter, result->num_processors);

    // Read the processors
    uint8_t player;
    uint64_t board_upper, board_lower;
    __uint128_t board_key;
    for(uint64_t p = 0; p < result->num_processors; p++) {
        Arraylist<void*>* stack = new Arraylist<void*>(1000);

        while(1) {
            player = 0;
            board_key = 0;

            fread(&player, sizeof(uint8_t), 1, fp);
            fread(&board_key, sizeof(__uint128_t), 1, fp);
            // fscanf(fp, "%c%lu%lu", &player, &board_upper, &board_lower);

            #ifdef filedebug
                printf("Found board %u: %lu %lu\n", player, ((uint64_t*)(&board_key))[1], ((uint64_t*)(&board_key))[0]);
            #endif

            // // Combine the upper and lower keys
            // board_key = board_upper;
            // board_key = board_key << 64;
            // board_key += board_lower;

            if(player || board_key) {
                board b = create_board_unhash_6(player, board_key);

                #ifdef filedebug
                    display_board_f(b);
                #endif

                stack->append(b);
            }
            else {
                break;
            }
        }

        printf("Read a processor with a stack of %lu elements\n", stack->pointer);
        result->processor_stacks->append(stack);
    }

    // Read in the hashtable
    printf("Restoring cache\n");
    result->cache = from_file_heir(fp);
    clear_file_cache();

    return result;
}

#pragma endregion

#pragma endregion