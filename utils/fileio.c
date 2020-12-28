#include "fileio.h"
#include "../gameplay/walker.h"

#include <err.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <stdio.h>

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

/**
 * @brief Generates a unique filename in the /tmp directory for checkpoint saving
 * 
 * @param filename The name of the file to save the data to in the /tmp directory (ex. "checkpoint.bin"), must be null-terminated
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary filename. Must be free'd by the user
 */
char* find_temp_filename(const char* filename) {
    // Setup the checkpoint saving system
    char temp_str[] = "/tmp/reversi.XXXXXX", *final_result = calloc(35, sizeof(char));
    char* dir_ptr = mkdtemp(temp_str);
    strncat(final_result, dir_ptr, 20);
    final_result[19] = '/';
    strncat(final_result + 20, filename, 14);
    #ifdef checkpointdebug
        printf("Saving to %s\n", final_result);
    #endif
    return final_result;
}

#pragma region version 1

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
    FILE* fp = fopen(filename, "rb+");
    if(!fp) err(7, "Cannot find/open given restore file %s\n", filename);

    printf("Restoring from file %s\n", filename);

    processed_file result = calloc(1, sizeof(processed_file_str));
    if(!result) err(1, "Memory error while allocating processed file\n");

    // Read the counters
    fread(&result->found_counter, sizeof(result->found_counter), 1, fp);
    fread(&result->explored_counter, sizeof(result->explored_counter), 1, fp);
    fread(&result->num_processors, sizeof(result->num_processors), 1, fp);
    result->processor_stacks = create_ptr_arraylist(result->num_processors + 1);
    printf("Restored progress, %lu final boards found, %lu boards explored, %lu processors\n", 
           result->found_counter, result->explored_counter, result->num_processors);

    // Read the processors
    uint8_t player;
    uint64_t board_upper, board_lower;
    __uint128_t board_key;
    for(uint64_t p = 0; p < result->num_processors; p++) {
        ptr_arraylist stack = create_ptr_arraylist(1000);

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
    printf("Restoring cache\n");
    result->cache = from_file_hs(fp, hash);

    return result;
}

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
    to_file_heir(*checkpoint_file, cache);
    pthread_mutex_unlock(file_lock);

    fclose(*checkpoint_file);

    while(pthread_mutex_trylock(&saving_lock)) sched_yield();
    SAVING_FLAG = 0;
    pthread_mutex_unlock(&saving_lock);
}

processed_file restore_progress_v2(char* filename) {
    FILE* fp = fopen(filename, "rb+");
    if(!fp) err(7, "Cannot find/open given restore file %s\n", filename);

    printf("Restoring from file %s\n", filename);

    processed_file result = calloc(1, sizeof(processed_file_str));
    if(!result) err(1, "Memory error while allocating processed file\n");

    // Read the counters
    fread(&result->found_counter, sizeof(result->found_counter), 1, fp);
    fread(&result->explored_counter, sizeof(result->explored_counter), 1, fp);
    fread(&result->num_processors, sizeof(result->num_processors), 1, fp);
    result->processor_stacks = create_ptr_arraylist(result->num_processors + 1);
    printf("Restored progress, %lu final boards found, %lu boards explored, %lu processors\n", 
           result->found_counter, result->explored_counter, result->num_processors);

    // Read the processors
    uint8_t player;
    uint64_t board_upper, board_lower;
    __uint128_t board_key;
    for(uint64_t p = 0; p < result->num_processors; p++) {
        ptr_arraylist stack = create_ptr_arraylist(1000);

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
    printf("Restoring cache\n");
    result->cache = from_file_heir(fp);

    return result;
}

#pragma endregion

#pragma endregion
#pragma region mempage swapping

/**
 * @brief Generates a unique directory in the /home/$USER/Temp directory for mempage swapping
 * 
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary directory. Must be free'd by the user
 */
char* find_temp_directory() {
    // Setup the checkpoint saving system
    struct passwd *p = getpwuid(getuid());  // Check for NULL!
    if(!p) err(1, "Memory Error while getting username for page swap\n");

    char* temp_str = malloc(sizeof(char) * strlen(p->pw_name) + 27);
    if(!temp_str) err(1, "Memory Error while allocating temporary directory for swap\n");
    snprintf(temp_str, strlen(p->pw_name) + 27, "/home/%s/Temp/reversi.XXXXXX", p->pw_name);

    char* dir_ptr = mkdtemp(temp_str), *result = malloc(sizeof(char) * (strlen(dir_ptr) + 1));
    memcpy(result, dir_ptr, strlen(dir_ptr) + 1);

    #ifdef checkpointdebug
        printf("Swapping to %s\n", dir_ptr);
    #endif

    free(temp_str);

    return result;
}

char* find_abs_path(size_t page_index, const char* swap_directory) {
    char* filename = malloc(sizeof(char) * 16), *abs_path = malloc(sizeof(char) * (17 + strlen(swap_directory)));
    if(!filename) err(1, "Memory Error while allocating filename for swap page\n");
    filename[15] = 0;
    snprintf(filename, 15, "p%lu.bin", page_index);

    #ifdef checkpointdebug
        printf("The checkpoint filename is %s\n", filename);
    #endif

    abs_path[0] = 0;
    abs_path = strcat(abs_path, swap_directory);
    abs_path[strlen(swap_directory)] = '/';
    abs_path[strlen(swap_directory) + 1] = 0;

    #ifdef checkpointdebug
        // abs_path[strlen(swap_directory) + 1] = 0;
        printf("Swap directory is %s\n", abs_path);
    #endif

    strcat(abs_path, filename);
    // abs_path[strlen(filename) + strlen(swap_directory) + 1] = 0;
    free(filename);

    #ifdef checkpointdebug
        printf("Saving index %ld to %s\n", page_index, abs_path);
    #endif

    return abs_path;
}

/**
 * @brief Saves a page from a mempage struct to disk
 * 
 * @param mp mempage to save from
 * @param page_index page to save
 * @param swap_directory directory where disk pages are being stored
 */
void save_mempage_page(mempage mp, size_t page_index, const char* swap_directory) {

    #ifdef swapdebug
        printf("Saving files for page %ld in %s\n", page_index, swap_directory);
    #endif

    char* abs_path = find_abs_path(page_index, swap_directory);

    /*
     * page_size (number of bins)
     * bin_counts (number of elements in each bin)
     * bin_contents (array of all elements)
     */

    FILE* fp = fopen(abs_path, "wb+");
    if(!fp) err(7, "Unable to open or create swap file\n");

    // Write page size
    fwrite(&mp->count_per_page, sizeof(mp->count_per_page), 1, fp);

    // Write the bin sizes
    fwrite(mp->bin_counts[page_index], sizeof(size_t), mp->count_per_page, fp);

    // Write the keys
    for(size_t b = 0; b < mp->count_per_page; b++) {
        fwrite(mp->pages[page_index][b], sizeof(__uint128_t), mp->bin_counts[page_index][b], fp);
        free(mp->pages[page_index][b]);
    }

    fclose(fp);
    free(abs_path);
    free(mp->bin_counts[page_index]);
    free(mp->pages[page_index]);

    // Mark the page as not present in RAM
    size_t byte = page_index >> 3, bit = page_index % 8;
    uint8_t ph = 1 << bit;
    mp->page_present[byte] ^= ph;

    // Reset the access counts
    for(size_t p = 0; p < mp->page_count; p++) mp->access_counts[p] = 0;
}

void load_mempage_page(mempage mp, size_t page_index, const char* swap_directory) {
    #ifdef swapdebug
        printf("Loading files for page %ld\n", page_index);
    #endif

    char* abs_path = find_abs_path(page_index, swap_directory);

    FILE* fp = fopen(abs_path, "rb");
    if(!fp) err(7, "Unable to open or create swap file\n");

    size_t page_size = 0;
    fread(&page_size, sizeof(page_size), 1, fp);

    size_t *sizes = malloc(sizeof(size_t) * page_size);
    if(!sizes) err(1, "Memory Error while allocating page from swap file\n");

    fread(sizes, sizeof(size_t), page_size, fp);
    mp->bin_counts[page_index] = sizes;

    __uint128_t** bins = malloc(sizeof(__uint128_t*) * page_size);
    if(!bins) err(1, "Memory Error while allocating page from swap file\n");
    mp->pages[page_index] = bins;

    for(size_t b = 0; b < page_size; b++) {
        __uint128_t* bin = malloc(sizeof(__uint128_t) * mp->bin_counts[page_index][b]);
        if(!bin) err(1, "Memory Error while allocating bin from swap file\n");
        fread(bin, sizeof(__uint128_t), mp->bin_counts[page_index][b], fp);
        bins[b] = bin;
    }


    fclose(fp);
    free(abs_path);

    // Mark the page as not present in RAM
    size_t byte = page_index >> 3, bit = page_index % 8;
    uint8_t ph = 1 << bit;
    mp->page_present[byte] |= ph;
}

/**
 * @brief Swaps a page from a mempage struct with a page that is in memory
 * 
 * @param mp mempage to swap from
 * @param spage_index page to save
 * @param rpage_index page to read
 * @param swap_directory the directory where swap file are stored
 */
void swap_mempage_page(mempage mp, size_t spage_index, size_t rpage_index, const char* swap_directory) {

    #ifdef swapdebug
        printf("Swapping files for indices %ld and %ld\n", spage_index, rpage_index);
    #endif

    save_mempage_page(mp, spage_index, swap_directory);
    load_mempage_page(mp, rpage_index, swap_directory);
}

#pragma endregion
#pragma region mempage_buff swapping

/**
 * @brief Saves a page from a mempage struct to disk
 * 
 * @param mp mempage to save from
 * @param page_index page to save
 * @param swap_directory directory where disk pages are being stored
 */
void save_mempage_buff_page(mempage_buff mp, size_t page_index, const char* swap_directory) {

    #ifdef swapdebug
        printf("Saving files for page %ld\n", page_index);
    #endif

    char* abs_path = find_abs_path(page_index, swap_directory);

    /*
     * page_size (number of bins)
     * bin_counts (number of elements in each bin)
     * bin_contents (array of all elements)
     */

    FILE* fp = fopen(abs_path, "wb+");
    if(!fp) err(7, "Unable to open or create swap file\n");

    // Write page size
    fwrite(&mp->count_per_page, sizeof(mp->count_per_page), 1, fp);

    // Write the keys
    fwrite(mp->pages[page_index], sizeof(__uint128_t), mp->count_per_page, fp);
    free(mp->pages[page_index]);

    fclose(fp);
    free(abs_path);
    free(mp->pages[page_index]);

    // Mark the page as not present in RAM
    size_t byte = page_index >> 3, bit = page_index % 8;
    uint8_t ph = 1 << bit;
    mp->page_present[byte] ^= ph;
}

void load_mempage_buff_page(mempage_buff mp, size_t page_index, const char* swap_directory) {

    #ifdef swapdebug
        printf("Loading files for page %ld\n", page_index);
    #endif

    char* abs_path = find_abs_path(page_index, swap_directory);

    FILE* fp = fopen(abs_path, "rb");
    if(!fp) err(7, "Unable to open or create swap file\n");

    size_t page_size = 0;
    fread(&page_size, sizeof(page_size), 1, fp);

    __uint128_t* bins = malloc(sizeof(__uint128_t*) * page_size);
    if(!bins) err(1, "Memory Error while allocating page from swap file\n");
    mp->pages[page_index] = bins;

    fread(bins, sizeof(__uint128_t), page_size, fp);

    fclose(fp);
    free(abs_path);

    // Mark the page as not present in RAM
    size_t byte = page_index >> 3, bit = page_index % 8;
    uint8_t ph = 1 << bit;
    mp->page_present[byte] |= ph;
}

/**
 * @brief Swaps a page from a mempage struct with a page that is in memory
 * 
 * @param mp mempage to swap from
 * @param spage_index page to save
 * @param rpage_index page to read
 * @param swap_directory the directory where swap file are stored
 */
void swap_mempage_buff_page(mempage_buff mp, size_t spage_index, size_t rpage_index, const char* swap_directory) {

    #ifdef swapdebug
        printf("Swapping files for indices %ld and %ld\n", spage_index, rpage_index);
    #endif

    save_mempage_buff_page(mp, spage_index, swap_directory);
    load_mempage_buff_page(mp, rpage_index, swap_directory);
}

#pragma endregion