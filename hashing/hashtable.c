#include "hashtable.h"
#include "arraylist.h"

#include <stdlib.h>
#include <err.h>
#include <assert.h>
#include <stdio.h>

#define BIN_PAGE_COUNT 1000000
#define BIN_SIZE 16

/**
 * @brief Create a hashtable object
 * 
 * @param initial_bin_count The number of bins to start off with
 * @param hash The hash function for hashing the data into the hashtable
 * @return hashtable 
 */
hashtable create_hashtable(uint64_t initial_bin_count, __uint128_t (*hash)(void*)) {
    assert(initial_bin_count);
    hashtable t = malloc(sizeof(hashtable_str));
    if(!t) err(1, "Memory Error while trying to allocate hashtable\n");
    if(pthread_mutex_init(&t->table_lock, 0) != 0) err(4, "Hashtable mutex init has failed\n");
    t->bins = create_mempage(BIN_PAGE_COUNT, initial_bin_count, BIN_SIZE); // calloc(initial_bin_count, sizeof(uint128_arraylist));
    // for(uint64_t i = 0; i < initial_bin_count; i++) mempage_put(t->bins, i, create_uint128_arraylist(65));
    t->bin_count = initial_bin_count;
    t->size = 0;
    t->hash = hash;
    return t;
}

/**
 * @brief Destroys the hashtable object
 * 
 * @param t 
 */
void destroy_hashtable(hashtable t) {
    if(t) {
        pthread_mutex_destroy(&t->table_lock);
        destroy_mempage(t->bins);
        free(t);
    }
}

/**
 * @brief Get all of the pairs of keys and values from the hashtable
 * 
 * @param t The hashtable to extract the pairs from
 * @return mempage_buff An array of keys that must be free'd by the user.
 */
mempage_buff get_pairs(hashtable t) {
    if(t) {
        #ifdef hashdebug
            printf("Getting pairs for hashtable with size %lu %lu\n", ((uint64_t*)&t->size)[1], ((uint64_t*)&t->size)[0]);
        #endif

        mempage_buff buff = create_mempage_buff(t->size, BIN_PAGE_COUNT);
        __uint128_t offset = 0;
        for(size_t p = 0; p < t->bins->page_count; p++) {
            __uint128_t** page = t->bins->pages[p];
            for(size_t b = 0; b < t->bins->count_per_page; b++) {
                __uint128_t* bin = page[b];
                size_t bcount = t->bins->bin_counts[p][b];

                for(size_t be = 0; be < bcount; be++) {
                    if(!bin[be]) break;

                    #ifdef hashdebug
                        printf("\nRetrieving %lu %lu into index %lu %lu", ((uint64_t*)&bin[be])[1], ((uint64_t*)&bin[be])[0], ((uint64_t*)&offset)[1], ((uint64_t*)&offset)[0]);
                        fflush(stdout);
                    #endif
                    
                    mempage_buff_put(buff, offset++, bin[be]);
                }
            }
        }

        #ifdef hashdebug
            printf("\n");
        #endif

        return buff;
    }
}

/**
 * @brief Inserts a value into the hash table
 * 
 * @param t The table to insert the value into
 * @param value The value to insert
 * @return uint64_t Returns the value of the key that value hashed to
 */
__uint128_t put_hs(hashtable t, void* value) {
    if(t) {
        while(1) {
            if(!pthread_mutex_trylock(&t->table_lock)){
                __uint128_t k = t->hash(value), b = k % (t->bin_count);
                if(!k) err(2, "Hit a hash value that is 0\n");

                mempage_append_bin(t->bins, b, k);
                
                if(++t->size > (t->bin_count * 15)) {
                    #ifdef hashdebug
                        printf("Rehashing hashtable\n");
                    #endif

                    //re-hash

                    #ifdef hashdebug
                        printf("Copying over the previous elements into a temporary buffer\n");
                    #endif

                    mempage_buff buff = get_pairs(t);

                    #ifdef hashdebug
                        printf("Clearing mempage\n");
                    #endif

                    mempage_clear_all(t->bins);

                    __uint128_t new_size = t->size + 64000;

                    #ifdef hashdebug
                        printf("Reallocating mempage to %lu %lu elements\n", ((uint64_t*)&new_size)[1], ((uint64_t*)&new_size)[0]);
                    #endif

                    mempage_realloc(t->bins, new_size);

                    #ifdef hashdebug
                        printf("Reinserting previous elements\n");
                    #endif

                    for(__uint128_t p = 0; p < buff->num_element; p++) {
                        __uint128_t k = mempage_buff_get(buff, p);

                        #ifdef hashdebug
                            printf("\rInserting element %lu %lu: %lu %lu", ((uint64_t*)&p)[1], ((uint64_t*)&p)[0], ((uint64_t*)&k)[1], ((uint64_t*)&k)[0]);
                        #endif

                        mempage_append_bin(t->bins, k % t->bins->num_bins, k);
                    }

                    #ifdef hashdebug
                        printf("\nRehash complete, new bin count is %lu %lu\n", ((uint64_t*)&t->bins->num_bins)[1], ((uint64_t*)&t->bins->num_bins)[0]);
                    #endif

                    // t->bins->num_bins = new_size;
                    t->bin_count = new_size;

                    destroy_mempage_buff(buff);
                }

                pthread_mutex_unlock(&t->table_lock);

                return k;
            }
            sched_yield();
        }
    }

    return 0;
}

/**
 * @brief Checks if the given value exists in the hashtable
 * 
 * @param t The hashtable to check
 * @param value The value to check for
 * @return uint8_t Returns 1 if the value exists, and 0 otherwise
 */
uint8_t exists_hs(hashtable t, void* value) {
    if(t && t->size) {
        while(1) {
            while(pthread_mutex_trylock(&t->table_lock)) sched_yield();

            __uint128_t key = t->hash(value), b = key % t->bin_count;
            if(!key) err(2, "Hit a hash value that is 0\n");

            uint8_t res = mempage_value_in_bin(t->bins, b, key);

            pthread_mutex_unlock(&t->table_lock);

            return res;
        }
    }
    return 0;
}

void to_file_hs(FILE* fp, hashtable t) {
    if(t) {
        if(fwrite(&t->bin_count, sizeof(t->bin_count), 1, fp) < 1) err(10, "Failed to write bin count to file\n");
        fwrite(&t->size, sizeof(t->size), 1, fp);

        mempage_buff pairs = get_pairs(t);
        uint64_t count = 0;

        printf("Writing %lu %lu entries to disk\n", ((uint64_t*)&pairs->num_element)[1], ((uint64_t*)&pairs->num_element)[0]);

        for(__uint128_t p = 0; p < pairs->num_element; p++) {
            __uint128_t k = mempage_buff_get(pairs, p);
            size_t written = fwrite(&k, sizeof(__uint128_t), 1, fp);
            if(written < 1) err(10, "Failed to save hashtable key, only wrote %lu/%lu on entry %lu\n", written, sizeof(__uint128_t), count);
            count++;
        }

        printf("Wrote %ld entries\n", count);

        destroy_mempage_buff(pairs);

        __uint128_t spacer = 0;
        fwrite(&spacer, sizeof(__uint128_t), 1, fp);
    }
}

hashtable from_file_hs(FILE* fp, __uint128_t (*hash)(void*)) {
    __uint128_t bin_count, size;
    fread(&bin_count, sizeof(__uint128_t), 1, fp);
    fread(&size, sizeof(__uint128_t), 1, fp);
    // fscanf(fp, "%lu%lu", &bin_count, &size);

    mempage_buff keys = create_mempage_buff(size, BIN_PAGE_COUNT);
    __uint128_t bk;
    for(__uint128_t k = 0; k < size; k++) {
        printf("\rReading %lu %lu/%lu %lu", ((uint64_t*)&k)[1], ((uint64_t*)&k)[0],
                                            ((uint64_t*)&size)[1], ((uint64_t*)&size)[0]);
        if(fread(&bk, sizeof(__uint128_t), 1, fp) < 1) err(11, "Failed to read hashtable key %lu %lu/%lu %lu\n", ((uint64_t*)&k)[1], ((uint64_t*)&k)[0], 
                                                                                                                 ((uint64_t*)&size)[1], ((uint64_t*)&size)[0]);
        if(bk) mempage_buff_put(keys, k, bk);
        else break;
    }

    printf("\n");

    hashtable ht = create_hashtable(bin_count, hash);
    
    // Insert the keys
    for(__uint128_t p = 0; p < size; p++) {
        printf("\rInserting %lu %lu/%lu %lu", ((uint64_t*)&p)[1], ((uint64_t*)&p)[0],
                                              ((uint64_t*)&size)[1], ((uint64_t*)&size)[0]);
        fflush(stdout);
        __uint128_t k = mempage_buff_get(keys, p);
        // printf("Retrieved the file from the buffer\n");
        mempage_append_bin(ht->bins, k % ht->bin_count, k);
    }
    ht->size = size;

    printf("\nRead in a hashtable with %lu %lu entries and %lu %lu bins\n", ((uint64_t*)&size)[1], ((uint64_t*)&size)[0], 
                                                                            ((uint64_t*)&bin_count)[1], ((uint64_t*)&bin_count)[0]);

    destroy_mempage_buff(keys);

    return ht;
}