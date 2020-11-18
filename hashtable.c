#include "hashtable.h"
#include "arraylist.h"

#include <stdlib.h>
#include <err.h>
#include <assert.h>
#include <stdio.h>

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
    t->bins = create_mempage(1000000, initial_bin_count); // calloc(initial_bin_count, sizeof(uint128_arraylist));
    for(uint64_t i = 0; i < initial_bin_count; i++) mempage_put(t->bins, i, create_uint128_arraylist(16));
    t->bin_count = initial_bin_count;
    t->size = 0;
    t->hash = hash;
    return t;
}

void clear_bins(hashtable t) {
    for(__uint128_t l = 0; l < t->bin_count; l++) destroy_uint128_arraylist((uint128_arraylist)mempage_get(t->bins, l));
    destroy_mempage(t->bins);
}

/**
 * @brief Destroys the hashtable object
 * 
 * @param t 
 */
void destroy_hashtable(hashtable t) {
    if(t) {
        pthread_mutex_destroy(&t->table_lock);
        clear_bins(t);
        free(t);
    }
}

/**
 * @brief Get all of the pairs of keys and values from the hashtable
 * 
 * @param t The hashtable to extract the pairs from
 * @return __uint128_t* An array of keys that must be free'd by the user.
 */
__uint128_t* get_pairs(hashtable t) {
    if(t) {
        uint128_arraylist result = create_uint128_arraylist(t->size + 1);
        for(uint64_t i = 0; i < t->bin_count; i++) {
            uint128_arraylist alist = (uint128_arraylist)mempage_get(t->bins, i);
            __uint128_t* arr = alist->data;
            for(uint64_t j = 0; j < alist->pointer; j++) append_ddal(result, arr[j]);
            // free(arr);
        }

        __uint128_t* resultarr = result->data;
        free(result);
        return resultarr;
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

                append_ddal((uint128_arraylist)mempage_get(t->bins, b), k);
                
                if(++t->size > (t->bin_count * 15)) {
                    #ifdef hashdebug
                        printf("Rehashing hashtable\n");
                    #endif

                    //re-hash
                    __uint128_t* pairs = get_pairs(t);

                    clear_bins(t);

                    t->bins = create_mempage(1000000, t->size + 64);
                    // if(!t->bins) err(1, "Memory Error while re allocating bins for hashtable\n");
                    for(uint64_t i = 0; i < t->size + 64; i++) mempage_put(t->bins, i, create_uint128_arraylist(65));
                    t->bin_count = t->size + 64;

                    for(__uint128_t* p = pairs; *p; p++) append_ddal((uint128_arraylist)mempage_get(t->bins, *p % t->bin_count), *p);

                    free(pairs);
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

            uint128_arraylist bin = mempage_get(t->bins, b);
            for(uint32_t n = 0; n < bin->pointer; n++) {
                __uint128_t p = bin->data[n];
                if(p == key) {
                    pthread_mutex_unlock(&t->table_lock);
                    return 1;
                }
            }

            pthread_mutex_unlock(&t->table_lock);

            return 0;
        }
    }
    return 0;
}

void to_file_hs(FILE* fp, hashtable t) {
    if(t) {
        fwrite(&t->bin_count, sizeof(t->bin_count), 1, fp);
        fwrite(&t->size, sizeof(t->size), 1, fp);

        __uint128_t* pairs = get_pairs(t);

        for(__uint128_t* p = pairs; *p; p++) {
            fwrite(p, sizeof(__uint128_t), 1, fp);
        }

        free(pairs);

        __uint128_t spacer = 0;
        fwrite(&spacer, sizeof(__uint128_t), 1, fp);
    }
}

hashtable from_file_hs(FILE* fp, __uint128_t (*hash)(void*)) {
    uint64_t bin_count, size;
    fread(&bin_count, sizeof(uint64_t), 1, fp);
    fread(&size, sizeof(uint64_t), 1, fp);
    // fscanf(fp, "%lu%lu", &bin_count, &size);

    uint128_arraylist keys = create_uint128_arraylist(size + 1);
    __uint128_t bk;
    for(uint64_t k = 0; k < size; k++) {
        fread(&bk, sizeof(__uint128_t), 1, fp);
        if(bk) append_ddal(keys, bk);
        else break;
    }

    hashtable ht = create_hashtable(bin_count, hash);
    
    // Insert the keys
    for(__uint128_t* p = keys->data; *p; p++) append_ddal((uint128_arraylist)mempage_get(ht->bins, *p % ht->bin_count), *p);

    printf("Read in a hashtable with %lu entries and %lu bins\n", keys->pointer, bin_count);

    return ht;
}