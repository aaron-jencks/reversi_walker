#include "hashtable.h"

#include <stdlib.h>
#include <err.h>
#include <assert.h>

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
    t->bins = calloc(initial_bin_count, sizeof(uint128_arraylist));
    for(uint64_t i = 0; i < initial_bin_count; i++) t->bins[i] = create_uint128_arraylist(65);
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
        for(uint128_arraylist* l = t->bins; *l; l++) destroy_uint128_arraylist(*l);
        free(t->bins);
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
            __uint128_t* arr = t->bins[i]->data;
            for(uint64_t j = 0; j < t->bins[i]->pointer; j++) append_ddal(result, arr[j]);
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
        __uint128_t k = t->hash(value);
        if(!k) err(2, "Hit a hash value that is 0\n");

        append_ddal(t->bins[k % t->bin_count], k);
        
        if(++t->size > (t->bin_count << 15)) {
            //re-hash
            __uint128_t* pairs = get_pairs(t);

            t->bins = realloc(t->bins, t->size + 64);
            if(!t->bins) err(1, "Memory Error while re allocating bins for hashtable\n");
            for(uint64_t i = t->size; i < t->size + 64; i++) t->bins[i] = create_uint128_arraylist(65);
            t->bin_count = t->size + 64;

            for(__uint128_t* p = pairs; *p; p++) append_ddal(t->bins[*p % t->bin_count], *p);
        }

        return k;
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
        __uint128_t key = t->hash(value);
        if(!key) err(2, "Hit a hash value that is 0\n");

        uint128_arraylist bin = t->bins[key % t->bin_count];
        for(__uint128_t* n = bin->data; *n; n++) {
            __uint128_t p = *n;
            if(p == key) return 1;
        }
    }
    return 0;
}