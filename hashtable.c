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
hashtable create_hashtable(uint64_t initial_bin_count, uint64_t (*hash)(void*)) {
    assert(initial_bin_count);
    hashtable t = malloc(sizeof(hashtable_str));
    if(!t) err(1, "Memory Error while trying to allocate hashtable\n");
    t->bins = calloc(initial_bin_count, sizeof(linkedlist));
    for(uint64_t i = 0; i < initial_bin_count; i++) t->bins[i] = create_ll();
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
        destroy_ll(t->bins);
        free(t);
    }
}

/**
 * @brief Get all of the pairs of keys and values from the hashtable
 * 
 * @param t The hashtable to extract the pairs from
 * @return keyval_pair* An array of key-value pairs that must be free'd by the user.
 */
keyval_pair* get_pairs(hashtable t) {
    if(t) {
        linkedlist result = create_ll();
        for(uint64_t i = 0; i < t->bin_count; i++) {
            void** arr = ll_to_arr(t->bins[i]);
            for(uint64_t j = 0; j < t->bins[i]->size; j++) append_ll(result, arr[j]);
            free(arr);
        }

        void** resultarr = ll_to_arr(result);
        destroy_ll(result);
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
uint64_t put_hs(hashtable t, void* value) {
    if(t) {
        uint64_t k = t->hash(value);

        keyval_pair p = malloc(sizeof(keyval_pair_str));
        if(!p) err(1, "Memory Error while allocating key value pair\n");
        p->key = k;
        p->value = value;

        append_ll(t->bins[k % t->bin_count], p);
        
        if(++t->size > (t->bin_count << 15)) {
            //re-hash
            keyval_pair* pairs = get_pairs(t);

            t->bins = realloc(t->bins, t->size + 64);
            if(!t->bins) err(1, "Memory Error while re allocating bins for hashtable\n");
            for(uint64_t i = t->size; i < t->size + 64; i++) t->bins[i] = create_ll();
            t->bin_count = t->size + 64;

            for(keyval_pair* p = pairs; *p; p++) append_ll(t->bins[(*p)->key % t->bin_count], *p);
        }
    }
}

/**
 * @brief Retrieve a value from the hashtable using the given key
 * 
 * @param t The table to retrieve the value from
 * @param key The key to retrieve the value for
 * @return void* Returns the data that corresponds to the given key, or 0 if the key doesn't exist
 */
void* get_hs(hashtable t, uint64_t key) {
    if(t && t->size) {
        linkedlist bin = t->bins[key % t->bin_count];
        for(ll_node n = bin->head; n; n = n->next) {
            keyval_pair p = n->data;
            if(p->key == key) return p->value;
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
        uint64_t key = t->hash(value);
        linkedlist bin = t->bins[key % t->bin_count];
        for(ll_node n = bin->head; n; n = n->next) {
            keyval_pair p = n->data;
            if(p->key == key) return 1;
        }
    }
    return 0;
}