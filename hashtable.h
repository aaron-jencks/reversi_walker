#pragma once

#include "ll.h"

#include <stdint.h>

typedef struct _keyval_pair_str {
    uint64_t key;
    void* value;
} keyval_pair_str;

typedef keyval_pair_str* keyval_pair;

typedef struct _hashtable_str {
    linkedlist* bins;
    uint64_t bin_count;
    uint64_t size;
    __uint128_t (*hash)(void*);
} hashtable_str;

typedef hashtable_str* hashtable;

/**
 * @brief Create a hashtable object
 * 
 * @param initial_bin_count The number of bins to start off with
 * @param hash The hash function for hashing the data into the hashtable
 * @return hashtable 
 */
hashtable create_hashtable(uint64_t initial_bin_count, __uint128_t (*hash)(void*));

/**
 * @brief Destroys the hashtable object
 * 
 * @param t 
 */
void destroy_hashtable(hashtable t);

/**
 * @brief Get all of the pairs of keys and values from the hashtable
 * 
 * @param t The hashtable to extract the pairs from
 * @return __uint128_t* An array of keys that must be free'd by the user.
 */
__uint128_t* get_pairs(hashtable t);

/**
 * @brief Inserts a value into the hash table
 * 
 * @param t The table to insert the value into
 * @param value The value to insert
 * @return uint64_t Returns the value of the key that value hashed to
 */
__uint128_t put_hs(hashtable t, void* value);

/**
 * @brief Checks if the given value exists in the hashtable
 * 
 * @param t The hashtable to check
 * @param value The value to check for
 * @return uint8_t Returns 1 if the value exists, and 0 otherwise
 */
uint8_t exists_hs(hashtable t, void* value);