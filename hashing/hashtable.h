#pragma once

#include "../mem_man/mempage.h"

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

typedef struct _keyval_pair_str {
    uint64_t key;
    void* value;
} keyval_pair_str;

typedef keyval_pair_str* keyval_pair;

typedef struct _hashtable_str {
    pthread_mutex_t table_lock;
    mempage bins;
    __uint128_t bin_count;
    __uint128_t size;
} hashtable_str;

typedef hashtable_str* hashtable;

/**
 * @brief Create a hashtable object
 * 
 * @param initial_bin_count The number of bins to start off with
 * @param hash The hash function for hashing the data into the hashtable
 * @return hashtable 
 */
hashtable create_hashtable(uint64_t initial_bin_count);

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
 * @return mempage_buff An array of keys that must be free'd by the user.
 */
mempage_buff get_pairs(hashtable t);

/**
 * @brief Inserts a value into the hash table
 * 
 * @param t The table to insert the value into
 * @param value The value to insert
 * @return uint64_t Returns the value of the key that value hashed to
 */
void put_hs(hashtable t, __uint128_t k, void* value);

/**
 * @brief Get the pointer mapped to the given key, returns 0 if it doesn't exist.
 * 
 * @param t 
 * @param k 
 * @return void* 
 */
void* get_hs(hashtable t, __uint128_t k);

/**
 * @brief Checks if the given value exists in the hashtable
 * 
 * @param t The hashtable to check
 * @param value The value to check for
 * @return uint8_t Returns 1 if the value exists, and 0 otherwise
 */
uint8_t exists_hs(hashtable t, __uint128_t k);

#pragma region file io

/**
 * @brief Converts the hashtable into a byte string and then appends it to the file given
 * 
 * @param t The hashtable to convert
 */
void to_file_hs(FILE* fp, hashtable t);

/**
 * @brief Reads in a hashtable from the byte string starting from the current position in the file given
 * 
 * @param fp The file pointer to read from
 * @param hash The hash function to use, because that can't be saved
 * @return hashtable Returns a hashtable containing the keys hashed in the given string
 */
hashtable from_file_hs(FILE* fp, __uint128_t (*hash)(void*));

#pragma endregion