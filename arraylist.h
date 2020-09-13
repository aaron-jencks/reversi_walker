#pragma once

#include <stdint.h>

typedef struct _arr_list_str {
    void** data;
    uint64_t size;
    uint64_t pointer;
} arraylist_str;

/**
 * @brief Represents an array list
 * 
 */
typedef arraylist_str* arraylist;

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
arraylist create_arraylist(uint64_t initial_capacity);

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_arraylist(arraylist l);

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_al(arraylist l, void* data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_al(arraylist l, uint64_t index, void* data);
