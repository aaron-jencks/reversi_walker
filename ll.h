#pragma once

#include <stdint.h>

typedef struct _ll_node_str {
    struct _ll_node_str* next;
    struct _ll_node_str* previous;
    void* data;
} ll_node_str;

typedef ll_node_str* ll_node;

typedef struct _linkedlist_str {
    ll_node head;
    ll_node tail;
    uint64_t size;
} linkedlist_str;

/**
 * @brief Represents a linkedlist
 * 
 */
typedef linkedlist_str* linkedlist;

/**
 * @brief Create a ll object
 * 
 * @return linkedlist 
 */
linkedlist create_ll();

/**
 * @brief Destroys a linkedlist object
 * 
 * @param ll 
 */
void destroy_ll(linkedlist ll);

/**
 * @brief Appends data to the back of a linkedlist object
 * 
 * @param ll 
 * @param data 
 */
void append_ll(linkedlist ll, void* data);

/**
 * @brief Removes the first item from a linkedlist object
 * 
 * @param ll 
 * @return void* 
 */
void* pop_front_ll(linkedlist ll);

/**
 * @brief Returns the first item from a linkedlist object
 * 
 * @param ll 
 * @return void* 
 */
void* get_front_ll(linkedlist ll);

/**
 * @brief Removes the last item from a linkedlist object
 * 
 * @param ll 
 * @return void* 
 */
void* pop_back_ll(linkedlist ll);

/**
 * @brief Returns an array representing the information stored in the linkedlist
 * The array must be free'd by the user
 * 
 * @param ll 
 * @return void** 
 */
void** ll_to_arr(linkedlist ll);