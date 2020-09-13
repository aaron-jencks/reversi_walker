#include "arraylist.h"

#include <stdlib.h>
#include <err.h>

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
arraylist create_arraylist(uint64_t initial_capacity) {
    arraylist l = malloc(sizeof(arraylist_str));
    if(!l) err(1, "Memory Error while allocating arraylist\n");
    l->data = calloc(initial_capacity, sizeof(void*));
    if(!l->data) err(1, "Memory Error while allocating arraylist\n");
    l->pointer = 0;
    l->size = initial_capacity;
    return l;
}

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_arraylist(arraylist l) {
    if(l) {
        if(l->data) free(l->data);
        free(l);
    }
}

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_al(arraylist l, void* data) {
    if(l) {
        if(l->pointer >= l->size) {
            // re-allocate the array
            l->size = (l->size) ? l->size << 1 : 1;
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating arraylist\n");
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[l->pointer++] = data;
    }
}

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_al(arraylist l, uint64_t index, void* data) {
    if(l) {
        if(index >= l->size) {
            // re-allocate the array
            l->size = index + 65;
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating arraylist\n");
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[index] = data;
        l->pointer = index + 1;
    }
}