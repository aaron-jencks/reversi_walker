#include "ll.h"

#include <stdlib.h>
#include <err.h>

/**
 * @brief Create a ll object
 * 
 * @return linkedlist 
 */
linkedlist create_ll() {
    linkedlist l = calloc(1, sizeof(linkedlist_str));
    if(!l) err(1, "Memory Error while allocating linked list");
    return l;
}

/**
 * @brief Destroys a linkedlist object
 * 
 * @param ll 
 */
void destroy_ll(linkedlist ll) {
    if(ll) {
        while(ll->size) pop_front_ll(ll);
        free(ll);
    }
}

/**
 * @brief Appends data to the back of a linkedlist object
 * 
 * @param ll 
 * @param data 
 */
void append_ll(linkedlist ll, void* data) {
    if(ll) {
        ll_node n = malloc(sizeof(ll_node_str));
        if(!n) err(1, "Memory Error while allocating linked list node\n");
        n->data = data;
        if(ll->tail) ll->tail->next = n;
        n->previous = ll->tail;
        ll->tail = n;
        n->next = 0;
        if(!(ll->size++)) ll->head = n;
    }
}

/**
 * @brief Removes the first item from a linkedlist object
 * 
 * @param ll 
 * @return void* 
 */
void* pop_front_ll(linkedlist ll) {
    if(ll && ll->size) {
        void* d = ll->head->data;

        ll_node h = ll->head;

        if(!(h->next)) ll->tail = 0;
        ll->head = h->next;
        ll->size--;

        free(h);

        return d;
    }
    return 0;
}

/**
 * @brief Removes the last item from a linkedlist object
 * 
 * @param ll 
 * @return void* 
 */
void* pop_back_ll(linkedlist ll) {
    if(ll && ll->size) {
        void* d = ll->tail->data;

        ll_node t = ll->tail;

        if(!(t->previous)) ll->tail = 0;
        ll->tail = t->previous;
        ll->size--;

        free(t);

        return d;
    }
    return 0;
}

/**
 * @brief Returns an array representing the information stored in the linkedlist
 * The array must be free'd by the user
 * 
 * @param ll 
 * @return void** 
 */
void** ll_to_arr(linkedlist ll) {
    void** result = malloc(sizeof(void*) * (ll->size + 1));
    if(!result) err(1, "Memory Error while allocating array from linked list\n");
    size_t pointer = 0;
    for(ll_node n = ll->head; pointer < ll->size; n = n->next) result[pointer++] = n->data;
    result[pointer] = 0;
    return result;
}