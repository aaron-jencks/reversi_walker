#include "arraylist.hpp"

#include <stdlib.h>
#include <err.h>

#pragma region pointer

ptr_arraylist create_ptr_arraylist(uint64_t initial_capacity) {
    ptr_arraylist l = malloc(sizeof(ptr_arraylist_str));
    if(!l) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->data = calloc(initial_capacity, sizeof(void*));
    if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->pointer = 0;
    l->size = initial_capacity;
    return l;
}

void destroy_ptr_arraylist(ptr_arraylist l) {
    if(l) {
        if(l->data) free(l->data);
        free(l);
    }
}

void append_pal(ptr_arraylist l, void* data) {
    if(l) {
        if(l->pointer >= l->size) {
            // re-allocate the array
            l->size = (l->size) ? l->size << 1 : 1;
            l->data = realloc(l->data, l->size * sizeof(void*));
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[l->pointer++] = data;
    }
}

void insert_pal(ptr_arraylist l, uint64_t index, void* data) {
    if(l) {
        if(index >= l->size) {
            // re-allocate the array
            l->size = index + 65;
            l->data = realloc(l->data, l->size * sizeof(void*));
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[index] = data;
        l->pointer = index + 1;
    }
}

void* pop_back_pal(ptr_arraylist l) {
    if(l && l->size) {
        void* d = l->data[--(l->pointer)];
        l->data[l->pointer] = 0;
        return d;
    }
    return 0;
}

void* pop_front_pal(ptr_arraylist l) {
    if(l && l->size) {
        void* d = l->data[0];
        for(uint64_t i = 1; i < l->pointer; i++) l->data[i - 1] = l->data[i];
        l->data[--(l->pointer)] = 0;
        return d;
    }
    return 0;
}

void realloc_pal(ptr_arraylist l, size_t size) {
    if(l && size) {
        l->data = realloc(l->data, sizeof(void*) * size);
        if(l->pointer >= size) l->pointer = size - 1;
        l->size = size;
    } 
}

void* pop_pal(ptr_arraylist l, size_t index) {
    if(l && l->size) {
        if(index <= l->pointer) {
            void* v = l->data[index];
            l->pointer--;
            for(size_t e = index; e < l->pointer; e++) l->data[e] = l->data[e + 1];
            return v;
        }
    }
    return 0;
}

#pragma endregion

#pragma region uint8

uint8_arraylist create_uint8_arraylist(uint64_t initial_capacity) {
    uint8_arraylist l = malloc(sizeof(uint8_arraylist_str));
    if(!l) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->data = calloc(initial_capacity, sizeof(uint8_t));
    if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->pointer = 0;
    l->size = initial_capacity;
    return l;
}

void destroy_uint8_arraylist(uint8_arraylist l) {
    if(l) {
        if(l->data) free(l->data);
        free(l);
    }
}

void append_cal(uint8_arraylist l, uint8_t data) {
    if(l) {
        if(l->pointer >= l->size) {
            // re-allocate the array
            l->size = (l->size) ? l->size << 1 : 1;
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[l->pointer++] = data;
    }
}

void insert_cal(uint8_arraylist l, uint64_t index, uint8_t data) {
    if(l) {
        if(index >= l->size) {
            // re-allocate the array
            l->size = index + 65;
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[index] = data;
        l->pointer = index + 1;
    }
}

uint8_t pop_back_cal(uint8_arraylist l) {
    if(l && l->size) {
        uint8_t d = l->data[--(l->pointer)];
        return d;
    }
    return 0;
}

#pragma endregion

#pragma region uint16

uint16_arraylist create_uint16_arraylist(uint64_t initial_capacity) {
    uint16_arraylist l = malloc(sizeof(uint16_arraylist_str));
    if(!l) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->data = calloc(initial_capacity, sizeof(uint16_t));
    if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->pointer = 0;
    l->size = initial_capacity;
    return l;
}

void destroy_uint16_arraylist(uint16_arraylist l) {
    if(l) {
        if(l->data) free(l->data);
        free(l);
    }
}

void append_sal(uint16_arraylist l, uint16_t data) {
    if(l) {
        if(l->pointer >= l->size) {
            // re-allocate the array
            l->size = (l->size) ? l->size << 1 : 1;
            l->data = realloc(l->data, l->size << 1);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[l->pointer++] = data;
    }
}

void insert_sal(uint16_arraylist l, uint64_t index, uint16_t data) {
    if(l) {
        if(index >= l->size) {
            // re-allocate the array
            l->size = index + 65;
            l->data = realloc(l->data, l->size << 1);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[index] = data;
        l->pointer = index + 1;
    }
}

uint16_t pop_back_sal(uint16_arraylist l) {
    if(l && l->size) {
        uint16_t d = l->data[--(l->pointer)];
        return d;
    }
    return 0;
}

#pragma endregion

#pragma region double

uint64_arraylist create_uint64_arraylist(uint64_t initial_capacity) {
    uint64_arraylist l = malloc(sizeof(uint64_arraylist_str));
    if(!l) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->data = calloc(initial_capacity, sizeof(uint64_t));
    if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->pointer = 0;
    l->size = initial_capacity;
    return l;
}

void destroy_uint64_arraylist(uint64_arraylist l) {
    if(l) {
        if(l->data) free(l->data);
        free(l);
    }
}

void append_dal(uint64_arraylist l, uint64_t data) {
    if(l) {
        if(l->pointer >= l->size) {
            // re-allocate the array
            l->size = (l->size) ? l->size << 1 : 1;
            l->data = realloc(l->data, l->size << 3);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[l->pointer++] = data;
    }
}

void insert_dal(uint64_arraylist l, uint64_t index, uint64_t data) {
    if(l) {
        if(index >= l->size) {
            // re-allocate the array
            l->size = index + 65;
            l->data = realloc(l->data, l->size << 3);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[index] = data;
        l->pointer = index + 1;
    }
}

uint64_t pop_back_dal(uint64_arraylist l) {
    if(l && l->size) {
        uint64_t d = l->data[--(l->pointer)];
        return d;
    }
    return 0;
}

void realloc_dal(uint64_arraylist l, size_t size) {
    if(l && size) {
        l->data = realloc(l->data, sizeof(uint64_t) * size);
        if(l->pointer >= size) l->pointer = size - 1;
        l->size = size;
    } 
}

uint64_t pop_dal(uint64_arraylist l, size_t index) {
    if(l && l->size) {
        if(index <= l->pointer) {
            void* v = l->data[index];
            l->pointer--;
            for(size_t e = index; e < l->pointer; e++) l->data[e] = l->data[e + 1];
            return v;
        }
    }
    return 0;
}

#pragma endregion

#pragma region double double

uint128_arraylist create_uint128_arraylist(uint64_t initial_capacity) {
    uint128_arraylist l = malloc(sizeof(uint128_arraylist_str));
    if(!l) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->data = calloc(initial_capacity, sizeof(__uint128_t));
    if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->pointer = 0;
    l->size = initial_capacity;
    return l;
}

void destroy_uint128_arraylist(uint128_arraylist l) {
    if(l) {
        if(l->data) free(l->data);
        free(l);
    }
}

void append_ddal(uint128_arraylist l, __uint128_t data) {
    if(l) {
        if(l->pointer >= l->size) {
            // re-allocate the array
            l->size = (l->size) ? l->size << 1 : 1;
            l->data = realloc(l->data, l->size << 4);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[l->pointer++] = data;
    }
}

void insert_ddal(uint128_arraylist l, uint64_t index, __uint128_t data) {
    if(l) {
        if(index >= l->size) {
            // re-allocate the array
            l->size = index + 65;
            l->data = realloc(l->data, l->size << 4);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[index] = data;
        l->pointer = index + 1;
    }
}

__uint128_t pop_back_ddal(uint128_arraylist l) {
    if(l && l->size) {
        uint64_t d = l->data[--(l->pointer)];
        return d;
    }
    return 0;
}

void realloc_ddal(uint128_arraylist l, size_t size) {
    if(l && size) {
        l->data = realloc(l->data, sizeof(__uint128_t) * size);
        if(l->pointer >= size) l->pointer = size - 1;
        l->size = size;
    } 
}

__uint128_t pop_ddal(uint128_arraylist l, size_t index) {
    if(l && l->size) {
        if(index <= l->pointer) {
            void* v = l->data[index];
            l->pointer--;
            for(size_t e = index; e < l->pointer; e++) l->data[e] = l->data[e + 1];
            return v;
        }
    }
    return 0;
}

#pragma endregion

#pragma region dicts

#pragma region udict

// TODO implement

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
udict_arraylist create_udict_arraylist(uint64_t initial_capacity) {
    udict_arraylist l = malloc(sizeof(udict_arraylist_str));
    if(!l) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->data = calloc(initial_capacity, sizeof(dict_usage_pair_t));
    if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
    l->pointer = 0;
    l->size = initial_capacity;
    return l;
}

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_udict_arraylist(udict_arraylist l) {
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
void append_udal(udict_arraylist l, dict_usage_pair_t data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_udal(udict_arraylist l, uint64_t index, dict_usage_pair_t data);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
dict_usage_pair_t pop_back_udal(udict_arraylist l);

/**
 * @brief Reallocates the arraylist to a given size, 
 * if the new size is smaller than the last one, data may be lost.
 * 
 * @param l 
 * @param size The new size for the arraylist, in elements.
 */
void realloc_udal(udict_arraylist l, size_t size);

/**
 * @brief Removes an element from the arraylist
 * 
 * @param l 
 * @param index 
 * @return void* Returns the removed element
 */
dict_usage_pair_t pop_udal(udict_arraylist l, size_t index);

#pragma endregion

#pragma region dict

// TODO implement

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
dict_arraylist create_dict_arraylist(uint64_t initial_capacity);

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_dict_arraylist(dict_arraylist l);

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_dtdal(dict_arraylist l, dict_pair_t data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_dtdal(dict_arraylist l, uint64_t index, dict_pair_t data);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
dict_pair_t pop_back_dtdal(dict_arraylist l);

/**
 * @brief Reallocates the arraylist to a given size, 
 * if the new size is smaller than the last one, data may be lost.
 * 
 * @param l 
 * @param size The new size for the arraylist, in elements.
 */
void realloc_dtdal(dict_arraylist l, size_t size);

/**
 * @brief Removes an element from the arraylist
 * 
 * @param l 
 * @param index 
 * @return void* Returns the removed element
 */
dict_pair_t pop_dtdal(dict_arraylist l, size_t index);

#pragma endregion

#pragma endregion