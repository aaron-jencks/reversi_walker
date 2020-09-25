#include "arraylist.h"

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
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
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
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
                l->data[i] = 0;
        }
        l->data[index] = data;
        l->pointer = index + 1;
    }
}

void* pop_back_pal(ptr_arraylist l) {
    if(l && l->size) {
        void* d = l->data[--(l->pointer)];
        return d;
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
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
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
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
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
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
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
            l->data = realloc(l->data, l->size);
            if(!l->data) err(1, "Memory Error while allocating ptr_arraylist\n");
            for(uint64_t i = l->pointer + 1; i < l->size; i++) 
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

#pragma endregion