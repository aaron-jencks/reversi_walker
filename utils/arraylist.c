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