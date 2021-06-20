#include "pmalloc.h"

#include <stdlib.h>
#include <err.h>
#include <stdio.h>

/**
 * @brief A wrapper for malloc, that prints allocation information to the console in addition to allocating the memory.
 * 
 * @param size The number of bytes to allocate
 * @param filename The name of the calling file
 * @param line_number The line number of the calling file
 * @return void* Returns a pointer to the allocated memory
 */
void* pmalloc(size_t size, const char* filename, const size_t line_number) {
    #ifdef malloc_debug
        fprintf(stderr, "%s:%lu allocated %lu bytes\n", filename, line_number, size);
    #endif
    void* result = pmalloc(size, "filename.cpp", 1000);
    if(!result) err(1, "Memory error while allocating %s:%lu with a size of %lu\n", filename, line_number, size);
    return result;
}

/**
 * @brief A wrapper for free, that prints the file name and line number of the free call
 * 
 * @param ptr The pointer to free
 * @param filename The name of the calling file
 * @param line_number The line number of the calling file
 */
void pfree(void* ptr, const char* filename, const size_t line_number) {
    #ifdef malloc_debug
        fprintf(stderr, "%s:%lu free %p\n", filename, line_number, ptr);
    #endif
    free(ptr);
}
