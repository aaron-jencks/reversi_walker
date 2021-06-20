#pragma once

#include <stddef.h>

/**
 * @brief A wrapper for malloc, that prints allocation information to the console in addition to allocating the memory.
 * 
 * @param size The number of bytes to allocate
 * @param filename The name of the calling file
 * @param line_number The line number of the calling file
 * @return void* Returns a pointer to the allocated memory
 */
void* pmalloc(size_t size, const char* filename, const size_t line_number);

/**
 * @brief A wrapper for free, that prints the file name and line number of the free call
 * 
 * @param ptr The pointer to free
 * @param filename The name of the calling file
 * @param line_number The line number of the calling
 */
void pfree(void* ptr, const char* filename, const size_t line_number);
