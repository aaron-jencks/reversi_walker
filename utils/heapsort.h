#pragma once

#include <stddef.h>

#include "dict_def.h"

void min_heapify(size_t** arr, size_t n, size_t i);
void build_min_heap(size_t** arr, size_t n);
void heapsort(size_t** arr, size_t n);

void min_heapify_dict(dict_element_t* arr, size_t n, size_t i);
void build_min_heap_dict(dict_element_t* arr, size_t n);
void heapsort_dict(dict_element_t* arr, size_t n);