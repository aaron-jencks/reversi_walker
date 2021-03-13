#pragma once

#include <stddef.h>

#include "./dictionary/dict_def.h"

#ifdef __cplusplus
extern "C" {
#endif

void min_heapify(size_t** arr, size_t n, size_t i);
void build_min_heap(size_t** arr, size_t n);
void heapsort(size_t** arr, size_t n);

void min_heapify_dict(dict_element_t* arr, size_t n, size_t i);
void build_min_heap_dict(dict_element_t* arr, size_t n);
void heapsort_dict(dict_element_t* arr, size_t n);

void heapify_dict_removal_order(dict_element_t* arr, size_t n, size_t i);
void build_heap_dict_removal_order(dict_element_t* arr, size_t n);
void heapsort_dict_removal_order(dict_element_t* arr, size_t n);

#ifdef __cplusplus
}
#endif