#include "heapsort.h"

#include <stdlib.h>
#include <err.h>

void swap_heap_elements(size_t** arr, size_t a, size_t b) {
    size_t* temp;
    temp = arr[a];
    arr[a] = arr[b];
    arr[b] = temp;
}

/**
 * parent = (i >> 1) - 1
 * left = (i + 1 << 1) - 1
 * right = i + 1 << 1
 */

void min_heapify(size_t** arr, size_t n, size_t i) {
    size_t l, r, smallest;

    while(1) {
        l = (i << 1) + 1;
        r = (i << 1) + 2;

        if(l < n && arr[l][0] < arr[i][0]) smallest = l;
        else smallest = i;
        if(r < n && arr[r][0] < arr[smallest][0]) smallest = r;
        if(smallest != i) {
            swap_heap_elements(arr, i, smallest);
            i = smallest;
        }
        else break;
    }
}

void build_min_heap(size_t** arr, size_t n) {
    for(size_t i = (n >> 1) - 1; i > 0; i--) min_heapify(arr, n, i);
    min_heapify(arr, n, 0);
}

void heapsort(size_t** arr, size_t n) {
    build_min_heap(arr, n);
    for(size_t i = n - 1; i > 0; i--) {
        swap_heap_elements(arr, i, 0);
        min_heapify(arr, i, 0);
    }
}

void swap_heap_elements_dict(dict_element_t* arr, size_t a, size_t b) {
    dict_element_t temp;
    temp = arr[a];
    arr[a] = arr[b];
    arr[b] = temp;
}

/**
 * parent = (i >> 1) - 1
 * left = (i + 1 << 1) - 1
 * right = i + 1 << 1
 */

void min_heapify_dict(dict_element_t* arr, size_t n, size_t i) {
    size_t l, r, smallest;

    while(1) {
        l = (i << 1) + 1;
        r = (i << 1) + 2;

        if(l < n && arr[l].pair.usage < arr[i].pair.usage) smallest = l;
        else smallest = i;
        if(r < n && arr[r].pair.usage < arr[smallest].pair.usage) smallest = r;
        if(smallest != i) {
            swap_heap_elements_dict(arr, i, smallest);
            i = smallest;
        }
        else break;
    }
}

void build_min_heap_dict(dict_element_t* arr, size_t n) {
    for(size_t i = (n >> 1) - 1; i > 0; i--) min_heapify_dict(arr, n, i);
    min_heapify_dict(arr, n, 0);
}

void heapsort_dict(dict_element_t* arr, size_t n) {
    build_min_heap_dict(arr, n);
    for(size_t i = n - 1; i > 0; i--) {
        swap_heap_elements_dict(arr, i, 0);
        min_heapify_dict(arr, i, 0);
    }
}

void heapify_dict_removal_order(dict_element_t* arr, size_t n, size_t i) {
    size_t l, r, smallest;

    while(1) {
        l = (i << 1) + 1;
        r = (i << 1) + 2;

        if(l < n && arr[l].bin < arr[i].bin) smallest = l;
        else if(l < n && arr[l].bin == arr[i].bin && arr[l].element < arr[i].element) smallest = l;
        else smallest = i;

        if(r < n && arr[r].bin < arr[smallest].bin) smallest = r;
        else if(r < n && arr[r].bin == arr[smallest].bin && arr[r].element < arr[smallest].element) smallest = r;

        if(smallest != i) {
            swap_heap_elements_dict(arr, i, smallest);
            i = smallest;
        }
        else break;
    }
}

void build_heap_dict_removal_order(dict_element_t* arr, size_t n) {
    for(size_t i = (n >> 1) - 1; i > 0; i--) heapify_dict_removal_order(arr, n, i);
    heapify_dict_removal_order(arr, n, 0);
}

void heapsort_dict_removal_order(dict_element_t* arr, size_t n) {
    build_heap_dict_removal_order(arr, n);
    for(size_t i = n - 1; i > 0; i--) {
        swap_heap_elements_dict(arr, i, 0);
        heapify_dict_removal_order(arr, i, 0);
    }
}