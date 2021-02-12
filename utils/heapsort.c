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
        if(r < n && arr[r][0] < arr[i][0]) smallest = r;
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