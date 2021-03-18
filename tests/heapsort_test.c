#include "heapsort_test.h"
#include "../utils/heapsort.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

void display_array(size_t *arr, size_t n) {
    printf("[");
    for(size_t e = 0; e < n; e++) printf("%lu%s", arr[e], (e == n - 1) ? "]\n" : ", ");
}

void heapsort_test() {
    size_t arr[] = {5, 3, 9, 11, 2, 12, 14}, partial_arr[] = {5, 3, 9, 11, 2, 12, 14}, 
        sorted_key[] = {2, 3, 5, 9, 11, 12, 14}, partial_sorted_key[] = {3, 5, 9, 11}, count = 7;
    heapsort_int(arr, count);
    display_array(arr, count);
    for(uint8_t e = 0; e < count; e++) assert(arr[e] == sorted_key[e]);
    heapsort_int(partial_arr, 4);
    display_array(partial_arr, 4);
    for(uint8_t e = 0; e < 4; e++) assert(partial_arr[e] == partial_sorted_key[e]);
}