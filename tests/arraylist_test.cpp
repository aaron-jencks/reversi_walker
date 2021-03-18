#include "arraylist_test.hpp"
#include "../utils/tarraylist.hpp"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void arr_deletion_test() {
    printf("Testing arraylist deletion\n");
    Arraylist<int> arr(10);
    assert(!arr.pointer);
    for(size_t i = 0; i < 9; i++) arr.append(i);
    assert(arr.pointer == 9);
    arr.pop(5);
    assert(arr.pointer == 8);
}

void arr_insertion_test() {
    printf("Testing arraylist insertion\n");
    Arraylist<int> arr(10);
    assert(!arr.pointer);
    for(size_t i = 0; i < 9; i++) arr.append(i);
    assert(arr.pointer == 9);
    arr.insert(1, 5);
    assert(arr.pointer == 10);
}