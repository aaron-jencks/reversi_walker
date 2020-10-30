#pragma once

#include <stddef.h>

#include "arraylist.h"

typedef struct _mempage_str {
    ptr_arraylist pages;
    size_t count_per_page;
    __uint128_t num_elements;
} mempage_str;

typedef mempage_str* mempage;

mempage create_mempage(size_t page_max, __uint128_t elements);

void destroy_mempage(mempage mp);

void* mempage_get(mempage mp, __uint128_t index);

void mempage_put(mempage mp, __uint128_t index, void* data);