#pragma once

#include <stdint.h>
#include <stddef.h>

#define INITIAL_BIN_COUNT 1000000
#define INITIAL_BIN_SIZE 15
#define DICT_LOAD_LIMIT 15
#define BIN_PAGE_COUNT 1000000

typedef struct _dict_pair_t {
    __uint128_t key;
    uint8_t* value;
} dict_pair_t;

typedef struct _dict_usage_pair_t {
    dict_pair_t pair;
    size_t usage;
} dict_usage_pair_t;

typedef struct _dict_element_t {
    dict_usage_pair_t pair;
    size_t bin;
    size_t element;
} dict_element_t;