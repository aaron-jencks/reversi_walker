#pragma once

#include <stdint.h>

#pragma region structs and typedefs

typedef struct _ptr_arr_list_str {
    void** data;
    uint64_t size;
    uint64_t pointer;
} ptr_arraylist_str;

/**
 * @brief Represents an array list of pointers
 * 
 */
typedef ptr_arraylist_str* ptr_arraylist;

typedef struct _uint8_arr_list_str {
    uint8_t* data;
    uint64_t size;
    uint64_t pointer;
} uint8_arraylist_str;

/**
 * @brief Represents an array list of bytes
 * 
 */
typedef uint8_arraylist_str* uint8_arraylist;

typedef struct _uint16_arr_list_str {
    uint16_t* data;
    uint64_t size;
    uint64_t pointer;
} uint16_arraylist_str;

/**
 * @brief Represents an array list of shorts
 * 
 */
typedef uint16_arraylist_str* uint16_arraylist;

typedef struct _uint64_arr_list_str {
    uint64_t* data;
    uint64_t size;
    uint64_t pointer;
} uint64_arraylist_str;

/**
 * @brief Represents an array list of 64-bit longs
 * 
 */
typedef uint64_arraylist_str* uint64_arraylist;

typedef struct _uint128_arr_list_str {
    __uint128_t* data;
    uint64_t size;
    uint64_t pointer;
} uint128_arraylist_str;

/**
 * @brief Represents an array list of 128-bit longs
 * 
 */
typedef uint128_arraylist_str* uint128_arraylist;

#pragma endregion

#pragma region pointer

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
ptr_arraylist create_ptr_arraylist(uint64_t initial_capacity);

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_ptr_arraylist(ptr_arraylist l);

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_pal(ptr_arraylist l, void* data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_pal(ptr_arraylist l, uint64_t index, void* data);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
void* pop_back_pal(ptr_arraylist l);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
void* pop_front_pal(ptr_arraylist l);

#pragma endregion

#pragma region uint8

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
uint8_arraylist create_uint8_arraylist(uint64_t initial_capacity);

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_uint8_arraylist(uint8_arraylist l);

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_cal(uint8_arraylist l, uint8_t data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_cal(uint8_arraylist l, uint64_t index, uint8_t data);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
uint8_t pop_back_cal(uint8_arraylist l);

#pragma endregion

#pragma region uint16

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
uint16_arraylist create_uint16_arraylist(uint64_t initial_capacity);

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_uint16_arraylist(uint16_arraylist l);

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_sal(uint16_arraylist l, uint16_t data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_sal(uint16_arraylist l, uint64_t index, uint16_t data);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
uint16_t pop_back_sal(uint16_arraylist l);

#pragma endregion

#pragma region uint64

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
uint64_arraylist create_uint64_arraylist(uint64_t initial_capacity);

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_uint64_arraylist(uint64_arraylist l);

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_dal(uint64_arraylist l, uint64_t data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_dal(uint64_arraylist l, uint64_t index, uint64_t data);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
uint64_t pop_back_dal(uint64_arraylist l);

#pragma endregion

#pragma region uint128

/**
 * @brief Create a arraylist object
 * 
 * @param initial_capacity The initial maximum capacity for this list,
 * the capacity is automatically adjusted once it runs out of room, 
 * be careful not to overflow a 64 bit int though.
 * @return arraylist 
 */
uint128_arraylist create_uint128_arraylist(uint64_t initial_capacity);

/**
 * @brief Destroys an arraylist
 * 
 * @param l 
 */
void destroy_uint128_arraylist(uint128_arraylist l);

/**
 * @brief Appends an item to the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 */
void append_ddal(uint128_arraylist l, __uint128_t data);

/**
 * @brief Inserts an item into the array list, re allocating the array if necessary
 * 
 * @param l 
 * @param data Data to insert
 * @param index The position to insert this data at
 */
void insert_ddal(uint128_arraylist l, uint64_t index, __uint128_t data);

/**
 * @brief Removes the last element from the arraylist, if there is one
 * 
 * @param l 
 * @return void* Returns the last element of the arraylist, or 0 if there isn't one
 */
__uint128_t pop_back_ddal(uint128_arraylist l);

#pragma endregion