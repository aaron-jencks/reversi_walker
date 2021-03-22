#include "tarraylist.cuh"
#include "../gameplay/reversi.h"

__device__ template<> void Arraylist<board_str>::append(board_str element) {
    if(pointer >= size) {
        // reallocate the array
        size = (size) ? size << 1 : 1;
        data = (board_str*)std::realloc(data, size * sizeof(T));
        if(!data) err(1, "Memory error while allocating arraylist\n");
    }
    clone_into_board_cuda(&data[pointer++], &element);
}

__host__ template<> void Arraylist<board_str>::append(board_str element) {
    if(pointer >= size) {
        // reallocate the array
        size = (size) ? size << 1 : 1;
        data = (T*)std::realloc(data, size * sizeof(T));
        if(!data) err(1, "Memory error while allocating arraylist\n");
    }
    clone_into_board(&data[pointer++], &element);
}

__host__ template<> void Arraylist<board_str>::insert(board_str element, size_t index) {
    if(index >= size) {
        // reallocate the array
        size = (size) ? size << 1 : 1;
        data = (board_str*)std::realloc(data, size * sizeof(T));
        if(!data) err(1, "Memory error while allocating arraylist");
        pointer = index + 1;
    }
    else {
        if(pointer + 1 >= size) {
            size = (size) ? size << 1 : 1;
            data = (board_str*)std::realloc(data, size * sizeof(T));
            if(!data) err(1, "Memory error while allocating arraylist");
        }

        for(size_t e = index; e < pointer; e++) { clone_into_board(&data[e + 1], &data[e]); }
        clone_into_board(&data[index], &element);
        pointer++;
    }
}

__host__ template<> board_str Arraylist<board_str>::pop(size_t index) {
    if(size && index < size) {
        board_str d = data[index];
        for(size_t e = index + 1; e < pointer; e++) clone_into_board(&data[e + 1], &data[e]);
        pointer--;
        return d;
    }
    return (board_str){0, 0, 0, 0};
}

__host__ template<> board_str Arraylist<board_str>::pop_front() {
    if(size) {
        board_str d = data[0];
        for(size_t e = 0; e < pointer; e++) clone_into_board(&data[e], &data[e + 1]);
        pointer--;
        return d;
    }
    return (board_str){0, 0, 0, 0};
}

__device__ __host__ template<> board_str Arraylist<board_str>::pop_back() {
    if(size) {
        board_str d = data[--pointer];
        return d;
    }
    return (board_str){0, 0, 0, 0};
}