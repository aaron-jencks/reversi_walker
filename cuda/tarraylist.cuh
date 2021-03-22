#pragma once

#include "../gameplay/reversi_defs.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <err.h>

template <typename T> class Arraylist {
	public:
		T* data;
		size_t size;
		size_t pointer;

		__device__ __host__ Arraylist(size_t initial_size) {
			data = (T*)malloc(sizeof(T) * initial_size);
			if(!data) err(1, "Memory error while allocating arraylist\n");
			size = initial_size;
			pointer = 0;
		}

		__device__ __host__ ~Arraylist() {
			free(data);
		}

		__device__ __host__ void append(T element) {
			if(pointer >= size) {
				// reallocate the array
				size = (size) ? size << 1 : 1;
				data = (T*)std::realloc(data, size * sizeof(T));
				if(!data) err(1, "Memory error while allocating arraylist\n");
			}
			data[pointer++] = element;
		}

		__host__ void insert(T element, size_t index) {
			if(index >= size) {
					// reallocate the array
					size = (size) ? size << 1 : 1;
					data = (T*)std::realloc(data, size * sizeof(T));
					if(!data) err(1, "Memory error while allocating arraylist");
					for(size_t e = pointer; e < index; e++) { data[e] = (T)NULL; }
					pointer = index + 1;
			}
			else {
				if(pointer + 1 >= size) {
					size = (size) ? size << 1 : 1;
					data = (T*)std::realloc(data, size * sizeof(T));
					if(!data) err(1, "Memory error while allocating arraylist");
				}

				for(size_t e = index; e < pointer; e++) { data[e + 1] = data[e]; }
				data[index] = element;
				pointer++;
			}
		}

		__device__ __host__ T pop_back() {
			if(size) {
				T d = data[--pointer];
				return d;
			}
			return (T)NULL;
		}

		__host__ T pop_front() {
			if(size) {
				T d = data[0];
				for(size_t e = 0; e < pointer; e++) data[e] = data[e + 1];
				pointer--;
				return d;
			}
			return (T)NULL;
		}

		__host__ T pop(size_t index) {
			if(size && index < size) {
				T d = data[index];
				for(size_t e = index + 1; e < pointer; e++) data[e - 1] = data[e];
				pointer--;
				return d;
			}
			return (T)NULL;
		}

		__device__ __host__ void realloc(size_t new_size) {
			if(new_size) {
				data = (T*)std::realloc(data, sizeof(T) * new_size);
				if(pointer >= new_size) pointer = new_size - 1;
				size = new_size;
			}
		}
};

__device__ __host__ template<> void Arraylist<board_str>::append(board_str element);
__host__ template<> void Arraylist<board_str>::insert(board_str element, size_t index);
__host__ template<> board_str Arraylist<board_str>::pop(size_t index);
__host__ template<> board_str Arraylist<board_str>::pop_front();
__device__ __host__ template<> board_str Arraylist<board_str>::pop_back();

