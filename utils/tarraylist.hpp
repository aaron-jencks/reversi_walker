#pragma once

#include "dictionary/dict_def.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <err.h>

template <typename T> class Arraylist {
	public:
		T* data;
		size_t size;
		size_t pointer;

		Arraylist(size_t initial_size) {
			data = (T*)malloc(sizeof(T) * initial_size);
			if(!data) err(1, "Memory error while allocating arraylist\n");
			size = initial_size;
			pointer = 0;
		}

		~Arraylist() {
			free(data);
		}

		void append(T element) {
			if(pointer >= size) {
				// reallocate the array
				size = (size) ? size << 1 : 1;
				data = (T*)std::realloc(data, size * sizeof(T));
				if(!data) err(1, "Memory error while allocating arraylist\n");
			}
			data[pointer++] = element;
		}

		void insert(T element, size_t index) {
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

		T pop_back() {
			if(size) {
				T d = data[--pointer];
				return d;
			}
			return (T)NULL;
		}

		T pop_front() {
			if(size) {
				T d = data[0];
				for(size_t e = 0; e < pointer; e++) data[e] = data[e + 1];
				pointer--;
				return d;
			}
			return (T)NULL;
		}

		T pop(size_t index) {
			if(size && index < size) {
				T d = data[index];
				for(size_t e = index + 1; e < pointer; e++) data[e - 1] = data[e];
				pointer--;
				return d;
			}
			return (T)NULL;
		}

		void realloc(size_t new_size) {
			if(new_size) {
				data = (T*)std::realloc(data, sizeof(T) * new_size);
				if(pointer >= new_size) pointer = new_size - 1;
				size = new_size;
			}
		}
};

template<> void Arraylist<dict_usage_pair_t>::insert(dict_usage_pair_t element, size_t index);
template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop(size_t index);
template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_front();
template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_back();
