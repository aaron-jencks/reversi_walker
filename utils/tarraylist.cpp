#include "tarraylist.hpp"
#include "./dictionary/dict_def.h"

#include <stdlib.h>
#include <err.h>

template <typename T> Arraylist<T>::Arraylist(size_t initial_size) {
	data = (T*)malloc(sizeof(T) * initial_size);
	size = initial_size;
	pointer = 0;
}

template <typename T> Arraylist<T>::~Arraylist() {
	free(data);
}

template <typename T> void Arraylist<T>::append(T element) {
	if(pointer >= size) {
		// reallocate the array
		size = (size) ? size << 1 : 1;
		data = (T*)std::realloc(data, size * sizeof(T));
		if(!data) err(1, "Memory error while allocating arraylist");
	}
	data[pointer++] = element;
}

template <typename T> void Arraylist<T>::insert(T element, size_t index) {
	if(index >= size) {
                // reallocate the array
                size = (size) ? size << 1 : 1;
                data = (T*)std::realloc(data, size * sizeof(T));
                if(!data) err(1, "Memory error while allocating arraylist");
        }
	data[index] = element;
	pointer = index + 1;
}

template <typename T> T Arraylist<T>::pop_back() {
	if(size) {
		T d = data[--pointer];
		return d;
	}
	return (T)NULL;
}

template <typename T> T Arraylist<T>::pop_front() {
	if(size) {
		T d = data[--pointer];
		return d;
	}
	return (T)NULL;
}

template <typename T> T Arraylist<T>::pop(size_t index) {
	if(size && index < size) {
		T d = data[index];
		for(size_t e = index + 1; e < pointer; e++) data[e - 1] = data[e];
		return d;
	}
	return (T)NULL;
}

template <typename T> void Arraylist<T>::realloc(size_t new_size) {
	if(new_size) {
		data = (T*)std::realloc(data, sizeof(T) * new_size);
		if(pointer >= new_size) pointer = new_size - 1;
		size = new_size;
	}
}

#pragma region dict_usage_pair_t

template <> Arraylist<dict_usage_pair_t>::Arraylist(size_t initial_size) {
	data = (dict_usage_pair_t*)malloc(sizeof(dict_usage_pair_t) * initial_size);
	size = initial_size;
	pointer = 0;
}

template <> Arraylist<dict_usage_pair_t>::~Arraylist() {
	free(data);
}

template <> void Arraylist<dict_usage_pair_t>::append(dict_usage_pair_t element) {
	if(pointer >= size) {
		// reallocate the array
		size = (size) ? size << 1 : 1;
		data = (dict_usage_pair_t*)std::realloc(data, size * sizeof(dict_usage_pair_t));
		if(!data) err(1, "Memory error while allocating arraylist");
	}
	data[pointer++] = element;
}

template <> void Arraylist<dict_usage_pair_t>::insert(dict_usage_pair_t element, size_t index) {
	if(index >= size) {
                // reallocate the array
                size = (size) ? size << 1 : 1;
                data = (dict_usage_pair_t*)std::realloc(data, size * sizeof(dict_usage_pair_t));
                if(!data) err(1, "Memory error while allocating arraylist");
        }
	data[index] = element;
	pointer = index + 1;
}

template <> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_back() {
	if(size) {
		dict_usage_pair_t d = data[--pointer];
		return d;
	}
	return dict_usage_pair_t {dict_pair_t {0, 0}, 0};
}

template <> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_front() {
	if(size) {
		dict_usage_pair_t d = data[--pointer];
		return d;
	}
	return dict_usage_pair_t {dict_pair_t {0, 0}, 0};
}

template <> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop(size_t index) {
	if(size && index < size) {
		dict_usage_pair_t d = data[index];
		for(size_t e = index + 1; e < pointer; e++) data[e - 1] = data[e];
		return d;
	}
	return dict_usage_pair_t {dict_pair_t {0, 0}, 0};
}

template <> void Arraylist<dict_usage_pair_t>::realloc(size_t new_size) {
	if(new_size) {
		data = (dict_usage_pair_t*)std::realloc(data, sizeof(dict_usage_pair_t) * new_size);
		if(pointer >= new_size) pointer = new_size - 1;
		size = new_size;
	}
}

#pragma endregion

#pragma region void*

template <> Arraylist<void*>::Arraylist(size_t initial_size) {
	data = (void**)malloc(sizeof(void*) * initial_size);
	size = initial_size;
	pointer = 0;
}

template <> Arraylist<void*>::~Arraylist() {
	free(data);
}

template <> void Arraylist<void*>::append(void* element) {
	if(pointer >= size) {
		// reallocate the array
		size = (size) ? size << 1 : 1;
		data = (void**)std::realloc(data, size * sizeof(void*));
		if(!data) err(1, "Memory error while allocating arraylist");
	}
	data[pointer++] = element;
}

template <> void Arraylist<void*>::insert(void* element, size_t index) {
	if(index >= size) {
                // reallocate the array
                size = (size) ? size << 1 : 1;
                data = (void**)std::realloc(data, size * sizeof(void*));
                if(!data) err(1, "Memory error while allocating arraylist");
        }
	data[index] = element;
	pointer = index + 1;
}

template <> void* Arraylist<void*>::pop_back() {
	if(size) {
		void* d = data[--pointer];
		return d;
	}
	return NULL;
}

template <> void* Arraylist<void*>::pop_front() {
	if(size) {
		void* d = data[--pointer];
		return d;
	}
	return NULL;
}

template <> void* Arraylist<void*>::pop(size_t index) {
	if(size && index < size) {
		void* d = data[index];
		for(size_t e = index + 1; e < pointer; e++) data[e - 1] = data[e];
		return d;
	}
	return NULL;
}

template <> void Arraylist<void*>::realloc(size_t new_size) {
	if(new_size) {
		data = (void**)std::realloc(data, sizeof(void*) * new_size);
		if(pointer >= new_size) pointer = new_size - 1;
		size = new_size;
	}
}

#pragma endregion