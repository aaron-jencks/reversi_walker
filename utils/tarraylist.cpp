#include "tarraylist.hpp"
#include <stdlib.h>
#include <err.h>

template <typename T> Arraylist<T>::Arraylist(size_t initial_size) {
	data = malloc(sizeof(T) * initial_size);
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
		data = realloc(data, size * sizeof(T));
		if(!data) err(1, "Memory error while allocating arraylist");
	}
	data[pointer++] = element;
}

template <typename T> void Arraylist<T>::insert(T element, size_t index) {
	if(index >= size) {
                // reallocate the array
                size = (size) ? size << 1 : 1;
                data = realloc(data, size * sizeof(T));
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
}

template <typename T> T Arraylist<T>::pop_front() {
	if(size) {
		T d = data[--pointer];
		return d;
	}
}

template <typename T> T Arraylist<T>::pop(size_t index) {
	if(size && index < size) {
		T d = data[index];
		for(size_t e = index + 1; e < pointer; e++) data[e - 1] = data[e];
		return d;
	}
}

template <typename T> void Arraylist<T>::realloc(size_t new_size) {
	if(new_size) {
		data = realloc(data, sizeof(T) * new_size);
		if(pointer >= new_size) pointer = new_size - 1;
		size = new_size;
	}
}
