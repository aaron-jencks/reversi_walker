#pragma once

#include <stddef.h>

template <typename T> class Arraylist {
	public:
		T* data;
		size_t size;
		size_t pointer;

		Arraylist(size_t initial_size);
		~Arraylist();

		void append(T element);
		void insert(T element, size_t index);

		T pop_back();
		T pop_front();
		T pop(size_t index);

		void realloc(size_t new_size);
};
