#pragma once

#include "dictionary/dict_def.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <err.h>
#include <pthread.h>

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

		size_t index(T value) {
			if(size) {
				for(size_t i = 0; i < size; i++) {
					if(data[i] == value) return i;
				}
				return size;
			}
			return 0;
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
template<> size_t Arraylist<dict_usage_pair_t>::index(dict_usage_pair_t value);
template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop(size_t index);
template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_front();
template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_back();

template <typename T> class RingBuffer {
	public:
		T* data;
		size_t size;
		size_t count;
		size_t pop_pointer;
		size_t push_pointer;

		RingBuffer(size_t initial_size) {
			data = (T*)malloc(sizeof(T) * initial_size);
			if(!data) err(1, "Memory error while allocating arraylist\n");
			size = initial_size;
			pop_pointer = 0;
			push_pointer = 0;
			count = 0;
		}

		~RingBuffer() {
			free(data);
		}

		void append(T element) {
			if(count) {
				if(count == size) {
					// reallocate the array
					size_t psize = size;
					size = (size) ? size << 1 : 1;
					data = (T*)std::realloc(data, size * sizeof(T));
					if(!data) err(1, "Memory error while allocating arraylist\n");
					if(push_pointer < psize) for(size_t e = psize - 1; e >= push_pointer; e--) { data[e + 1] = data[e]; }
				}
				else if(push_pointer >= size) {
					push_pointer = 0;
				}
			}
			data[push_pointer++] = element;
			count++;
		}

		T pop_back() {
			if(size && count) {
				T d;

				if(push_pointer) {
					d = data[--push_pointer];
				}
				else {
					push_pointer = size - 1;
					d = data[push_pointer];
				}

				count--;

				return d;
			}
			return (T)NULL;
		}

		T pop_front() {
			if(size && count) {
				T d = data[pop_pointer++];

				if(pop_pointer >= size) pop_pointer = 0;

				count--;

				return d;
			}
			return (T)NULL;
		}
};

template <typename T> class LockedRingBuffer : public RingBuffer<T> {
	public:
		pthread_mutex_t mutex;

		LockedRingBuffer(size_t initial_size) : RingBuffer<T>(initial_size) {
			if(pthread_mutex_init(&mutex, 0)) err(4, "Failed to initialize mutex for locked arraylist\n");
		}

		~LockedRingBuffer() {
			free(RingBuffer<T>::data);
		}

		void append(T element) {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			RingBuffer<T>::append(element);
			pthread_mutex_unlock(&mutex);
		}

		T pop_back() {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			T result = RingBuffer<T>::pop_back();
			pthread_mutex_unlock(&mutex);
			return result;
		}

		T pop_front() {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			T result = RingBuffer<T>::pop_front();
			pthread_mutex_unlock(&mutex);
			return result;
		}
};

template <typename T> class LockedArraylist : public Arraylist<T> {
	public:
		pthread_mutex_t mutex;

		LockedArraylist(size_t initial_size) : Arraylist<T>(initial_size) {
			if(pthread_mutex_init(&mutex, 0)) err(4, "Failed to initialize mutex for locked arraylist\n");
		}

		~LockedArraylist() { free(Arraylist<T>::data); }

		void append(T element) {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			Arraylist<T>::append(element);
			pthread_mutex_unlock(&mutex);
		}

		void insert(T element, size_t index) {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			Arraylist<T>::insert(element, index);
			pthread_mutex_unlock(&mutex);
		}

		T pop_front() {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			T result = Arraylist<T>::pop_front();
			pthread_mutex_unlock(&mutex);
			return result;
		}

		T pop_back() {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			T result = Arraylist<T>::pop_back();
			pthread_mutex_unlock(&mutex);
			return result;
		}

		T pop(size_t element) {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			T result = Arraylist<T>::pop(element);
			pthread_mutex_unlock(&mutex);
			return result;
		}

		T get(size_t index) {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			T result = Arraylist<T>::data[index];
			pthread_mutex_unlock(&mutex);
			return result;
		}

		void put(T element, size_t index) {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			Arraylist<T>::data[index] = element;
			pthread_mutex_unlock(&mutex);
		}

		void realloc(size_t new_size) {
			while(pthread_mutex_trylock(&mutex)) sched_yield();
			Arraylist<T>::realloc(new_size);
			pthread_mutex_unlock(&mutex);
		}

};
