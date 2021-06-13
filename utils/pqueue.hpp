#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <pthread.h>

#include "tarraylist.hpp"
#include "../gameplay/reversi_defs.h"

template <typename T> class LockedPriorityQueue {
    protected:
        void swap_heap_elements(size_t a, size_t b) {
            T temp;
            temp = data[a];
            data[a] = data[b];
            data[b] = temp;
        }

        void min_heapify(size_t n, size_t i) {
            size_t l, r, smallest;

            while(1) {
                l = (i << 1) + 1;
                r = (i << 1) + 2;

                if(l < n && data[l] < data[i]) smallest = l;
                else smallest = i;
                if(r < n && data[r] < data[smallest]) smallest = r;
                if(smallest != i) {
                    swap_heap_elements(i, smallest);
                    i = smallest;
                }
                else break;
            }
        }

        void build_min_heap(size_t n) {
            for(size_t i = (n >> 1) - 1; i > 0; i--) min_heapify(n, i);
            min_heapify(n, 0);
        }

        void heapsort(size_t n) {
            build_min_heap(n);
            for(size_t i = n - 1; i > 0; i--) {
                swap_heap_elements(i, 0);
                min_heapify(i, 0);
            }
        }

    public:
        T* data;
        size_t size;
        size_t count;
        pthread_mutex_t mutex;

        LockedPriorityQueue(size_t initial_size) {
            data = (T*)malloc(sizeof(T) * initial_size);
			if(!data) err(1, "Memory error while allocating arraylist\n");
			size = initial_size;
			count = 0;
            pthread_mutex_init(&mutex, 0);
        }

        ~LockedPriorityQueue() {
            free(data);
        }

        void push(T element) {
            while(pthread_mutex_trylock(&mutex)) sched_yield();

            if(count >= size) {
				// reallocate the array
				size = (size) ? size << 1 : 1;
				data = (T*)std::realloc(data, size * sizeof(T));
				if(!data) err(1, "Memory error while allocating arraylist\n");
			}

            data[count] = element;
            size_t i = count++;
            while(i && data[i >> 1] > data[i]) {
                swap_heap_elements(i, i >> 1);
                i = i >> 1;
            }

            pthread_mutex_unlock(&mutex);
        }

        void push_bulk(T* elements, size_t n) {
            while(pthread_mutex_trylock(&mutex)) sched_yield();

            for(size_t ni = 0; ni < n; ni++) {
                if(count >= size) {
                    // reallocate the array
                    size = (size) ? size << 1 : 1;
                    data = (T*)std::realloc(data, size * sizeof(T));
                    if(!data) err(1, "Memory error while allocating arraylist\n");
                }

                data[count] = elements[ni];
                size_t i = count++;
                while(i && data[i >> 1] > data[i]) {
                    swap_heap_elements(i, i >> 1);
                    i = i >> 1;
                }
            }

            pthread_mutex_unlock(&mutex);
        }

        T pop() {
            while(pthread_mutex_trylock(&mutex)) sched_yield();

            if(count) {
                T result = data[0];
                data[0] = data[--count];
                min_heapify(count, 0);
                pthread_mutex_unlock(&mutex);
                return result;
            }

            pthread_mutex_unlock(&mutex);

            return (T)NULL;
        }

        T* pop_bulk(size_t n) {
            while(pthread_mutex_trylock(&mutex)) sched_yield();

            if(count >= n | count) {
                T* result = (T*)malloc(sizeof(T) * n);
                if(!result) err(1, "Memory error while allocating priority queue pop chunk of size %lu\n", n);

                size_t prev_count = count;

                for(size_t ni = 0; ni < ((prev_count >= n) ? n : prev_count); ni++) {
                    T res = data[0];
                    data[0] = data[--count];
                    min_heapify(count, 0);
                    result[ni] = res;
                }

                if(prev_count < n) {
                    size_t diff = n - prev_count;
                    for(size_t ni = 0; ni < diff; ni++) result[prev_count + ni] = 0;
                }

                pthread_mutex_unlock(&mutex);

                return result;
            }

            pthread_mutex_unlock(&mutex);

            return (T*)NULL;
        }
};

template<> void LockedPriorityQueue<board>::min_heapify(size_t n, size_t i);
