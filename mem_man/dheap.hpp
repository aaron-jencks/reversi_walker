#pragma once

#include "./mempage.h"
#include "./mmap_man.h"
#include "../project_defs.hpp"
#include "../utils/path_util.h"
#include "../gameplay/reversi_defs.h"
#include <stdint.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <err.h>
#include <pthread.h>
#include <string.h>


template <typename T> class MMappedLockedPriorityQueue {
    protected:
        inline size_t get_bin_number(size_t i) {
            return ++i / (mem_manager->elements_per_bin / sizeof(T));
        }

        inline size_t get_bin_index(size_t i) {
            return i % (mem_manager->elements_per_bin / sizeof(T));
        }

        void swap_heap_elements(size_t a, size_t b) {
            T temp;
            temp = get(a);
            set(a, get(b));
            set(b, temp);
        }

        virtual void min_heapify(size_t n, size_t i) {
            size_t l, r, smallest;

            while(1) {
                l = (i << 1) + 1;
                r = (i << 1) + 2;

                if(l < n && get(l) < get(i)) smallest = l;
                else smallest = i;
                if(r < n && get(r) < get(smallest)) smallest = r;
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
        T** data;
        size_t size;
        size_t bin_count;
        size_t count;
        pthread_mutex_t mutex;
        mmap_man mem_manager;

        MMappedLockedPriorityQueue(size_t initial_size) {
            size_t bin_size = ARRAYLIST_INCREMENT * sizeof(T);
            size_t initial_bin_count = initial_size / ARRAYLIST_INCREMENT;
            data = (T**)malloc(sizeof(T*) * initial_bin_count);
			if(!data) err(1, "Memory error while allocating arraylist\n");
			size = initial_size;
			count = 0;
            pthread_mutex_init(&mutex, 0);

            // Setup the heap directory
            struct stat st;
            char* temp = (char*)malloc(sizeof(char) * (strlen(get_temp_path()) + 6));
            temp = (char*)memcpy(temp, get_temp_path(), strlen(get_temp_path()));
            memcpy(temp + strlen(get_temp_path()), "/heap", 6);
            if(stat(temp, &st) == -1) mkdir(temp, 0700);

            mem_manager = create_mmap_man(FILE_SIZE_INCREMENT, bin_size, temp);
            free(temp);

            for(size_t b = 0; b < initial_bin_count; b++) {
                data[b] = (T*)mmap_allocate_bin(mem_manager);
            }
            bin_count = initial_bin_count;
        }

        ~MMappedLockedPriorityQueue() {
            free(data);
            destroy_mmap_man(mem_manager);
        }

        inline T get(size_t i) {
            return data[get_bin_number(i)][get_bin_index(i)];
        }

        inline void set(size_t i, T v) {
            data[get_bin_number(i)][get_bin_index(i)] = v;
        }

        void push(T element) {
            while(pthread_mutex_trylock(&mutex)) sched_yield();

            if(count >= size) {
				// reallocate the array
				size = ++bin_count * ARRAYLIST_INCREMENT;
				data = (T**)std::realloc(data, bin_count * sizeof(T*));
				if(!data) err(1, "Memory error while allocating arraylist\n");
                data[bin_count - 1] = (T*)mmap_allocate_bin(mem_manager);
			}

            set(count, element);
            size_t i = count++;
            while(i && get(i >> 1) > get(i)) {
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
                    size = ++bin_count * ARRAYLIST_INCREMENT;
                    data = (T**)std::realloc(data, bin_count * sizeof(T*));
                    if(!data) err(1, "Memory error while allocating arraylist\n");
                    data[bin_count - 1] = (T*)mmap_allocate_bin(mem_manager);
                }

                set(count, elements[ni]);
                size_t i = count++;
                while(i && get(i >> 1) > get(i)) {
                    swap_heap_elements(i, i >> 1);
                    i = i >> 1;
                }
            }

            pthread_mutex_unlock(&mutex);
        }

        T pop() {
            while(pthread_mutex_trylock(&mutex)) sched_yield();

            if(count) {
                T result = get(0);
                set(0, get(--count));
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
                T* result = (T*)malloc(sizeof(T) * ((count >= n) ? n : count + 1));
                if(!result) err(1, "Memory error while allocating priority queue pop chunk of size %lu\n", n);

                size_t prev_count = count;

                for(size_t ni = 0; ni < ((prev_count >= n) ? n : prev_count); ni++) {
                    T res = get(0);
                    set(0, get(--count));
                    min_heapify(count, 0);
                    result[ni] = res;
                }

                if(prev_count < n) {
                    result[prev_count] = 0;
                }

                pthread_mutex_unlock(&mutex);

                return result;
            }

            pthread_mutex_unlock(&mutex);

            return (T*)NULL;
        }
};