#include "pqueue.hpp"

template<> void LockedPriorityQueue<board>::min_heapify(size_t n, size_t i) {
    size_t l, r, smallest;

    while(1) {
        l = (i << 1) + 1;
        r = (i << 1) + 2;

        if(l < n && data[l]->level < data[i]->level) smallest = l;
        else smallest = i;
        if(r < n && data[r]->level < data[smallest]->level) smallest = r;
        if(smallest != i) {
            swap_heap_elements(i, smallest);
            i = smallest;
        }
        else break;
    }
}
