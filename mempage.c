#include "mempage.h"
#include "arraylist.h"

#include <stdlib.h>
#include <err.h>

mempage create_mempage(size_t page_max, __uint128_t elements) {
    mempage mp = malloc(sizeof(mempage_str));
    if(!mp) err(1, "Memory Error while allocating memory page manager\n");

    __uint128_t pages = (elements / page_max) + 1;

    #ifdef mempagedebug
        printf("Creating a mempage with %lu %lu pages\n", ((uint64_t*)&pages)[1], ((uint64_t*)&pages)[0]);
    #endif

    // mp->pages = create_ll();
    mp->pages = create_ptr_arraylist(pages + 1);
    mp->count_per_page = page_max;
    mp->num_elements = elements;

    for(__uint128_t p = 0; p < pages; p++) append_pal(mp->pages, create_ptr_arraylist(page_max));

    return mp;
}

void destroy_mempage(mempage mp) {
    if(mp) {
        ptr_arraylist* lists = (ptr_arraylist*)(mp->pages->data);
        for(ptr_arraylist* l = lists; *l; l++) destroy_ptr_arraylist(*l);
        destroy_ptr_arraylist(mp->pages);
        free(mp);
    }
}

void* mempage_get(mempage mp, __uint128_t index) {
    if(index >= mp->num_elements) err(4, "Index out of bounds in mempage\n");

    uint32_t page = index / mp->count_per_page, page_index = index % mp->count_per_page;

    // Extract the page
    ptr_arraylist l = (ptr_arraylist)mp->pages->data[page];

    // Extract the int
    return (l)->data[page_index];
}

void mempage_put(mempage mp, __uint128_t index, void* data) {
    if(index >= mp->num_elements) err(4, "Index out of bounds in mempage\n");
    
    uint32_t page = index / mp->count_per_page, page_index = index % mp->count_per_page;

    // Extract the page
    ptr_arraylist l = (ptr_arraylist)mp->pages->data[page];

    // Extract the int
    (l)->data[page_index] = data;
}