#include "mempage.h"
#include "arraylist.h"

#include <stdlib.h>
#include <err.h>

mempage create_mempage(size_t page_max, __uint128_t elements) {
    mempage mp = malloc(sizeof(mempage_str));
    if(!mp) err(1, "Memory Error while allocating memory page manager\n");

    mp->pages = create_ll();
    mp->count_per_page = page_max;
    mp->num_elements = elements;

    __uint128_t pages = (elements / page_max) + 1;
    for(__uint128_t p = 0; p < pages; p++) append_ll(mp->pages, create_uint128_arraylist(page_max));

    return mp;
}

void destroy_mempage(mempage mp) {
    if(mp) {
        uint128_arraylist* lists = ll_to_arr(mp->pages);
        for(uint128_arraylist* l = lists; *l; l++) destroy_uint128_arraylist(l);
        free(lists);

        destroy_ll(mp->pages);
        free(mp);
    }
}

__uint128_t mempage_get(mempage mp, __uint128_t index) {
    __uint128_t page = index / mp->count_per_page, page_index = index % mp->count_per_page;

    // Extract the page
    ll_node n = mp->pages->head;
    for(__uint128_t p = 1; p < page; p++) n = n->next;

    // Extract the int
    return ((uint128_arraylist)n->data)->data[page_index];
}

void mempage_put(mempage mp, __uint128_t index, __uint128_t data) {
    __uint128_t page = index / mp->count_per_page, page_index = index % mp->count_per_page;

    // Extract the page
    ll_node n = mp->pages->head;
    for(__uint128_t p = 1; p < page; p++) n = n->next;

    // Extract the int
    ((uint128_arraylist)n->data)->data[page_index] = data;
}