#include "mempage_test.h"
#include "../mempage.h"

#include <assert.h>
#include <stdio.h>

void mp_test_index() {
    printf("Testing mempage insert/retrieve system\n");
    mempage mp = create_mempage(25, 500, 65);
    for(int i = 0; i < 500; i++) mempage_append_bin(mp, i, 1);
    for(int i = 0; i < 500; i++) assert(mempage_value_in_bin(mp, i, 1) == 1);
    destroy_mempage(mp);
    mp = create_mempage(25, 500, 65);
    destroy_mempage(mp);
}

void mp_test_realloc() {
    printf("Testing mempage realloc system\n");
    mempage mp = create_mempage(25, 500, 65);
    for(int i = 0; i < 500; i++) mempage_append_bin(mp, i % 500, 1);
    mempage_clear_all(mp);
    mempage_realloc(mp, 1000);

    for(int i = 0; i < 500; i++) mempage_append_bin(mp, i % 1000, 1);

    mempage_buff buff = create_mempage_buff(500, 25);
    __uint128_t offset = 0;
    for(size_t p = 0; p < mp->page_count; p++) {
        __uint128_t** page = mp->pages[p];
        for(size_t b = 0; b < mp->count_per_page; b++) {
            __uint128_t* bin = page[b];
            size_t bcount = mp->bin_counts[p][b];

            for(size_t be = 0; be < bcount; be++) {
                if(!bin[be]) break;

                printf("\nRetrieving %lu %lu into index %lu %lu", ((uint64_t*)&bin[be])[1], ((uint64_t*)&bin[be])[0], ((uint64_t*)&offset)[1], ((uint64_t*)&offset)[0]);
                fflush(stdout);
                
                mempage_buff_put(buff, offset++, bin[be]);
            }
        }
    }
}

void mp_buff_test_index() {
    printf("Testing mempage buffer insert/retrieve system\n");
    mempage_buff b = create_mempage_buff(500, 25);
    for(int i = 0; i < 500; i++) mempage_buff_put(b, i, i);
    for(int i = 0; i < 500; i++) assert(mempage_buff_get(b, i) == i);
    destroy_mempage_buff(b);
}