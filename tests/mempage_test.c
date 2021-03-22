#include "mempage_test.h"
#include "../utils/dictionary/dmempage.h"

#include <assert.h>
#include <stdio.h>

void mp_test_index() {
    printf("Testing mempage insert/retrieve system\n");
    dmempage mp = create_dmempage(25, 500, 65);
    for(int i = 0; i < 500; i++) dmempage_append_bin(mp, i, (dict_pair_t){1, 0});
    for(int i = 0; i < 500; i++) assert(dmempage_value_in_bin(mp, i, 1) == 1);
    destroy_dmempage(mp);
    mp = create_dmempage(25, 500, 65);
    destroy_dmempage(mp);
}

void mp_test_clear() {
    printf("Testing mempage realloc system\n");
    dmempage mp = create_dmempage(25, 500, 65);
    for(int i = 0; i < 500; i++) dmempage_append_bin(mp, i % 500, (dict_pair_t){i, 0});
    dmempage_clear_all(mp);

    for(size_t p = 0; p < mp->page_count; p++) {
        dict_pair_t** page = mp->pages[p];
        for(size_t b = 0; b < mp->count_per_page; b++) {
            dict_pair_t* bin = page[b];
            size_t bcount = mp->bin_counts[p][b];

            for(size_t be = 0; be < bcount; be++) {
                assert(!bin[be].key);
                break;
            }
        }
    }
}

void mp_test_realloc() {
    printf("Testing mempage realloc system\n");
    dmempage mp = create_dmempage(25, 500, 65);
    for(int i = 0; i < 500; i++) dmempage_append_bin(mp, i % 500, (dict_pair_t){i, 0});
    dmempage_clear_all(mp);
    dmempage_realloc(mp, 1000);

    for(int i = 0; i < 500; i++) 
        dmempage_append_bin(mp, i % 1000, (dict_pair_t){i, 0});

    dmempage_buff buff = create_dmempage_buff(500, 25);
    __uint128_t offset = 0;
    uint64_t previous = 0, current = 0;
    for(size_t p = 0; p < mp->page_count; p++) {
        dict_pair_t** page = mp->pages[p];
        for(size_t b = 0; b < mp->count_per_page; b++) {
            dict_pair_t* bin = page[b];
            size_t bcount = mp->bin_counts[p][b];

            for(size_t be = 0; be < bcount; be++) {
                if(!bin[be].key) break;

                current = bin[be].key;

                printf("\nRetrieving %lu %lu into index %lu %lu", ((uint64_t*)&bin[be].key)[1], ((uint64_t*)&bin[be].key)[0], ((uint64_t*)&offset)[1], ((uint64_t*)&offset)[0]);
                fflush(stdout);
                
                dmempage_buff_put(buff, offset++, bin[be]);
                previous = current;
            }
        }
    }
}

void mp_buff_test_index() {
    printf("Testing mempage buffer insert/retrieve system\n");
    dmempage_buff b = create_dmempage_buff(500, 25);
    for(int i = 0; i < 500; i++) dmempage_buff_put(b, i, (dict_pair_t){i, 0});
    for(int i = 0; i < 500; i++) assert(dmempage_buff_get(b, i).key == i);
    destroy_dmempage_buff(b);
}