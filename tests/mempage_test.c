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

void mp_buff_test_index() {
    printf("Testing mempage buffer insert/retrieve system\n");
    mempage_buff b = create_mempage_buff(500, 25);
    for(int i = 0; i < 500; i++) mempage_buff_put(b, i, i);
    for(int i = 0; i < 500; i++) assert(mempage_buff_get(b, i) == i);
    destroy_mempage_buff(b);
}