#include "mempage_test.h"
#include "../mempage.h"

#include <assert.h>
#include <stdio.h>

void mp_test_index() {
    printf("Testing mempage insert/retrieve system\n");
    mempage mp = create_mempage(25, 500);
    for(int i = 0; i < 500; i++) mempage_put(mp, i, 1);
    for(int i = 0; i < 500; i++) assert(mempage_get(mp, i) == 1);
    destroy_mempage(mp);
    mp = create_mempage(25, 500);
    destroy_mempage(mp);
}