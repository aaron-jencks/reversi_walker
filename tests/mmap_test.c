#include "../mem_man/heir.h"
#include "../mem_man/mmap_man.h"
#include "../hashing/hash_functions.h"
#include "../gameplay/reversi.h"
#include "mmap_test.h"
#include "../utils/arraylist.h"

#include <stdio.h>
#include <assert.h>
#include <err.h>


void display_board(board b) {
    if(b) {
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c++) {
                printf("%c", board_get(b, r, c) + '0');
            }
            printf("\n");
        }
        printf("\n");
    }
}


void mmap_test_readback() {
    heirarchy h = create_heirarchy();
    printf("Heirarchy statistics:\nBits per level: %lu\nNumber of bits in the final level: %lu\nNumber of levels: %lu\n", h->num_bits_per_level, h->num_bits_per_final_level, h->num_levels);

    __uint128_t k = 0;
    for(size_t b = 0; b < 65536; b++) {
        printf("Testing insertion of a new bit %lu\n", b);
        assert(heirarchy_insert(h, ++k));
        printf("Testing duplicate insertion\n");
        assert(!heirarchy_insert(h, k));
    }
    destroy_heirarchy(h);
}

void mmap_spiral_hash_test() {
    printf("Testing hash function uniqueness\n");
    uint128_arraylist keys = create_uint128_arraylist(10000000);
    ptr_arraylist boards = create_ptr_arraylist(10000000);
    board b = create_board(2, 8, 8), bb;
    uint8_t finished = 0;

    append_ddal(keys, board_spiral_hash(b));

    while(!finished) {
        board bb = clone_board(b);

        finished = 1;
        for(uint8_t i = 0; i < (b->height * b->width); i++) {
            uint8_t r = i / b->width, c = i % b->width;
            if(board_get(b, r, c) < 2) {
                board_put(bb, r, c, board_get(b, r, c) + 1);
                finished = 0;
                break;
            }
            else board_put(bb, r, c, 0);
        }

        __uint128_t hash = board_spiral_hash(bb);
        for(size_t k = 0; k < keys->pointer; k++) {
            if(keys->data[k] == hash) {
                printf("Identical hashes for boards:\n");
                display_board(bb);
                printf("matches:\n");
                display_board(boards->data[k]);
                printf("with hash %lu %lu and id %lu matches hash %lu %lu from board id %lu\n",
                ((uint64_t*)(&hash))[1], ((uint64_t*)(&hash))[0], keys->pointer, 
                ((uint64_t*)(&keys->data[k]))[1], ((uint64_t*)(&keys->data[k]))[0], k);
            }
            assert(keys->data[k] != hash);
        }
        append_ddal(keys, hash);
        append_pal(boards, bb);

        b = bb;
    }

    destroy_board(b);
}