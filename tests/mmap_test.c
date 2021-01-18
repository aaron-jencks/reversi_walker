#include "../mem_man/heir.h"
#include "../mem_man/mmap_man.h"
#include "../hashing/hash_functions.h"
#include "../gameplay/reversi.h"
#include "mmap_test.h"

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

    for(size_t b = 0; b < 128; b++) {
        __uint128_t k = ((__uint128_t)1) << b;
        printf("Testing insertion of a new bit %lu\n", b);
        assert(heirarchy_insert(h, k));
        printf("Testing duplicate insertion\n");
        assert(!heirarchy_insert(h, k));
    }
    destroy_heirarchy(h);
}

void mmap_spiral_hash_test() {
    printf("Testing hash function uniqueness\n");
    board b = create_board(2, 8, 8);

    __uint128_t v1, v2;

    display_board(b);
    v1 = board_spiral_hash(b);
    printf("V1 hash is %lu %lu\n", ((uint64_t*)&v1)[1], ((uint64_t*)&v1)[0]);

    printf("Player: %u\n", b->player);
    printf("Legality: %u\n", board_is_legal_move(b, 2, 4));
    printf("captures: %lu\n", board_place_piece(b, 2, 4));

    v2 = board_spiral_hash(b);
    display_board(b);
    printf("V2 hash is %lu %lu\n", ((uint64_t*)&v2)[1], ((uint64_t*)&v2)[0]);

    assert(v1 != v2);
    destroy_board(b);
}