#include <stdlib.h>
#include <err.h>

#include "valid_moves.h"
#include "../utils/tarraylist.hpp"

uint64_t encode_valid_position(uint64_t b, uint8_t row, uint8_t column) {
    uint8_t* bytes = (uint8_t*)&b;
    uint8_t ph = 1 << column;
    bytes[row] |= ph;
    return b;
}

uint64_t remove_valid_position(uint64_t b, uint8_t row, uint8_t column) {
    if(is_valid_position(b, row, column)) {
        uint8_t* bytes = (uint8_t*)&b;
        uint8_t ph = 1 << column;
        bytes[row] ^= ph;
    }

    return b;
}

uint8_t is_valid_position(uint64_t b, uint8_t row, uint8_t column) {
    uint8_t* bytes = (uint8_t*)&b;
    uint8_t ph = 1 << column;
    uint8_t r = bytes[row] & ph;
    return r;
}

uint64_t find_valid_positions_from_coord(uint64_t bi, board_t b, uint8_t row, uint8_t column) {
    int8_t sr, sc, rd, cd;
    Arraylist<uint16_t>* edges = create_uint16_arraylist(9);

    for(rd = -1; rd < 2; rd++) {
        for(cd = -1; cd < 2; cd++) {
            if(!rd && !cd) continue;
            
            sr = row + rd;
            sc = column + cd;

            if(sr >= 0 && sr < 8 && sc >= 0 && sc < 8) {
                if(board_get(b, row, column) != b->player) {
                    if(board_is_legal_move(b, sr, sc))
                        append_sal(edges, coord_to_short_ints(sr, sc));
                }
            }
        }
    }

    uint64_t result;
    for(uint64_t i = 0; i < 8; i++) {
        coord c = short_to_coord(edges->data[i]);
        result = encode_valid_position(result, c->row, c->column);
        free(c);
    }
    destroy_uint16_arraylist(edges);
    
    return result;
}

coord* retrieve_all_valid_positions(uint64_t b) {
    Arraylist<void*>* res = create_ptr_arraylist(65);

    for(uint8_t row = 0; row < 8; row++) {
        for(uint8_t column = 0; column < 8; column++) {
            if(is_valid_position(b, row, column)) {
                coord c = malloc(sizeof(coord_str));
                if(!c) err(1, "Memory Error while allocating coord\n");

                c->row = row;
                c->column = column;

                append_pal(res, c);
            }
        }
    }

    coord* arr = 0;
    if(res->size) arr = (coord*)(res->data);
    free(res);
    
    return arr;
}