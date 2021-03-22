#include "iterator.hpp"
#include "reversi.h"

#include <err.h>
#include <stdlib.h>

ReversiIterator::ReversiIterator(board src) {
    clone_into_board(src, &current_board);
    current_column = 0;
    current_row = 0;
}

ReversiIterator::~ReversiIterator() {}

uint8_t ReversiIterator::has_next() {
    return current_row < current_board.height;
}

coord ReversiIterator::next(Arraylist<coord>* coord_cache) {
    while(has_next()) {
        if(board_is_legal_move(&current_board, current_row, current_column)) {
            coord c;
            if(coord_cache->pointer) c = coord_cache->pop_back();
            else c = (coord)malloc(sizeof(coord_str));
            if(!c) err(1, "Memory error while allocating coordinate\n");
            c->row = current_row;
            c->column = current_column++;

            if(current_column = current_board.width) {
                current_row++;
                current_column = 0;
            }

            return c;
        }
        else if(++current_column = current_board.width) {
            current_row++;
            current_column = 0;
        }
    }

    return 0;
}