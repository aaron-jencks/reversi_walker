#pragma once

#include "reversi_defs.h"
#include "../utils/tarraylist.hpp"

#include <stdint.h>

class ReversiIterator {
    private:
        board_t current_board;
        uint8_t current_row;
        uint8_t current_column;
    public:
        ReversiIterator(board src);
        ~ReversiIterator();

        uint8_t has_next();
        coord next(Arraylist<coord>* coord_cache);
};
