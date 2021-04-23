#pragma once

#include "../gameplay/reversi_defs.h"
#include "../mem_man/heir.hpp"
#include "tarraylist.hpp"

#include <stdio.h>
#include <stdint.h>


uint8_t ask_yes_no(const char* string);
void display_board(board b);
void display_moves(board b, Arraylist<void*>* coords);
void display_capture_counts(uint64_t cc);

void initialize_main_loop_display(char* csv_filename, heirarchy cache, uint64_t* count, uint64_t* explored_count);
void display_main_loop();
