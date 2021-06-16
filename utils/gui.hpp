#pragma once

#include "../gameplay/reversi_defs.h"
#include "../mem_man/heir.hpp"
#include "tarraylist.hpp"
#include "csv.h"

#include <stdio.h>
#include <stdint.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <time.h>


uint8_t ask_yes_no(const char* string);
void display_board(board b);
void display_moves(board b, Arraylist<void*>* coords);
void display_capture_counts(uint64_t cc);

typedef struct {
    struct statvfs disk_usage_buff;
    double GB;
    double disk_total;
    time_t start, current, fps_timer, sleep_timer, log_timer;
    clock_t cstart;
    uint32_t cpu_time, cpu_days, cpu_hours, cpu_minutes, cpu_seconds,
                run_time, run_days, run_hours, run_minutes, run_seconds, previous_run_time = start, fps_update_time, print_sleep;
    uint64_t previous_board_count = 0, fps = 0, *ccount, *eccount;
    double disk_avail, disk_used, disk_perc;
    heirarchy hcache;

    csv_cont csv;
    struct stat sbuff;
} loop_display_t;

loop_display_t* initialize_main_loop_display(char* csv_filename, heirarchy cache, uint64_t* count, uint64_t* explored_count);
void display_main_loop(loop_display_t* display);
