#include "gui.hpp"
#include "../gameplay/reversi.h"

#include <err.h>

uint8_t ask_yes_no(const char* string) {
    char d;
    while(1) {
        printf("%s(y/N): ", string);
        d = getc(stdin);
        if(d == '\n' || d == 'n' || d == 'N') return 0;
        else if(d == 'y' || d == 'Y') return 1;
    }
    getc(stdin);    // Read the extra \n character
}

void display_board(board b) {
    if(b) {
        printf("%s's Turn\n", (b->player == 1) ? "White" : "Black");
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c++) {
                printf("%c", board_get(b, r, c) + '0');
            }
            printf("\n");
        }
        printf("\n");
    }
}

void display_moves(board b, Arraylist<void*>* coords) {
    if(b) {
        printf("%s's Turn\n", (b->player == 1) ? "White" : "Black");
        for(uint8_t r = 0; r < b->height; r++) {
            for(uint8_t c = 0; c < b->width; c++) {
                uint8_t move = 0;
                for(size_t cd = 0; cd < coords->pointer; cd++) {
                    coord cord = (coord)coords->data[cd];
                    if(r == cord->row && c == cord->column) {
                        printf("x");
                        move = 1;
                        break;
                    }
                }
                if(!move) printf("%c", board_get(b, r, c) + '0');
            }
            printf("\n");
        }
        printf("\n");
    }
}

void display_capture_counts(uint64_t cc) {
    /*
    * 0: upper-left
    * 1: up
    * 2: upper-right
    * 3: left
    * 4: right
    * 5: lower-left
    * 6: lower
    * 7: lower-right
    */
    printf("Capture Counts:\n");
    uint8_t c;
    for(uint8_t i = 0; i < 8; i++) {
        c = capture_count_get_count(cc, i);
        switch(i) {
            case 0:
                printf("\tNorthwest: ");
                break;
            case 1:
                printf("\tNorth: ");
                break;
            case 2:
                printf("\tNortheast: ");
                break;
            case 3:
                printf("\tWest: ");
                break;
            case 4:
                printf("\tEast: ");
                break;
            case 5:
                printf("\tSouthwest: ");
                break;
            case 6:
                printf("\tSouth: ");
                break;
            case 7:
                printf("\tSoutheast: ");
                break;
        }
        printf("%u\n", c);
    }
    printf("\n");
}

#pragma region Main Loop Display

loop_display_t* initialize_main_loop_display(char* csv_filename, heirarchy cache, uint64_t* count, uint64_t* explored_count) {
    loop_display_t* display = (loop_display_t*)calloc(1, sizeof(loop_display_t));
    if(!display) err(1, "Memory error while allocating display loop\n");
    display->GB = 1024 * 1024 * 1024;
    if(statvfs("/home", &display->disk_usage_buff)) err(2, "Finding information about the disk failed\n");
    display->disk_total = (double)(display->disk_usage_buff.f_blocks * display->disk_usage_buff.f_bsize) / display->GB;
    display->csv = create_csv_cont(csv_filename, "%u,%lu,%lu,%lu,%.2f,%.4f,%.4f,%.4f,%lu\n", 9);
    display->hcache = cache;
    display->ccount = count;
    display->eccount = explored_count;
    display->start = time(0);
    display->fps_timer = time(0);
    display->sleep_timer = time(0);
    display->log_timer = time(0);
    display->cstart = clock();
    return display;
}

void display_main_loop(loop_display_t* display) {
    if(statvfs("/home", &display->disk_usage_buff)) err(2, "Finding information about the disk failed\n");
    display->disk_avail = (double)(display->disk_usage_buff.f_bfree * display->disk_usage_buff.f_frsize) / display->GB;
    // fprintf(stderr, "The available disk space is %lu %lu for a total of %lu\n", display->disk_usage_buff.f_bfree, display->disk_usage_buff.f_frsize, display->disk_avail);
    display->disk_used = display->disk_total - display->disk_avail;
    display->disk_perc = (double)(display->disk_used / display->disk_total) * (double)100;
    
    display->current = time(0);
    display->run_time = display->current - display->start;

    #ifdef slowprint
        print_sleep = (current - sleep_timer) / 60;
    #endif

    display->fps_update_time = (display->current - display->fps_timer) / 1;

    display->run_days = display->run_time / 86400;
    display->run_hours = (display->run_time / 3600) % 24;
    display->run_minutes = (display->run_time / 60) % 60;
    display->run_seconds = display->run_time % 60;

    display->cpu_time = (uint32_t)(((double)(clock() - display->cstart)) / CLOCKS_PER_SEC);
    display->cpu_days = display->cpu_time / 86400;
    display->cpu_hours = (display->cpu_time / 3600) % 24;
    display->cpu_minutes = (display->cpu_time / 60) % 60;
    display->cpu_seconds = display->cpu_time % 60;

    if(display->fps_update_time) {
        display->fps = (*display->eccount - display->previous_board_count);
        display->previous_board_count = *display->eccount;
        display->fps_timer = time(0);
    }

    if((display->current - display->log_timer) / 60) {
        append_data(display->csv, display->run_time, display->fps, *display->ccount, *display->eccount, display->disk_perc, 
            fdict_load_factor(display->hcache->fixed_cache), 
            hdict_load_factor(display->hcache->rehashing_cache), 
            fdict_load_factor(display->hcache->temp_board_cache),
            display->hcache->collision_count);
        display->log_timer = time(0);
    }

    #ifndef hideprint
    
    #ifdef slowprint
        if(print_sleep) {
            sleep_timer = time(0);
    #else
        printf("\r");
    #endif
    
        printf("Found %'lu final board states. Explored %'lu boards with %'lu collisions @ %'lu b/s. Runtime: %0d:%02d:%02d:%02d CPU Time: %0d:%02d:%02d:%02d Disk usage: %.2f%%", 
                *display->ccount, *display->eccount, display->hcache->collision_count, display->fps,
                display->run_days, display->run_hours, display->run_minutes, display->run_seconds,
                display->cpu_days, display->cpu_hours, display->cpu_minutes, display->cpu_seconds,
                display->disk_perc);

    #ifdef slowprint
            printf("\n");
        }
        else sched_yield();
    #endif
    
    #endif
}

#pragma endregion
