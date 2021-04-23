#include "gui.hpp"
#include "../gameplay/reversi.h"
#include "csv.h"

#include <sys/statvfs.h>
#include <sys/stat.h>
#include <time.h>

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

struct statvfs disk_usage_buff;
const double GB = 1024 * 1024 * 1024;
double disk_total;
time_t start = time(0), current, fps_timer = time(0), sleep_timer = time(0), log_timer = time(0);
clock_t cstart = clock();
uint32_t cpu_time, cpu_days, cpu_hours, cpu_minutes, cpu_seconds,
            run_time, run_days, run_hours, run_minutes, run_seconds, previous_run_time = start, fps_update_time, print_sleep;
uint64_t previous_board_count = 0, fps = 0, *ccount, *eccount;
double disk_avail, disk_used, disk_perc;
heirarchy hcache;

csv_cont csv;
struct stat sbuff;

void initialize_main_loop_display(char* csv_filename, heirarchy cache, uint64_t* count, uint64_t* explored_count) {
    if(statvfs("/home", &disk_usage_buff)) err(2, "Finding information about the disk failed\n");
    csv = create_csv_cont(csv_filename, "%u,%lu,%lu,%lu,%.2f,%.4f,%.4f,%.4f,%lu\n", 9);
    hcache = cache;
    ccount = count;
    eccount = explored_count;
}

void display_main_loop() {
    if(statvfs("/home", &disk_usage_buff)) err(2, "Finding information about the disk failed\n");
    disk_avail = (double)(disk_usage_buff.f_bfree * disk_usage_buff.f_frsize) / GB;
    disk_used = disk_total - disk_avail;
    disk_perc = (double)(disk_used / disk_total) * (double)100;
    
    current = time(0);
    run_time = current - start;

    #ifdef slowprint
        print_sleep = (current - sleep_timer) / 60;
    #endif

    fps_update_time = (current - fps_timer) / 1;

    run_days = run_time / 86400;
    run_hours = (run_time / 3600) % 24;
    run_minutes = (run_time / 60) % 60;
    run_seconds = run_time % 60;

    cpu_time = (uint32_t)(((double)(clock() - cstart)) / CLOCKS_PER_SEC);
    cpu_days = cpu_time / 86400;
    cpu_hours = (cpu_time / 3600) % 24;
    cpu_minutes = (cpu_time / 60) % 60;
    cpu_seconds = cpu_time % 60;

    if(fps_update_time) {
        fps = (*eccount - previous_board_count);
        previous_board_count = *eccount;
        fps_timer = time(0);
    }

    if((current - log_timer) / 60) {
        append_data(csv, run_time, fps, *ccount, *eccount, disk_perc, 
            fdict_load_factor(hcache->fixed_cache), 
            hdict_load_factor(hcache->rehashing_cache), 
            fdict_load_factor(hcache->temp_board_cache),
            hcache->collision_count);
        log_timer = time(0);
    }

    #ifndef hideprint
    
    #ifdef slowprint
        if(print_sleep) {
            sleep_timer = time(0);
    #else
        printf("\r");
    #endif
    
        printf("Found %'lu final board states. Explored %'lu boards with %'lu collisions @ %'lu b/s. Runtime: %0d:%02d:%02d:%02d CPU Time: %0d:%02d:%02d:%02d Disk usage: %.2f%%", 
                *ccount, *eccount, hcache->collision_count, fps,
                run_days, run_hours, run_minutes, run_seconds,
                cpu_days, cpu_hours, cpu_minutes, cpu_seconds,
                disk_perc);

    #ifdef slowprint
            printf("\n");
        }
        else sched_yield();
    #endif
    
    #endif
}

#pragma endregion
