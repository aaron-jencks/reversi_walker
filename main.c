#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <err.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#include <locale.h>
#include <libgen.h>

#include "./gameplay/reversi.h"
#include "./hashing/hash_functions.h"
#include "./mem_man/heir.h"
#include "./gameplay/walker.h"
#include "./utils/ll.h"
#include "./utils/arraylist.h"
#include "./gameplay/valid_moves.h"
#include "./utils/fileio.h"
#include "./utils/path_util.h"
#include "./utils/csv.h"

// TODO you can use the previous two board states to predict the next set of valid moves.

/**
 * 
 * use 'ulimit -c unlimited' to make core dump
 * 
 * TODO add ability to make checkpoints to save progress
 *  - Interpret Ctrl+C interrupt signal and cause a save
 * 
 * [DONE] make a way to swap out memory from disk when it's not used.
 * 
 */

// TODO Maybe try to use heroseh's gui stuff to make interface look nice

uint8_t SHUTDOWN_FLAG = 0;
pthread_mutex_t shutdown_lock;


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

void display_moves(board b, ptr_arraylist coords) {
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


void graceful_shutdown(int sig) {
    pthread_mutex_lock(&shutdown_lock);
    if(sig == SIGINT) {
        printf("Shutting down processor\n");
        if(SHUTDOWN_FLAG) exit(0);
        SHUTDOWN_FLAG = 1;
    }
    pthread_mutex_unlock(&shutdown_lock);
}


int main() {

    // This way we can remove the mmapped files
    signal(SIGINT, graceful_shutdown);

    SHUTDOWN_FLAG = 1;

    setlocale(LC_NUMERIC, "");

    char* temp_dir = (getenv("TEMP_DIR")) ? getenv("TEMP_DIR") : "/tmp";

    #ifndef reusefiles
        char* temp_result = malloc(sizeof(char) * (strlen(temp_dir) + 16));
        snprintf(temp_result, strlen(temp_dir) + 16, "%s/reversi.XXXXXX", temp_dir);
        temp_dir = mkdtemp(temp_result);
    #endif

    char* checkpoint_filename, *csv_filename;

    if(!getenv("CHECKPOINT_PATH")) {
        checkpoint_filename = malloc(sizeof(char) * (strlen(temp_dir) + 16));
        snprintf(checkpoint_filename, strlen(temp_dir) + 16, "%s/checkpoint.bin", temp_dir);
    }
    else checkpoint_filename = getenv("CHECKPOINT_PATH");

    csv_filename = malloc(sizeof(char) * (strlen(temp_dir) + 9));
    if(!csv_filename) err(1, "Memory error while allocating csv_filename\n");
    snprintf(csv_filename, strlen(temp_dir) + 16, "%s/log.csv", temp_dir);

    char d;
    while(1) {
        printf("Would you like to restore from a checkpoint?(y/N): ");
        d = getc(stdin);
        if(d == '\n' || d == 'n' || d == 'N') {
            d = 'n';
            break;
        }
        else if(d == 'y' || d == 'Y') {
            d = 'y';
            break;
        }
    }

    getc(stdin);    // Read the extra \n character

    // Allocate all of the stack parameters necessary for file restoration
    heirarchy cache;
    uint64_t count = 0, explored_count = 1, repeated_count = 0;
    size_t finished_count = 0;
    // char* checkpoint_filename;

    // Calculate the number of processors to use
    uint32_t procs = get_nprocs();
    #ifndef limitprocs
        procs = procs << 1;
    #else
        procs = 1;
    #endif

    // Setup the locks
    pthread_mutex_t counter_lock, explored_lock, file_lock, repeated_lock, finished_lock;

    // Setup the checkpoint saving system
    FILE** checkpoint_file = malloc(sizeof(FILE*));
    if(!checkpoint_file) err(1, "Memory error while allocating checkpoint file pointer\n");
    uint64_t saving_counter;

    // Initialize the locks
    if(pthread_mutex_init(&counter_lock, 0) || pthread_mutex_init(&explored_lock, 0) || pthread_mutex_init(&repeated_lock, 0) || 
       pthread_mutex_init(&file_lock, 0) || pthread_mutex_init(&saving_lock, 0) || pthread_mutex_init(&shutdown_lock, 0) || 
       pthread_mutex_init(&heirarchy_lock, 0) || pthread_mutex_init(&finished_lock, 0)) 
        err(4, "Initialization of counter mutex failed\n");

    #pragma region Round nprocs to the correct number
    board b = create_board(1, 6, 6);
    
    // Setup the queue
    ptr_arraylist search_queue = create_ptr_arraylist(procs + 1);
    ptr_arraylist coord_buff = create_ptr_arraylist(65), coord_cache = create_ptr_arraylist(1000);
    for(size_t c = 0; c < 1000; c++) append_pal(coord_cache, create_coord(0, 0));

    // Account for reflections and symmetry by using 1 of the 4 possible starting moves
    find_next_boards(b, coord_buff, coord_cache);

    #ifdef debug
        display_board(b);
        printf("Found %lu moves\n", coord_buff->pointer);
    #endif

    for(uint8_t im = 0; im < coord_buff->pointer; im++) {
        coord m = coord_buff->data[im];
        uint16_t sm = coord_to_short(m);
        board cb = clone_board(b);
        board_place_piece(cb, m->row, m->column);
        append_pal(search_queue, cb);
        append_pal(coord_cache, m);
        break;
    }

    destroy_board(b);

    // Perform the BFS
    while(search_queue->pointer < procs) {
        b = pop_front_pal(search_queue);
        find_next_boards(b, coord_buff, coord_cache);

        #ifdef debug
            display_moves(b, coord_buff);
            // printf("Found %lu moves\n", coord_buff->pointer);
        #endif

        for(uint8_t im = 0; im < coord_buff->pointer; im++) {
            coord m = coord_buff->data[im];
            uint16_t sm = coord_to_short(m);
            board cb = clone_board(b);
            board_place_piece(cb, m->row, m->column);
            append_pal(search_queue, cb);
            free(m);
        }

        explored_count++;

        destroy_board(b);
    }

    procs = search_queue->pointer;
    while(coord_cache->pointer) free(pop_back_pal(coord_cache));

    printf("Rounded nprocs to %d threads\n", procs);
    #pragma endregion

    ptr_arraylist threads;

    #pragma region determine if loading checkpoint

    if(d == 'y') {
        char** restore_filename = malloc(sizeof(char*));
        printf("Please enter a file to restore from: ");
        scanf("%ms", restore_filename);
        getc(stdin);    // Read the extra \n character
        processed_file pf = restore_progress_v2(*restore_filename);

        #ifndef skipconfirm
            while(1) {
                printf("Would you like to continue saving to this checkpoint?(y/N): ");
                d = getc(stdin);
                // printf("%c\n", d);
                if(d == '\n' || d == 'n' || d == 'N') {
                    checkpoint_filename = (getenv("CHECKPOINT_PATH")) ? getenv("CHECKPOINT_PATH") : checkpoint_filename; // find_temp_filename("checkpoint.bin\0");
                    break;
                }
                else if(d == 'y' || d == 'Y') {
                    checkpoint_filename = malloc(sizeof(char) * strlen(*restore_filename));
                    strcpy(checkpoint_filename, *restore_filename);
                    csv_filename = malloc(sizeof(char) * (strlen(pf->cache->final_level->file_directory) + 9));
                    if(!csv_filename) err(1, "Memory error while allocating csv_filename\n");
                    temp_dir = malloc(sizeof(char) * strlen(checkpoint_filename));
                    if(!temp_dir) err(1, "Memory error while allocating csv_filename\n");
                    strcpy(temp_dir, checkpoint_filename);
                    temp_dir = dirname(temp_dir);
                    snprintf(csv_filename, strlen(temp_dir) + 16, "%s/log.csv", temp_dir);
                    break;
                }
                else printf("\n");
            }
        #else
            checkpoint_filename = malloc(sizeof(char) * strlen(*restore_filename));
            strcpy(checkpoint_filename, *restore_filename);
            csv_filename = malloc(sizeof(char) * (strlen(pf->cache->final_level->file_directory) + 9));
            if(!csv_filename) err(1, "Memory error while allocating csv_filename\n");
            temp_dir = malloc(sizeof(char) * strlen(checkpoint_filename));
            if(!temp_dir) err(1, "Memory error while allocating csv_filename\n");
            strcpy(temp_dir, checkpoint_filename);
            temp_dir = dirname(temp_dir);
            snprintf(csv_filename, strlen(temp_dir) + 16, "%s/log.csv", temp_dir);
        #endif

        free(*restore_filename);

        // De-allocate the stuff we did to round procs, because we don't need it now.
        while(search_queue->pointer) destroy_board(pop_front_pal(search_queue));
        destroy_ptr_arraylist(search_queue);

        // Begin restore
        count = pf->found_counter;
        explored_count = pf->explored_counter;

        ptr_arraylist stacks = pf->processor_stacks;
        if(procs != pf->num_processors) {
            printf("Redistributing workload to match core count\n");
            // Redistribute workload
            ptr_arraylist all_current_boards = create_ptr_arraylist(procs * 1000), important_boards = create_ptr_arraylist(pf->num_processors + 1);
            for(ptr_arraylist* p = (ptr_arraylist*)pf->processor_stacks->data; *p; p++) {
                for(uint64_t pb = 0; pb < (*p)->pointer; pb++) {
                    board b = (*p)->data[pb];
                    // printf("Collected\n"); display_board(b);
                    append_pal((pb) ? all_current_boards : important_boards, b);
                }
                destroy_ptr_arraylist(*p);
            }

            while(stacks->pointer) pop_back_pal(stacks);

            // printf("Distributing %lu boards\n", all_current_boards->pointer + important_boards->pointer);
            
            for(uint64_t p = 0; p < procs; p++) append_pal(stacks, create_ptr_arraylist(10000));

            uint64_t p_ptr = 0;

            while(important_boards->pointer) {
                append_pal(stacks->data[p_ptr++], pop_back_pal(important_boards));
                if(p_ptr == procs) p_ptr = 0;
            }

            while(all_current_boards->pointer) {
                append_pal(stacks->data[p_ptr++], pop_back_pal(all_current_boards));
                if(p_ptr == procs) p_ptr = 0;
            }
        }

        // Create threads

        // Distribute the initial states to a set of new pthreads.
        threads = create_ptr_arraylist(procs + 1);

        for(uint64_t t = 0; t < procs; t++) {
            pthread_t* thread_id = (pthread_t*)malloc(sizeof(pthread_t));
            if(!thread_id) err(1, "Memory error while allocating thread id\n");

            #ifdef filedebug
                printf("%p %s with %lu elements\n", stacks->data[t], 
                                                    (((ptr_arraylist)(stacks->data[t]))) ? "Valid" : "Not Valid",
                                                    ((ptr_arraylist)(stacks->data[t]))->pointer);
            #endif

            processor_args args = create_processor_args(t, stacks->data[t], pf->cache, 
                                                        &count, &counter_lock,
                                                        &explored_count, &explored_lock,
                                                        &repeated_count, &repeated_lock,
                                                        &saving_counter, checkpoint_file, &file_lock,
                                                        &finished_count, &finished_lock);

            // walker_processor(args);
            pthread_create(thread_id, 0, walker_processor_pre_stacked, (void*)args);
            append_pal(threads, thread_id);
        }

        cache = pf->cache;
    }
    else {
        // #ifdef smallcache
        //     cache = create_hashtable(10, &board_hash);
        // #else
        //     cache = create_hashtable(1000000, &board_hash);
        // #endif
        cache = create_heirarchy(temp_dir);

        checkpoint_filename = (getenv("CHECKPOINT_PATH")) ? getenv("CHECKPOINT_PATH") : checkpoint_filename; // find_temp_filename("checkpoint.bin\0");

        // Distribute the initial states to a set of new pthreads.
        threads = create_ptr_arraylist(procs + 1);

        for(uint64_t t = 0; t < procs; t++) {
            pthread_t* thread_id = (pthread_t*)malloc(sizeof(pthread_t));
            if(!thread_id) err(1, "Memory error while allocating thread id\n");

            processor_args args = create_processor_args(t, search_queue->data[t], cache, 
                                                        &count, &counter_lock,
                                                        &explored_count, &explored_lock,
                                                        &repeated_count, &repeated_lock,
                                                        &saving_counter, checkpoint_file, &file_lock,
                                                        &finished_count, &finished_lock);

            // walker_processor(args);
            pthread_create(thread_id, 0, walker_processor, (void*)args);
            append_pal(threads, thread_id);
        }
    }

    #pragma endregion

    printf("Starting walk...\n");
    printf("Running on %d threads\n", procs);
    printf("Saving checkpoints to %s\n", checkpoint_filename);

    struct statvfs disk_usage_buff;
    const double GB = 1024 * 1024 * 1024;
    if(statvfs("/home", &disk_usage_buff)) err(2, "Finding information about the disk failed\n");
    const double disk_total = (double)(disk_usage_buff.f_blocks * disk_usage_buff.f_frsize) / GB;

    // for(uint64_t t = 0; t < threads->pointer; t++) pthread_join(*(pthread_t*)(threads->data[0]), 0);
    SHUTDOWN_FLAG = 0;
    time_t start = time(0), current, save_timer = time(0), fps_timer = time(0), sleep_timer = time(0), log_timer = time(0);
    clock_t cstart = clock();
    uint32_t cpu_time, cpu_days, cpu_hours, cpu_minutes, cpu_seconds,
             run_time, run_days, run_hours, run_minutes, run_seconds, save_time, previous_run_time = start, fps_update_time, print_sleep;
    uint64_t previous_board_count = 0, fps = 0;
    double disk_avail, disk_used, disk_perc;

    csv_cont csv = create_csv_cont(csv_filename, "%u,%lu,%lu,%lu,%.2f,%.4f,%lu\n", 7);
    struct stat sbuff;
    if(stat(csv_filename, &sbuff)) initialize_file(csv, "runtime", "fps", "found", "explored", "disk_usage", "hash_load_factor", "collisions");

    while(1) {
        if(statvfs("/home", &disk_usage_buff)) err(2, "Finding information about the disk failed\n");
        disk_avail = (double)(disk_usage_buff.f_bfree * disk_usage_buff.f_frsize) / GB;
        disk_used = disk_total - disk_avail;
        disk_perc = (double)(disk_used / disk_total) * (double)100;
        
        current = time(0);
        run_time = current - start;

        #ifdef fastsave
            save_time = (current - save_timer) / 5;
        #else
            save_time = (current - save_timer) / 3600;
        #endif

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
            fps = (explored_count - previous_board_count);
            previous_board_count = explored_count;
            fps_timer = time(0);
        }

        if((current - log_timer) / 60) {
            append_data(csv, run_time, fps, count, explored_count, disk_perc, bin_dict_load_factor(cache->bin_map), cache->collision_count);
            log_timer = time(0);
        }

        #ifndef hideprint
        
        #ifdef slowprint
            if(print_sleep) {
                sleep_timer = time(0);
        #else
            printf("\r");
        #endif
        
            printf("Found %'lu final board states. Explored %'lu boards with %'lu collisions @ %'lu b/s. Runtime: %0d:%02d:%02d:%02d CPU Time: %0d:%02d:%02d:%02d Disk usage: %.2f%% %s", 
                count, explored_count, cache->collision_count, fps,
                run_days, run_hours, run_minutes, run_seconds,
                cpu_days, cpu_hours, cpu_minutes, cpu_seconds,
                disk_perc,
                (save_time) ? "Saving..." : "");

        #ifdef slowprint
                printf("\n");
            }
            else sched_yield();
        #endif
        
        #endif

        if(finished_count == procs) break;

        if(save_time) {
            // SHUTDOWN_FLAG = 1;
            save_progress_v2(checkpoint_file, &file_lock, checkpoint_filename, &saving_counter, cache, count, explored_count, repeated_count, procs - finished_count);
            save_timer = time(0);
        }

        if(SHUTDOWN_FLAG) {
            fflush(stdout);
            save_progress_v2(checkpoint_file, &file_lock, checkpoint_filename, &saving_counter, cache, count, explored_count, repeated_count, procs - finished_count);
            WALKER_KILL_FLAG = 1;
            while(finished_count < procs) sched_yield();
            destroy_heirarchy(cache);
            exit(0);
        }
        else {
            fflush(stdout);
            sched_yield();
        }
    }

    fflush(stdout);
    // save_progress_v2(checkpoint_file, &file_lock, checkpoint_filename, &saving_counter, cache, count, explored_count, repeated_count, procs - finished_count);
    destroy_heirarchy(cache);

    printf("\nThere are %ld possible board states\n", count);
}
