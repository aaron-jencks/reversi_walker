#include "walker.hpp"
#include "../utils/ll.h"
#include "../utils/tarraylist.hpp"
#include "../hashing/hash_functions.h"
#include "./reversi_defs.h"
#include "../mem_man/heir.hpp"
#include "../utils/pqueue.hpp"
#include "../project_defs.hpp"
#include "../utils/fileio.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <sys/sysinfo.h>
#include <err.h>

#ifdef debug
void display_board_w(board b) {
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

void display_moves_w(board b, Arraylist<void*>* coords) {
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
#endif

uint8_t SAVING_FLAG = 0, WALKER_KILL_FLAG = 0;
pthread_mutex_t saving_lock;

coord create_coord(uint8_t row, uint8_t col) {
    coord c = (coord)malloc(sizeof(coord_str));
    if(!c) err(1, "Memory error while allocating a coordinate\n");
    c->column = col;
    c->row = row;
    return c;
}

void find_next_boards(board b, Arraylist<void*>* edges, Arraylist<void*>* coord_cache) {
    uint8_t sr = (b->height >> 1) - 1, sc = (b->width >> 1) - 1, visited[b->height][b->width];
    edges->pointer = 0;

    // 6 times faster
    for(uint8_t i = 0; i < b->height; i++) {
        for(uint8_t j = 0; j < b->width; j++) {
            if(board_is_legal_move(b, i, j)) {
                coord c = ((coord_cache->pointer) ? (coord)coord_cache->pop_back() : create_coord(i, j));
                c->column = j;
                c->row = i;
                edges->append(c);
            }
        }
    }

    // printf("Found %lu moves\n", edges->pointer);
}


// coord* find_next_boards_from_coord(board b, coord c) {
//     uint8_t sr, sc;
//     // TODO Use an arraylist here
//     linkedlist edges = create_ll();

//     for(int8_t rd = -1; rd < 2; rd++) {
//         for(int8_t cd = -1; cd < 2; cd++) {
//             if(!rd && !cd) continue;
            
//             sr = c->row + rd;
//             sc = c->column + cd;

//             if(sr >= 0 && sr < b->height && sc >= 0 && sc < b->width) {
//                 if(board_get(b, c->row, c->column) != b->player) {
//                     if(board_is_legal_move(b, sr, sc))
//                         append_ll(edges, create_coord(sr, sc));
//                 }
//             }
//         }
//     }

//     coord* result = (coord*)ll_to_arr(edges);
//     destroy_ll(edges);
//     return result;
// }

coord* find_next_boards_from_coord_opposing_player(board b, coord c) {
    uint8_t sr, sc;
    Arraylist<void*>* edges = new Arraylist<void*>(9);

    uint8_t found, bv;
    for(int8_t rd = -1; rd < 2; rd++) {
        for(int8_t cd = -1; cd < 2; cd++) {
            found = 0;
            if(!rd && !cd) continue;
            
            sr = c->row + rd;
            sc = c->column + cd;

            if(sr >= 0 && sr < b->height && sc >= 0 && sc < b->width) {
                bv = board_get(b, c->row, c->column);
                if(bv == b->player) found = 1;
                else if(!bv && found) edges->append(create_coord(sr, sc));
                else continue;
            }
        }
    }

    coord* result = (coord*)edges->data;
    free(edges);
    return result;
}

// void* walker_processor(void* args) {
//     // Unpack the arguments
//     processor_args pargs = (processor_args)args;
//     if(pargs) {
//         Arraylist<void*>* search_stack = new Arraylist<void*>(10000);
//         return walker_processor_pre_stacked(pargs);
//     }
//     return (void*)1;
// }

void* walker_processor(void* args) {
    // Unpack the arguments
    processor_args pargs = (processor_args)args;
    heirarchy cache = pargs->cache;
    uint64_t* counter = pargs->counter, *explored = pargs->explored_counter, *repeated = pargs->repeated_counter;
    pthread_mutex_t* counter_lock = pargs->counter_lock, *explored_lock = pargs->explored_lock, *repeated_lock = pargs->repeated_lock;

    if(cache) {
        printf("Processor %d has started\n", pargs->identifier);

        // Setup the stacks
        uint64_t count = 0;
        Arraylist<void*>* board_cache = new Arraylist<void*>(1000), *coord_cache = new Arraylist<void*>(1000), *coord_buff = new Arraylist<void*>(65);
        for(size_t bc = 0; bc < 1000; bc++) {
            board_cache->append(create_board(1, BOARD_HEIGHT, BOARD_WIDTH, 0));
            coord_cache->append(create_coord(0, 0));
        }

        // printf("Starting walk...\n");

        uint64_t iter = 0, intercap = 0;
        uint8_t running = 0, starved = 0;
        while(!WALKER_KILL_FLAG) {
            if(pargs->inputq->count) {
                if(!running) running = 1;
                if(starved) starved = 0;
                board sb = pargs->inputq->pop_front(), bc;

                // #ifdef debug
                //     display_board(sb);
                // #endif

                // find_next_boards(sb, coord_buff, coord_cache);

                #ifdef incore_cache

                if(heirarchy_insert_cache(cache, board_fast_hash_6(sb))) {

                #endif

                    find_next_boards(sb, coord_buff, coord_cache);

                    #ifdef debug
                        __uint128_t hash = board_spiral_hash(sb);
                        printf("Board hashed to %lu %lu\n", ((uint64_t*)&hash)[1], ((uint64_t*)&hash)[0]);
                        display_moves_w(sb, coord_buff);
                    #endif

                    if(coord_buff->pointer) {
                        uint8_t move_count = 0;
                        // If the move is legal, then append it to the search stack
                        for(uint8_t im = 0; im < coord_buff->pointer; im++) {
                            coord mm = (coord)coord_buff->data[im];
                            
                            if(board_cache->pointer) bc = (board)board_cache->pop_back();
                            else bc = create_board(1, sb->height, sb->width, sb->level + 1);

                            clone_into_board(sb, bc);

                            if(board_is_legal_move(bc, mm->row, mm->column)) {
                                board_place_piece(bc, mm->row, mm->column);
                                pargs->outputq->push(bc);
                                move_count++;
                            }
                            else {
                                board_cache->append(bc);
                            }

                            coord_cache->append(mm);
                        }

                        #ifdef debug
                            printf("Found %u moves\n", move_count);
                        #endif
                    }
                    else {
                        // The opponenet has no moves, try the other player
                        #ifdef debug
                            printf("No moves for opponent, switching back to the current player\n");
                        #endif
                        sb->player = (sb->player == 1) ? 2 : 1;

                        find_next_boards(sb, coord_buff, coord_cache);

                        if(coord_buff->pointer) {
                            uint8_t move_count = 0;
                            // If the move is legal, then append it to the search stack
                            for(uint8_t im = 0; im < coord_buff->pointer; im++) {
                                coord mm = (coord)coord_buff->data[im];
                                
                                if(board_cache->pointer) bc = (board)board_cache->pop_back();
                                else bc = create_board(1, sb->height, sb->width, sb->level);

                                clone_into_board(sb, bc);

                                if(board_is_legal_move(bc, mm->row, mm->column)) {
                                    board_place_piece(bc, mm->row, mm->column);
                                    pargs->outputq->push(bc);
                                    move_count++;
                                }
                                else {
                                    board_cache->append(bc);
                                }

                                coord_cache->append(mm);
                            }
                        }
                        else {
                            // The opponenet has no moves, try the other player
                            #ifdef debug
                                printf("No moves for anybody, game has ended.\n");
                            #endif

                            // if(!exists_hs(cache, sb)) {
                            //     put_hs(cache, sb);
                            //     while(pthread_mutex_trylock(counter_lock)) sched_yield();
                            //     *counter += 1;
                            //     // *explored += count;
                            //     pthread_mutex_unlock(counter_lock);

                            //     // while(pthread_mutex_trylock(explored_lock)) sched_yield();
                            //     // *explored += count;
                            //     // pthread_mutex_unlock(explored_lock);

                            //     count = 0;
                            // }
                            // else {
                            //     #ifdef debug
                            //         printf("The given board is already counted\n");
                            //     #endif
                            // }

                            // TODO move this into the scheduler
                            // if(heirarchy_insert(cache, board_fast_hash_6(sb), sb->level)) {

                            //     fprintf(stderr, "Found a board\n");

                            //     // printf("Found a new board to count\n");
                            //     while(pthread_mutex_trylock(counter_lock)) sched_yield();
                            //     *counter += 1;
                            //     // *explored += count;
                            //     pthread_mutex_unlock(counter_lock);

                            //     // while(pthread_mutex_trylock(explored_lock)) sched_yield();
                            //     // *explored += count;
                            //     // pthread_mutex_unlock(explored_lock);

                            //     // count = 0;
                            // }
                            // else {
                            //     fprintf(stderr, "Found a repeated board\n");
                            //     while(pthread_mutex_trylock(repeated_lock)) sched_yield();
                            //     *repeated += 1;
                            //     pthread_mutex_unlock(repeated_lock);
                            // }
                        }
                    }

                    while(pthread_mutex_trylock(explored_lock)) sched_yield();
                    *explored += 1;
                    pthread_mutex_unlock(explored_lock);

                #ifdef incore_cache

                }
                else {
                    #ifdef debug
                        __uint128_t hash = board_spiral_hash(sb);
                        printf("Board hashed to %lu %lu <-- REPEAT\n", ((uint64_t*)&hash)[1], ((uint64_t*)&hash)[0]);
                    #endif
                }

                #endif
                
                board_cache->append(sb);

                // if(SAVING_FLAG) {
                //     while(pthread_mutex_trylock(pargs->file_lock)) sched_yield();

                //     #ifdef debug
                //         printf("Saving thread\n");
                //     #endif

                //     walker_to_file(*(pargs->checkpoint_file), pargs->inputq);
                //     *(pargs->saving_counter) += 1;
                //     pthread_mutex_unlock(pargs->file_lock);

                //     #ifdef debug
                //         printf("Finished saving thread\n");
                //     #endif

                //     uint8_t temp_saving_flag = SAVING_FLAG;
                //     while(temp_saving_flag) {
                //         while(pthread_mutex_trylock(&saving_lock)) sched_yield();
                //         temp_saving_flag = SAVING_FLAG;
                //         pthread_mutex_unlock(&saving_lock);
                //         sched_yield();
                //     }
                // }
            }
            else if(running) {
                // We've finished the current batch
                printf("Processor %d has finished it's batch\n", pargs->identifier);

                while(pthread_mutex_trylock(pargs->finished_lock)) sched_yield();
                *pargs->finished_count += 1;
                pthread_mutex_unlock(pargs->finished_lock);

                running = 0;
            }
            else if(!starved) {
                fprintf(stderr, "Processor %d is being starved\n", pargs->identifier);
                starved = 1;
            }
        }

        printf("Processor %d has finished\n", pargs->identifier);

        while(pthread_mutex_trylock(pargs->finished_lock)) sched_yield();
        *pargs->finished_count += 1;
        pthread_mutex_unlock(pargs->finished_lock);

        free(pargs);
        delete board_cache, coord_cache, coord_buff;
        // while(pargs->inputq->count) destroy_board(pargs->inputq->pop_back());

        return 0;
    }

    printf("Either the cache or stack is 0\n");
    return (void*)1;
}

uint16_t coord_to_short(coord c) {
    return coord_to_short_ints(c->row, c->column);
}

uint16_t coord_to_short_ints(uint8_t row, uint8_t column) {
    uint16_t r = 0;
    r += row;
    r = (r << 8) + column;
    return r;
}

coord short_to_coord(uint16_t s) {
    coord r = create_coord(s >> 8, s);
    return r;
}

processor_args create_processor_args(uint32_t identifier, LockedRingBuffer<board>* inputq, LockedPriorityQueue<board>* outputq, 
                                     heirarchy cache, Arraylist<board>* board_cache, Arraylist<coord>* coord_cache, 
                                     uint64_t* counter, pthread_mutex_t* counter_lock,
                                     uint64_t* explored_counter, pthread_mutex_t* explored_lock,
                                     uint64_t* repeated_counter, pthread_mutex_t* repeated_lock,
                                     uint64_t* saving_counter, FILE** checkpoint_file, pthread_mutex_t* file_lock,
                                     size_t* finished_count, pthread_mutex_t* finished_lock) {
    processor_args args = (processor_args)malloc(sizeof(processor_args_str));
    if(!args) err(1, "Memory error while allocating processor args\n");

    args->identifier = identifier;
    args->inputq = inputq;
    args->outputq = outputq;
    args->cache = cache;
    args->board_cache = board_cache;
    args->coord_cache = coord_cache;
    args->counter = counter;
    args->counter_lock = counter_lock;
    args->explored_counter = explored_counter;
    args->explored_lock = explored_lock;
    args->repeated_counter = repeated_counter;
    args->repeated_lock = repeated_lock;
    args->checkpoint_file = checkpoint_file;
    args->file_lock = file_lock;
    args->saving_counter = saving_counter;
    args->finished_count = finished_count;
    args->finished_lock = finished_lock;

    return args;
}

processor_scheduler_args_t* create_processor_scheduler_args(heirarchy cache, LockedPriorityQueue<board>* inputq, size_t nprocs, 
                                                            uint64_t* counter, pthread_mutex_t* counter_lock,
                                                            uint64_t* explored_counter, pthread_mutex_t* explored_lock,
                                                            uint64_t* repeated_counter, pthread_mutex_t* repeated_lock,
                                                            uint64_t* saving_counter, char* checkpoint_file, pthread_mutex_t* file_lock,
                                                            size_t* finished_count, pthread_mutex_t* finished_lock) {
    processor_scheduler_args_t* args = (processor_scheduler_args_t*)malloc(sizeof(processor_scheduler_args_t));
    if(!args) err(1, "Memory error while allocating processor scheduler args\n");

    args->cache = cache;
    args->inputq = inputq;
    args->nprocs = nprocs;
    args->counter = counter;
    args->counter_lock = counter_lock;
    args->explored_counter = explored_counter;
    args->explored_lock = explored_lock;
    args->repeated_counter = repeated_counter;
    args->repeated_lock = repeated_lock;
    args->checkpoint_file = checkpoint_file;
    args->file_lock = file_lock;
    args->saving_counter = saving_counter;
    args->finished_count = finished_count;
    args->finished_lock = finished_lock;

    return args;
}

void walker_to_file(FILE* fp, Arraylist<void*>* search_stack) {
    if(search_stack) {
        __uint128_t result;

        #ifdef filedebug
            printf("Writing %lu boards to file from processor\n", search_stack->pointer);
        #endif

        // fwrite(&search_stack->pointer, sizeof(search_stack->pointer), 1, fp);
        for(board* ptr = (board*)search_stack->data; *ptr; ptr++) {
            board b = *ptr;

            result = 0;

            if(!fwrite(&b->player, sizeof(uint8_t), 1, fp)) err(8, "Writing to the file failed\n");
            // result = result << 2;

            // NO YOU CAN'T, but luckily, I don't actually need the player
            // You can fit 2 spaces in 3 bits if you really try,
            // so on an 8x8 board, 
            // we end up only using 96 bits instead of the entire 128.
            // well
            for(uint8_t r = 0; r < b->height; r++) {
                for(uint8_t c = 0; c < b->width; c++) {
                    uint8_t s1 = board_get(b, r, c);

                    result += s1;

                    if(c < (b->width - 1) || r < (b->height - 1)) result = result << 2;
                }
            }

            #ifdef filedebug
                printf("Writing board %u: %lu %lu\n", b->player, ((uint64_t*)(&result))[1], ((uint64_t*)(&result))[0]);
                display_board_w(b);
            #endif

            fwrite(&result, sizeof(__uint128_t), 1, fp);
        }

        result = 0;
        uint8_t spaceholder = 0;

        fwrite(&spaceholder, sizeof(uint8_t), 1, fp);
        fwrite(&result, sizeof(__uint128_t), 1, fp);
    }
}

void* walker_task_scheduler(void* args) {
    processor_scheduler_args_t* pargs = (processor_scheduler_args_t*)args;

    pthread_mutex_t level_lock;
    if(pthread_mutex_init(&level_lock, 0)) err(4, "Failed to initialize level lock mutex\n");

    size_t level_counts[64];
    for(size_t l = 0; l < 64; l++) { level_counts[l] = 0; }

    // Distribute the initial states to a set of new pthreads.
    Arraylist<void*>* threads = new Arraylist<void*>(pargs->nprocs + 1);
    Arraylist<LockedRingBuffer<board>*>* procqs = new Arraylist<LockedRingBuffer<board>*>(pargs->nprocs + 1);
    LockedArraylist<board>* board_cache = new LockedArraylist<board>(1000 * pargs->nprocs);
    LockedArraylist<coord>* coord_cache = new LockedArraylist<coord>(1000 * pargs->nprocs);

    FILE** checkpoint_file_pointer;
    FILE* file_pointer;
    checkpoint_file_pointer = &file_pointer;

    for(uint64_t t = 0; t < pargs->nprocs; t++) {
        pthread_t* thread_id = (pthread_t*)malloc(sizeof(pthread_t));
        if(!thread_id) err(1, "Memory error while allocating thread id\n");

        LockedRingBuffer<board>* inputq = new LockedRingBuffer<board>(1000);
        if(!inputq) err(1, "Memory error while allocating locked ring buffer\n");

        processor_args args = create_processor_args(t, inputq, pargs->inputq, 
                                                    pargs->cache, board_cache, coord_cache, 
                                                    pargs->counter, pargs->counter_lock,
                                                    pargs->explored_counter, pargs->explored_lock,
                                                    pargs->repeated_counter, pargs->repeated_lock,
                                                    pargs->saving_counter, checkpoint_file_pointer, pargs->file_lock,
                                                    pargs->finished_count, pargs->finished_lock);

        // walker_processor(args);
        pthread_create(thread_id, 0, walker_processor, (void*)args);

        threads->append(thread_id);
        procqs->append(inputq);
    }

    time_t current, save_timer = time(0);
    uint32_t save_time;

    size_t current_target = 0, current_level = 0;
    #ifdef debug
        printf("Starting scheduler\n");
    #endif
    while((pargs->inputq->count || *pargs->finished_count < pargs->nprocs) && !WALKER_KILL_FLAG) {
        size_t assigned_procs = 0;

        while(pargs->inputq->count) {
            board* b = pargs->inputq->pop_bulk(CHUNK_SIZE * pargs->nprocs);

            size_t actual_count;
            for(actual_count = 0; actual_count < CHUNK_SIZE * pargs->nprocs && b[actual_count]; actual_count++) {
                // Purge the previous level
                #ifdef debug
                    printf("Found board at level %u\n", b[actual_count]->level);
                #endif
                if(b[actual_count]->level > current_level) {
                    #ifdef debug
                        printf("Saving current level\n");
                    #endif
                    save_progress_v3(checkpoint_file_pointer, pargs->file_lock, pargs->checkpoint_file, pargs->cache, 
                        current_level, pargs->cache->level_mappings[current_level]->data, pargs->cache->level_mappings[current_level]->pointer,
                        *pargs->counter, *pargs->explored_counter, *pargs->repeated_counter);
                    heirarchy_purge_level(pargs->cache, current_level++);
                }
            }

            fprintf(stderr, "Found %lu boards in the chunk\n", actual_count);

            if(actual_count < CHUNK_SIZE) {
                procqs->data[current_target++]->append_bulk(b, actual_count);
                if(current_target >= procqs->pointer) current_target = 0;
                if(++assigned_procs > procqs->pointer) assigned_procs = procqs->pointer;
            }
            else {
                size_t chunk_count = actual_count >> 5;
                if(actual_count % CHUNK_SIZE) chunk_count++;

                for(size_t c = 0; c < chunk_count; c++) {
                    procqs->data[current_target++]->append_bulk(b, 
                        (c < (chunk_count - 1) && actual_count % CHUNK_SIZE) ? 
                            CHUNK_SIZE : actual_count % CHUNK_SIZE);

                    if(current_target >= procqs->size) current_target = 0;
                    if(++assigned_procs > procqs->pointer) assigned_procs = procqs->pointer;
                }
            }

            free(b);
        }

        #ifdef debug
            printf("Scheduler waiting for processors to finish batch\n");
        #endif
        while(*pargs->finished_count < assigned_procs) sched_yield();
        *pargs->finished_count = 0;

        #ifdef debug
            printf("Scheduler moving to next batch\n");
        #endif

        // fprintf(stderr, "The scheduler ran out of boards\n");

        // #ifdef fastsave
        //     save_time = (current - save_timer) / 5;
        // #else
        //     save_time = (current - save_timer) / 3600;
        // #endif

        // if(save_time) {
        //     printf(" Saving...\n");
        //     // SHUTDOWN_FLAG = 1;
        //     save_progress_v3(checkpoint_file_pointer, pargs->file_lock, pargs->checkpoint_file, pargs->cache, 
        //                 current_level, pargs->cache->level_mappings[current_level]->data, pargs->cache->level_mappings[current_level]->pointer,
        //                 *pargs->counter, *pargs->explored_counter, *pargs->repeated_counter);
        //     save_timer = time(0);
        // }
        
        sched_yield();
    }

    for(size_t i = 0; i < procqs->pointer; i++) {
        while(procqs->data[i]->count) destroy_board(procqs->data[i]->pop_back());
        delete procqs->data[i];
    }
    while(board_cache->pointer) destroy_board(board_cache->pop_back());
    while(coord_cache->pointer) free(coord_cache->pop_back());
    delete threads, procqs, board_cache, coord_cache;

    free(pargs);

    return nullptr;
}
