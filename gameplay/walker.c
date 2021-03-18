#include "walker.h"
#include "../utils/ll.h"
#include "../utils/arraylist.h"
#include "../hashing/hash_functions.h"

#include <stdlib.h>
#include <stdio.h>
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

void display_moves_w(board b, ptr_arraylist coords) {
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
    coord c = malloc(sizeof(coord_str));
    if(!c) err(1, "Memory error while allocating a coordinate\n");
    c->column = col;
    c->row = row;
    return c;
}

void find_next_boards(board b, ptr_arraylist edges, ptr_arraylist coord_cache) {
    uint8_t sr = (b->height >> 1) - 1, sc = (b->width >> 1) - 1, visited[b->height][b->width];
    edges->pointer = 0;

    // 6 times faster
    for(uint8_t i = 0; i < b->height; i++) {
        for(uint8_t j = 0; j < b->width; j++) {
            if(board_is_legal_move(b, i, j)) {
                coord c = pop_back_pal(coord_cache);
                c->column = j;
                c->row = i;
                append_pal(edges, c);
            }
        }
    }
}


coord* find_next_boards_from_coord(board b, coord c) {
    uint8_t sr, sc;
    // TODO Use an arraylist here
    linkedlist edges = create_ll();

    for(int8_t rd = -1; rd < 2; rd++) {
        for(int8_t cd = -1; cd < 2; cd++) {
            if(!rd && !cd) continue;
            
            sr = c->row + rd;
            sc = c->column + cd;

            if(sr >= 0 && sr < b->height && sc >= 0 && sc < b->width) {
                if(board_get(b, c->row, c->column) != b->player) {
                    if(board_is_legal_move(b, sr, sc))
                        append_ll(edges, create_coord(sr, sc));
                }
            }
        }
    }

    coord* result = (coord*)ll_to_arr(edges);
    destroy_ll(edges);
    return result;
}

coord* find_next_boards_from_coord_opposing_player(board b, coord c) {
    uint8_t sr, sc;
    ptr_arraylist edges = create_ptr_arraylist(9);

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
                else if(!bv && found) append_pal(edges, create_coord(sr, sc));
                else continue;
            }
        }
    }

    coord* result = (coord*)edges->data;
    free(edges);
    return result;
}

void* walker_processor(void* args) {
    // Unpack the arguments
    processor_args pargs = (processor_args)args;
    if(pargs && pargs->starting_board) {
        ptr_arraylist search_stack = create_ptr_arraylist(10000);
        append_pal(search_stack, pargs->starting_board);
        pargs->starting_board = (board)search_stack;
        return walker_processor_pre_stacked(pargs);
    }
    return (void*)1;
}

void* walker_processor_pre_stacked(void* args) {
    // Unpack the arguments
    processor_args pargs = (processor_args)args;
    ptr_arraylist starting_stack = (ptr_arraylist)pargs->starting_board;
    heirarchy cache = pargs->cache;
    uint64_t* counter = pargs->counter, *explored = pargs->explored_counter, *repeated = pargs->repeated_counter;
    pthread_mutex_t* counter_lock = pargs->counter_lock, *explored_lock = pargs->explored_lock, *repeated_lock = pargs->repeated_lock;

    if(starting_stack && cache) {
        printf("Processor %d has started\n", pargs->identifier);

        // Setup the stacks
        uint64_t count = 0;
        ptr_arraylist search_stack = starting_stack;
        ptr_arraylist board_cache = create_ptr_arraylist(1000), coord_cache = create_ptr_arraylist(1000), coord_buff = create_ptr_arraylist(65);
        for(size_t bc = 0; bc < 1000; bc++) {
            append_pal(board_cache, create_board(1, 6, 6));
            append_pal(coord_cache, create_coord(0, 0));
        }

        // printf("Starting walk...\n");

        uint64_t iter = 0, intercap = 0;
        while(search_stack->pointer) {
            board sb = pop_back_pal(search_stack), bc;

            // if(heirarchy_insert(cache, board_spiral_hash(sb))) {

                #ifdef debug
                    __uint128_t hash = board_spiral_hash(sb);
                    printf("Board hashed to %lu %lu\n", ((uint64_t*)&hash)[1], ((uint64_t*)&hash)[0]);
                #endif

                find_next_boards(sb, coord_buff, coord_cache);

                #ifdef debug
                    display_moves_w(sb, coord_buff);
                #endif

                if(coord_buff->pointer) {
                    uint8_t move_count = 0;
                    // If the move is legal, then append it to the search stack
                    for(uint8_t im = 0; im < coord_buff->pointer; im++) {
                        coord mm = coord_buff->data[im];

                        if(board_cache->pointer) bc = pop_back_pal(board_cache);
                        else bc = create_board(1, sb->height, sb->width);

                        clone_into_board(sb, bc);

                        if(board_is_legal_move(bc, mm->row, mm->column)) {
                            board_place_piece(bc, mm->row, mm->column);
                            append_pal(search_stack, bc);
                            move_count++;
                        }
                        else {
                            append_pal(board_cache, bc);
                        }

                        append_pal(coord_cache, mm);
                    }

                    #ifdef debug
                        // printf("Found %u moves\n", move_count);
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
                            coord mm = coord_buff->data[im];
                            
                            if(board_cache->pointer) bc = pop_back_pal(board_cache);
                            else bc = create_board(1, sb->height, sb->width);

                            clone_into_board(sb, bc);

                            if(board_is_legal_move(bc, mm->row, mm->column)) {
                                board_place_piece(bc, mm->row, mm->column);
                                append_pal(search_stack, bc);
                                move_count++;
                            }
                            else {
                                append_pal(board_cache, bc);
                            }

                            append_pal(coord_cache, mm);
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

                        if(heirarchy_insert(cache, board_spiral_hash(sb))) {

                            // printf("Found a new board to count\n");
                            while(pthread_mutex_trylock(counter_lock)) sched_yield();
                            *counter += 1;
                            // *explored += count;
                            pthread_mutex_unlock(counter_lock);

                            // while(pthread_mutex_trylock(explored_lock)) sched_yield();
                            // *explored += count;
                            // pthread_mutex_unlock(explored_lock);

                            // count = 0;
                        }
                        // else {
                        //     while(pthread_mutex_trylock(repeated_lock)) sched_yield();
                        //     *repeated += 1;
                        //     pthread_mutex_unlock(repeated_lock);
                        // }
                    }
                }

                while(pthread_mutex_trylock(explored_lock)) sched_yield();
                *explored += 1;
                pthread_mutex_unlock(explored_lock);
            // }
            // else {
            //     #ifdef debug
            //         __uint128_t hash = board_spiral_hash(sb);
            //         printf("Board hashed to %lu %lu <-- REPEAT\n", ((uint64_t*)&hash)[1], ((uint64_t*)&hash)[0]);
            //     #endif
            // }
            append_pal(board_cache, sb);

            if(SAVING_FLAG) {
                while(pthread_mutex_trylock(pargs->file_lock)) sched_yield();

                #ifdef debug
                    printf("Saving thread\n");
                #endif

                walker_to_file(*(pargs->checkpoint_file), search_stack);
                *(pargs->saving_counter) += 1;
                pthread_mutex_unlock(pargs->file_lock);

                #ifdef debug
                    printf("Finished saving thread\n");
                #endif

                uint8_t temp_saving_flag = SAVING_FLAG;
                while(temp_saving_flag) {
                    while(pthread_mutex_trylock(&saving_lock)) sched_yield();
                    temp_saving_flag = SAVING_FLAG;
                    pthread_mutex_unlock(&saving_lock);
                    sched_yield();
                }
            }

            if(WALKER_KILL_FLAG) break;
        }

        printf("Processor %d has finished\n", pargs->identifier);

        while(pthread_mutex_trylock(pargs->finished_lock)) sched_yield();
        *pargs->finished_count += 1;
        pthread_mutex_unlock(pargs->finished_lock);

        free(pargs);
        while(search_stack->pointer) destroy_board(pop_back_pal(search_stack));
        destroy_ptr_arraylist(search_stack);

        while(board_cache->pointer) {
            destroy_board(pop_back_pal(board_cache));
        }
        destroy_ptr_arraylist(board_cache);

        while(coord_cache->pointer) {
            free(pop_back_pal(coord_cache));
        }
        destroy_ptr_arraylist(coord_cache);

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

processor_args create_processor_args(uint32_t identifier, board starting_board, heirarchy cache, 
                                     uint64_t* counter, pthread_mutex_t* counter_lock,
                                     uint64_t* explored_counter, pthread_mutex_t* explored_lock,
                                     uint64_t* repeated_counter, pthread_mutex_t* repeated_lock,
                                     uint64_t* saving_counter, FILE** checkpoint_file, pthread_mutex_t* file_lock,
                                     size_t* finished_count, pthread_mutex_t* finished_lock) {
    processor_args args = malloc(sizeof(processor_args_str));
    if(!args) err(1, "Memory error while allocating processor args\n");

    args->identifier = identifier;
    args->starting_board = starting_board;
    args->cache = cache;
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

void walker_to_file(FILE* fp, ptr_arraylist search_stack) {
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
