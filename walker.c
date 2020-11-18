#include "walker.h"
#include "ll.h"
#include "arraylist.h"

#include <stdlib.h>
#include <stdio.h>
#include <err.h>

uint8_t SAVING_FLAG = 0;
pthread_mutex_t saving_lock;

coord create_coord(uint8_t row, uint8_t col) {
    coord c = malloc(sizeof(coord_str));
    if(!c) err(1, "Memory error while allocating a coordinate\n");
    c->column = col;
    c->row = row;
    return c;
}

coord* find_next_boards(board b) {
    uint8_t sr = (b->height >> 1) - 1, sc = (b->width >> 1) - 1, visited[b->height][b->width];
    ptr_arraylist edges = create_ptr_arraylist(65), queue = create_ptr_arraylist(65);

    for(uint8_t i = 0; i < b->height; i++) 
        for(uint8_t j = 0; j < b->width; j++) 
            visited[i][j] = 0;

    append_pal(queue, create_coord(sr, sc));

    while(queue->pointer) {
        coord c = pop_front_pal(queue);
        visited[c->row][c->column] = 1;

        for(int8_t rd = -1; rd < 2; rd++) {
            for(int8_t cd = -1; cd < 2; cd++) {
                if(!rd && !cd) continue;
                
                sr = c->row + rd;
                sc = c->column + cd;

                if(sr >= 0 && sr < b->height && sc >= 0 && sc < b->width) {
                    uint8_t v = visited[sr][sc];
                    if(board_get(b, sr, sc) && !v) 
                        append_pal(queue, create_coord(sr, sc));
                    else if(!v && board_get(b, c->row, c->column) != b->player) {
                        visited[sr][sc] = 1;
                        if(board_is_legal_move(b, sr, sc))
                            append_pal(edges, create_coord(sr, sc));
                    }
                }
            }
        }

        free(c);
    }

    for(uint64_t i = 0; i < queue->pointer; i++) free(queue->data[i]);  // In case I missed one?
    destroy_ptr_arraylist(queue);
    coord* result = (coord*)(edges->data);
    free(edges);
    return result;
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
    board starting_board = pargs->starting_board;
    hashtable cache = pargs->cache;
    uint64_t* counter = pargs->counter, *explored = pargs->explored_counter;
    pthread_mutex_t* counter_lock = pargs->counter_lock, *explored_lock = pargs->explored_lock;

    if(starting_board && cache) {
        printf("Processor %d has started\n", pargs->identifier);

        // Setup the stacks
        uint64_t count = 0;
        ptr_arraylist search_stack = create_ptr_arraylist(1000);

        append_pal(search_stack, starting_board);

        printf("Starting walk...\n");

        uint64_t iter = 0, intercap = 0;
        while(search_stack->pointer) {
            board sb = pop_back_pal(search_stack), bc;

            #ifdef debug
                display_board(sb);
            #endif

            coord* next_moves = find_next_boards(sb);

            if(next_moves[0]) {
                uint8_t move_count = 0;
                // If the move is legal, then append it to the search stack
                for(uint8_t im = 0; next_moves[im]; im++) {
                    coord mm = next_moves[im];
                    bc = clone_board(sb);

                    if(board_is_legal_move(bc, mm->row, mm->column)) {
                        board_place_piece(bc, mm->row, mm->column);
                        append_pal(search_stack, bc);
                        move_count++;
                    }
                    else {
                        destroy_board(bc);
                    }

                    free(mm);
                }

                #ifdef debug
                    printf("Found %u moves\n", move_count);
                #endif

                free(next_moves);
            }
            else {
                // The opponenet has no moves, try the other player
                #ifdef debug
                    printf("No moves for opponent, switching back to the current player\n");
                #endif
                sb->player = (sb->player == 1) ? 2 : 1;

                free(next_moves);
                next_moves = find_next_boards(sb);

                if(next_moves[0]) {
                    uint8_t move_count = 0;
                    // If the move is legal, then append it to the search stack
                    for(uint8_t im = 0; next_moves[im]; im++) {
                        coord mm = next_moves[im];
                        bc = clone_board(sb);

                        if(board_is_legal_move(bc, mm->row, mm->column)) {
                            board_place_piece(bc, mm->row, mm->column);
                            append_pal(search_stack, bc);
                            move_count++;
                        }
                        else {
                            destroy_board(bc);
                        }

                        free(mm);
                    }

                    free(next_moves);
                }
                else {
                    // The opponenet has no moves, try the other player
                    #ifdef debug
                        printf("No moves for anybody, game has ended.\n");
                    #endif

                    free(next_moves);

                    if(!exists_hs(cache, sb)) {
                        put_hs(cache, sb);
                        while(pthread_mutex_trylock(counter_lock)) sched_yield();
                        *counter += 1;
                        // *explored += count;
                        pthread_mutex_unlock(counter_lock);

                        // while(pthread_mutex_trylock(explored_lock)) sched_yield();
                        // *explored += count;
                        // pthread_mutex_unlock(explored_lock);

                        count = 0;
                    }
                    else {
                        #ifdef debug
                            printf("The given board is already counted\n");
                        #endif
                    }
                }
            }

            while(pthread_mutex_trylock(explored_lock)) sched_yield();
            *explored += 1;
            pthread_mutex_unlock(explored_lock);

            destroy_board(sb);

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
        }

        printf("Processor %d has finished\n", pargs->identifier);

        return 0;
    }
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

processor_args create_processor_args(uint32_t identifier, board starting_board, hashtable cache, 
                                     uint64_t* counter, pthread_mutex_t* counter_lock,
                                     uint64_t* explored_counter, pthread_mutex_t* explored_lock,
                                     uint64_t* saving_counter, FILE** checkpoint_file, pthread_mutex_t* file_lock) {
    processor_args args = malloc(sizeof(processor_args_str));
    if(!args) err(1, "Memory error while allocating processor args\n");

    args->identifier = identifier;
    args->starting_board = starting_board;
    args->cache = cache;
    args->counter = counter;
    args->counter_lock = counter_lock;
    args->explored_counter = explored_counter;
    args->explored_lock = explored_lock;
    args->checkpoint_file = checkpoint_file;
    args->file_lock = file_lock;
    args->saving_counter = saving_counter;

    return args;
}

void walker_to_file(FILE* fp, ptr_arraylist search_stack) {
    if(search_stack) {
        __uint128_t result;

        for(board* ptr = (board*)search_stack->data; *ptr; ptr++) {
            board b = *ptr;

            result = 0;

            fwrite(&b->player, sizeof(uint8_t), 1, fp);
            result = result << 2;

            // NO YOU CAN'T, but luckily, I don't actually need the player
            // You can fit 2 spaces in 3 bits if you really try,
            // so on an 8x8 board, 
            // we end up only using 96 bits instead of the entire 128.
            // well
            for(uint8_t r = 0; r < b->height; r++) {
                for(uint8_t c = 0; c < b->width; c++) {
                    uint8_t s1 = board_get(b, r, c);

                    result += s1;

                    if(c < b->width - 1) result = result << 2;
                }
            }

            fwrite(&result, sizeof(__uint128_t), 1, fp);
        }

        result = 0;

        fwrite(&result, sizeof(__uint128_t), 1, fp);
    }
}
