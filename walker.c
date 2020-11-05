#include "walker.h"
#include "ll.h"
#include "arraylist.h"

#include <stdlib.h>
#include <stdio.h>
#include <err.h>

coord create_coord(uint8_t row, uint8_t col) {
    coord c = malloc(sizeof(coord_str));
    if(!c) err(1, "Memory error while allocating a coordinate\n");
    c->column = col;
    c->row = row;
    return c;
}

coord* find_next_boards(board b) {
    uint8_t sr = (b->height >> 1) - 1, sc = (b->width >> 1) - 1, visited[b->height][b->width];
    ptr_arraylist edges = create_ptr_arraylist(65);
    linkedlist queue = create_ll();

    for(uint8_t i = 0; i < b->height; i++) 
        for(uint8_t j = 0; j < b->width; j++) 
            visited[i][j] = 0;

    append_ll(queue, create_coord(sr, sc));
    while(queue->size) {
        coord c = pop_front_ll(queue);
        visited[c->row][c->column] = 1;

        for(int8_t rd = -1; rd < 2; rd++) {
            for(int8_t cd = -1; cd < 2; cd++) {
                if(!rd && !cd) continue;
                
                sr = c->row + rd;
                sc = c->column + cd;

                if(sr >= 0 && sr < b->height && sc >= 0 && sc < b->width) {
                    uint8_t v = visited[sr][sc];
                    if(board_get(b, sr, sc) && !v) 
                        append_ll(queue, create_coord(sr, sc));
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

    destroy_ll(queue);
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
    uint64_t* counter = pargs->counter;
    pthread_mutex_t* counter_lock = pargs->counter_lock;

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
                        count++;
                        while(pthread_mutex_trylock(counter_lock)) sched_yield();
                        (*counter)++;
                        pthread_mutex_unlock(counter_lock);
                    }
                    else {
                        #ifdef debug
                            printf("The given board is already counted\n");
                        #endif
                    }
                }
            }

            destroy_board(sb);
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

processor_args create_processor_args(uint32_t identifier, board starting_board, hashtable cache, uint64_t* counter, pthread_mutex_t* counter_lock) {
    processor_args args = malloc(sizeof(processor_args_str));
    if(!args) err(1, "Memory error while allocating processor args\n");

    args->identifier = identifier;
    args->starting_board = starting_board;
    args->cache = cache;
    args->counter = counter;
    args->counter_lock = counter_lock;

    return args;
}