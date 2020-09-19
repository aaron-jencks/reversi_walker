#include "walker.h"
#include "ll.h"

#include <stdlib.h>
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
    linkedlist edges = create_ll(), queue = create_ll();

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
                            append_ll(edges, create_coord(sr, sc));
                    }
                }
            }
        }

        free(c);
    }

    destroy_ll(queue);
    coord* result = (coord*)ll_to_arr(edges);
    destroy_ll(edges);
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