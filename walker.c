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
    unsigned char sr = (b->height >> 1) - 1, sc = (b->width >> 1) - 1, psr, psc, visited[b->height][b->width];
    linkedlist edges = create_ll(), queue = create_ll();

    for(char i = 0; i < b->height; i++) 
        for(char j = 0; j < b->width; j++) 
            visited[i][j] = 0;

    append_ll(queue, create_coord(sr, sc));
    while(queue->size) {
        coord c = pop_front_ll(queue);
        visited[c->row][c->column] = 1;

        for(signed char rd = -1; rd < 2; rd++) {
            for(signed char cd = -1; cd < 2; cd++) {
                psr = sr;
                psc = sc;
                sr = c->row + rd;
                sc = c->column + cd;

                if(sr >= 0 && sr < 8 && sc >= 0 && sc < 8) {
                    if(board_get(b, sr, sc) && !visited[sr][sc]) 
                        append_ll(queue, create_coord(sr, sc));
                    else if(!visited[sr][sc] && board_get(b, psr, psc) != b->player) {
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
    coord* result = ll_to_arr(edges);
    destroy_ll(edges);
    return result;
}