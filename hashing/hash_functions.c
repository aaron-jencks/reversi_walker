#include "./hash_functions.h"

#include "./lookup3.h"
#include "../gameplay/reversi.h"


__uint128_t board_hash(void* brd) {
    if(brd) {
        board b = (board)brd;

        __uint128_t result = 0;

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

        // we still need to use the entire 128 bits though,
        // because the hashing algorithm works best in powers of 2
        uint32_t upperupper = 0, upperlower = 0, lowerupper = 0, lowerlower = 0;
        hashlittle2(&result, 8, &upperupper, &upperlower);
        hashlittle2(((char*)&result) + 8, 8, &lowerupper, &lowerlower);
        result = 0;
        result += upperupper;
        result = result << 32;
        result += upperlower;
        result = result << 32;
        result += lowerupper;
        result = result << 32;
        result += lowerlower;

        return result;
    }
    return 0;
}

__uint128_t board_spiral_hash(void* brd) {
    if(brd) {
        board b = (board)brd;

        __uint128_t result = 0;

        uint8_t r = b->height >> 1, c = b->width >> 1, spiral_dimension, iter, v;
        for(uint8_t c = 0; c < (b->width >> 1); c++) {
            spiral_dimension = 2 << c;

            // Perform the box
            for(iter = 0; iter < spiral_dimension; iter++) {
                v = board_get(b, r - iter, c);
                result += v;
                if(c < (b->width - 1) || r < (b->height - 1)) result = result << 2;
            }
            for(iter = 0; iter < spiral_dimension; iter++) {
                v = board_get(b, r - spiral_dimension, c - iter);
                result += v;
                if(c < (b->width - 1) || r < (b->height - 1)) result = result << 2;
            }
            for(iter = 0; iter < spiral_dimension; iter++) {
                v = board_get(b, r - spiral_dimension + iter, c - spiral_dimension);
                result += v;
                if(c < (b->width - 1) || r < (b->height - 1)) result = result << 2;
            }
            for(iter = 0; iter < spiral_dimension; iter++) {
                v = board_get(b, r, c - spiral_dimension + iter);
                result += v;
                if(c < (b->width - 1) || r < (b->height - 1)) result = result << 2;
            }

            // Move to the next spiral
            r++;
            c++;
        }
    }
}