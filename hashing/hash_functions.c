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

        uint8_t r = b->height >> 1, c = b->width >> 1, spiral_dimension, iter, v, delta_dimension;
        for(uint8_t cb = 0; cb < (b->width >> 1); cb++) {
            spiral_dimension = 2 * (cb + 1);
            delta_dimension = spiral_dimension - 1;

            // Perform the box
            for(iter = 0; iter < spiral_dimension; iter++) {
                v = board_get(b, r - iter, c);
                result += v;
                result = result << 2;
            }
            for(iter = 1; iter < spiral_dimension; iter++) {
                v = board_get(b, r - delta_dimension, c - iter);
                result += v;
                result = result << 2;
            }
            for(iter = 1; iter < spiral_dimension; iter++) {
                v = board_get(b, r - delta_dimension + iter, c - delta_dimension);
                result += v;
                result = result << 2;
            }
            for(iter = 1; iter < (spiral_dimension - 1); iter++) {
                v = board_get(b, r, c - delta_dimension + iter);
                result += v;
                if(iter < spiral_dimension - 2 || r < b->height - 1 && c < b->width - 1) result = result << 2;
            }

            // Move to the next spiral
            r++;
            c++;
        }

        return result;
    }

    return 0;
}