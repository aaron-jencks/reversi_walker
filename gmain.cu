#include <stdio.h>

#include "./cuda/walker.cuh"
#include "./gameplay/reversi.h"
#include "./cuda/reversi.cuh"
#include "./gameplay/reversi_defs.h"


int main() {
    board_str *test, *result;
    size_t pitch = 65;

    cudaMallocManaged(&test, sizeof(board_str) * 1000000);
    cudaMallocManaged(&result, sizeof(board_str) * 65000000);

    board template_board = create_board_cuda(1, 8, 8);
    for(size_t i = 0; i < 1000000; i++) { clone_into_board(&test[i], template_board); }
    destroy_board_cuda(template_board);

    compute_mass_next_cuda<<<16, 1024>>>(test, result, 1000000, pitch);
    cudaDeviceSynchronize();
    printf("Cuda Error %d should be %d\n", cudaPeekAtLastError(), cudaSuccess);

    printf("Finished processing\n");

    size_t count = 0;
    for(size_t bs = 0; bs < 1000000; bs++) {
        board_str* row = (board_str*)((char*)result + bs * pitch);

        for(size_t b = 0; b < 64; b++) {
            if(row[b].height) count++;
            else break;
        }
    }

    cudaFree(test);
    cudaFree(result);

    printf("Finished computation with %lu boards found\n", count);
}