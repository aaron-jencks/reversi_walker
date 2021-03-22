#include "walker.cuh"
#include "reversi.cuh"

// Remember to turn unified memory profiling off before profiling!

__global__ void compute_mass_next_cuda(board_str* boards, board_str* result, size_t n, size_t pitch) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    size_t result_pointer = 0;

    for(int i = index; i < n; i += stride) {
        board_str b, *row = (board_str*)(result + i * pitch);
        clone_into_board_cuda(&b, &boards[i]);

        for(uint8_t r = 0; r < b.height; r++) {
            for(uint8_t c = 0; c < b.width; c++) {
                if(board_is_legal_move_cuda(&b, r, c)) {
                    board_place_piece_cuda(&b, r, c);
                    clone_into_board_cuda(&row[result_pointer++], &b);
                }
            }
        }

        row[result_pointer] = board_str {0, 0, 0, 0};
    }
}