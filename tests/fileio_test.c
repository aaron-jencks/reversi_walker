#include "fileio_test.h"
#include "../hashtable.h"
#include "../fileio.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define INSERTIONS 16000000
#define SAVING_CYCLES 1

__uint128_t key = 1;

__uint128_t debug_hash(void* brd) {
    return key++;
}

void fio_test_hashtable_write() {
    printf("Testing fileio with a full hashtable\n");
    char* checkpoint_filename = find_temp_filename("checkpoint.bin\0");
    printf("Saving to %s\n", checkpoint_filename);
    hashtable hs = create_hashtable(1000000, &debug_hash);

    printf("Inserting elements\n");
    for(uint32_t k = 0; k < INSERTIONS; k++) {
        printf("\r%u/%d", k, INSERTIONS);
        fflush(stdout);
        put_hs(hs, k);
    }

    printf("\nMoving to saving cycle\n");
    for(uint32_t h = 0; h < SAVING_CYCLES; h++) {
        printf("Iteration: %d, Saving file with %d entries\n", h, INSERTIONS);
        FILE* fp = fopen(checkpoint_filename, "wb+");
        to_file_hs(fp, hs);
        fclose(fp);

        destroy_hashtable(hs);

        fp = fopen(checkpoint_filename, "rb+");
        hs = from_file_hs(fp, &debug_hash);
        fclose(fp);

        printf("New hashtable size %ld\n", hs->size);
        assert(hs->size == INSERTIONS);
    }

    destroy_hashtable(hs);

    printf("Deleting file\n");
    remove(checkpoint_filename);
    free(checkpoint_filename);
}