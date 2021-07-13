#pragma once

#define BOARD_WIDTH 6
#define BOARD_HEIGHT 6

#ifdef limitprocs
    #define PROC_COUNT 1
#else
    #define PROC_COUNT 32
#endif

#define BOARD_MEMORY_SIZE 13
#ifdef lowmem
    #define SYSTEM_MEMORY 5368709120
#else
    #define SYSTEM_MEMORY 32813092000
#endif
#define CHUNK_SIZE (SYSTEM_MEMORY / (BOARD_MEMORY_SIZE * (PROC_COUNT + 1)))

#define FILE_SIZE_INCREMENT 5368709120
#define ARRAYLIST_INCREMENT BOARD_WIDTH * BOARD_HEIGHT + 1

// TODO add error codes here.