#pragma once

#ifdef limitprocs
    #define PROC_COUNT 1
#else
    #define PROC_COUNT 32
#endif

#define BOARD_MEMORY_SIZE 13
#define SYSTEM_MEMORY 30000000000
#define CHUNK_SIZE (SYSTEM_MEMORY / (BOARD_MEMORY_SIZE * PROC_COUNT))

// TODO add error codes here.