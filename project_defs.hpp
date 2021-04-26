#pragma once

#define CHUNK_SIZE 337500000
#ifdef limitprocs
    #define PROC_COUNT 1
#else
    #define PROC_COUNT 32
#endif

// TODO add error codes here.