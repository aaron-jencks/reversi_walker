#include <stdlib.h>
#include <err.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "mmap_man.h"
#include "../utils/fileio.h"

#ifdef mmapdebug
    #include <stdio.h>
#endif

#define PAGE_BIN_COUNT 2000

// mmap_bin create_mmap_bin(uint8_t in_use, __uint128_t bid, size_t num_elements) {
//     mmap_bin bin = calloc(1, sizeof(mmap_bin_str));
//     if(!bin) err(1, "Memory error while allocating mmap bin for mmap manager\n");
//     bin->usage_flag |= (in_use) ? 2 : 0;
//     bin->bid = bid;
//     return bin;
// }

// void destroy_mmap_bin(mmap_bin bin) {
//     if(bin) {
//         free(bin);
//     }
// }

mmap_page create_mmap_page(const char* filename, size_t size) {
    mmap_page page = malloc(sizeof(mmap_page_str));
    if(!page) err(1, "Memory error while allocating mmap for mmap manager\n");
    page->filename = malloc(sizeof(char) * (strlen(filename) + 1));
    memcpy(page->filename, filename, strlen(filename) + 1);

    #ifdef mmapdebug
        printf("Creating mmap_page at %s\n", page->filename);
    #endif

    page->fd = open(filename, O_RDWR | O_CREAT);
    posix_fallocate(page->fd, 0, size);

    page->map = (uint8_t*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, page->fd, 0);
    if(page->map == MAP_FAILED) err(11, "Mapping failed!\n");
    page->free_pointer = page->map;
    page->size = 0;

    return page;
}

void destroy_mmap_page(mmap_page page, size_t size){
    if(page) {
        munmap(page->map, size);
        close(page->fd);
        remove(page->filename);
        free(page->filename);
        free(page);
    }
}

mmap_man create_mmap_man(size_t page_size, size_t bin_size) {
    mmap_man man = malloc(sizeof(mmap_man_str));
    if(!man) err(1, "Memory error while allocating mmap manager\n");
    man->num_pages = 1;

    man->pages = malloc(sizeof(mmap_page) * man->num_pages);
    if(!man->pages) err(1, "Memory error while allocating pages for mmap manager\n");

    man->file_directory = find_temp_directory();
    man->max_page_size = page_size;
    man->bins_per_page = page_size / bin_size;
    man->elements_per_bin = bin_size;

    for(size_t p = 0; p < man->num_pages; p++) {
        char* filename = find_abs_path(p, man->file_directory);
        man->pages[p] = create_mmap_page(filename, page_size);
        free(filename);
    }

    return man;
}

void destroy_mmap_man(mmap_man man) {
    if(man) {
        for(size_t p = 0; p < man->num_pages; p++) destroy_mmap_page(man->pages[p], man->max_page_size);
        free(man->file_directory);
        free(man->pages);
        free(man);
    }
}

// size_t find_mmap_bin_total_size(size_t initial_bin_size) {
//     return sizeof(mmap_bin_str) + sizeof(__uint128_t) * (initial_bin_size + 1);
// }

// /**
//  * @brief We need a way to insert new elements
//  * 
//  * @param man 
//  * @param bin_index 
//  * @param value 
//  */
// void mmap_man_append_bin(mmap_man man, __uint128_t bin_index, __uint128_t value) {

// }

// /**
//  * @brief We need a way to extend previously allocated bins
//  * 
//  * @param man 
//  * @param bin_index 
//  * @return mmap_bin 
//  */
// mmap_bin mmap_man_generate_bin_extension(mmap_man man, __uint128_t bin_index) {

// }

// /**
//  * @brief We need a way to move previously allocated bins to make room for bin extensions
//  * 
//  * We assume that the bin is already in the page that we don't want it in anymore.
//  * 
//  * @param man 
//  * @param bin_index 
//  */
// void mmap_man_migrate_bin(mmap_man man, __uint128_t bin_index) {

// }

uint8_t* mmap_allocate_bin(mmap_man man) {
    if(man->pages[man->num_pages - 1]->size >= man->bins_per_page) {
        man->pages = realloc(man->pages, man->num_pages++ + 1);

        char* filename = find_abs_path(man->num_pages - 1, man->file_directory);
        man->pages[man->num_pages - 1] = create_mmap_page(filename, man->max_page_size);
    }

    mmap_page page = man->pages[man->num_pages - 1];
    uint8_t* position = page->free_pointer;
    page->free_pointer += man->elements_per_bin;
    page->size++;

    #ifdef mmapdebug
        printf("Returned a new bin at pointer %p with %lu elements\n", position, man->elements_per_bin);
    #endif

    return position;
}