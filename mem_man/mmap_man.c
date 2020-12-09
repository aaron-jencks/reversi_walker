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
#include "fileio.h"

#define PAGE_BIN_COUNT 2000

mmap_bin create_mmap_bin(uint8_t in_use, __uint128_t bid, size_t num_elements) {
    mmap_bin bin = calloc(1, sizeof(mmap_bin_str));
    if(!bin) err(1, "Memory error while allocating mmap bin for mmap manager\n");
    bin->usage_flag |= (in_use) ? 2 : 0;
    bin->bid = bid;
    return bin;
}

void destroy_mmap_bin(mmap_bin bin) {
    if(bin) {
        free(bin);
    }
}

mmap_page create_mmap_page(const char* filename, size_t size) {
    mmap_page page = malloc(sizeof(mmap_page_str));
    if(!page) err(1, "Memory error while allocating mmap for mmap manager\n");
    page->filename = malloc(sizeof(char) * (strlen(filename) + 1));
    memcpy(page->filename, filename, strlen(filename) + 1);

    page->fd = open(filename, O_RDWR | O_CREAT);
    posix_fallocate(page->fd, 0, size);

    page->map = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, page->fd, 0);
    if(page->map == MAP_FAILED) err(11, "Mapping failed!\n");

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

// mmap_man create_mmap_man(__uint128_t num_bins, size_t page_size, size_t initial_bin_size, const char* file_directory) {
//     mmap_man man = malloc(sizeof(mmap_man_str));
//     if(!man) err(1, "Memory error while allocating mmap manager\n");
//     man->num_bins = num_bins;
//     man->num_pages = (num_bins * find_mmap_bin_total_size(initial_bin_size) / page_size) + 1;

//     man->pages = malloc(sizeof(mmap_page) * man->num_pages);
//     if(!man->pages) err(1, "Memory error while allocating pages for mmap manager\n");

//     man->file_directory = find_temp_directory();

//     for(size_t p = 0; p < man->num_pages; p++) {
//         char* filename = find_abs_path(p, man->file_directory);
//         man->pages[p] = create_mmap_page(filename, page_size);
//     }

//     size_t bin_pages = num_bins / PAGE_BIN_COUNT + 1;

//     man->bins = malloc(sizeof(mmap_bin*) * bin_pages);
//     if(!man->bins) err(1, "Memory error while allocating bins for mmap manager\n");

//     size_t ubc = 0;
//     __uint128_t bid = 0;
//     for(size_t p = 0; p < bin_pages; p++) {
//         man->bins[p] = malloc(sizeof(mmap_bin) * PAGE_BIN_COUNT);
//         if(!man->bins[p]) err(1, "Memory error while allocating bins for mmap manager\n");

//         for(size_t b = 0; b < PAGE_BIN_COUNT; b++) {
//             man->bins[p][b] = create_mmap_bin((++ubc <= num_bins) ? 1 : 0, bid++, initial_bin_size);
//             // TODO insert the bins into the actual page
//             // NVM do this in append to bin, and just check if bin->elements is null
//         }
//     }
// }

// void destroy_mmap_man(mmap_man man) {
//     if(man) {
//         for(size_t p = 0; p < man->num_pages; p++) destroy_mmap_page(man->pages[p], man->max_page_size);
//         size_t bin_pages = man->num_bins / PAGE_BIN_COUNT + 1;
//         for(size_t p = 0; p < bin_pages; p++) {
//             for(size_t b = 0; b < PAGE_BIN_COUNT; b++) destroy_mmap_bin(man->bins[p][b]);
//             free(man->bins[p]);
//         }
//         free(man->bins);
//         free(man->file_directory);
//         free(man->pages);
//         free(man);
//     }
// }

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