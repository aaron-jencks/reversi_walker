#include "mempage.h"
#include "../utils/fileio.h"

#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <time.h>

#define MEMORY_THRESHOLD 750000000
#define CHECK_SWAP_INTERV 1000000

#pragma region mempage

mempage create_mempage(size_t page_max, __uint128_t num_bins, size_t bin_size) {
    mempage mp = malloc(sizeof(mempage_str));
    if(!mp) err(1, "Memory Error while allocating memory page manager\n");

    __uint128_t pages = (num_bins / page_max) + 1;

    #ifdef mempagedebug
        printf("Creating a mempage with %lu %lu pages\n", ((uint64_t*)&pages)[1], ((uint64_t*)&pages)[0]);
    #endif

    // mp->pages = create_ll();
    mp->pages = malloc(sizeof(__uint128_t**) * pages);
    if(!mp->pages) err(1, "Memory error while allocating book for mempage\n");

    mp->access_counts = calloc(pages, sizeof(size_t));
    if(!mp->access_counts) err(1, "Memory error while allocating book for mempage\n");

    mp->page_present = calloc(((pages >> 3) + 1), sizeof(uint8_t));
    if(!mp->page_present) err(1, "Memory error while allocating book for mempage\n");
    // for(uint8_t b = 0; b < ((pages >> 3) + 1); b++) mp->page_present[b] = 255;

    mp->bin_counts = malloc(sizeof(size_t*) * pages);
    if(!mp->bin_counts) err(1, "Memory error while allocating book for mempage\n");

    mp->page_count = pages;
    mp->count_per_page = page_max;
    mp->num_bins = num_bins;
    mp->bin_size = bin_size;
    mp->swap_directory = find_temp_directory();
    mp->save_interv_counter = 0;

    for(__uint128_t p = 0; p < pages; p++) {
        __uint128_t** bins = malloc(sizeof(__uint128_t*) * page_max);
        if(!bins) err(1, "Memory error while allocating array for mempage page\n");

        size_t* sizes = malloc(sizeof(size_t) * page_max);
        if(!sizes) err(1, "Memory error while allocating array for mempage page\n");

        for(size_t b = 0; b < page_max; b++) {
            bins[b] = calloc(bin_size, sizeof(__uint128_t));
            if(!bins[b]) err(1, "Memory error while allocating bin for mempage\n");

            sizes[b] = mp->bin_size;
        }

        mp->pages[p] = bins;
        mp->bin_counts[p] = sizes;

        size_t page_index = p >> 3, bit = p % 8;
        uint8_t ph = 1 << bit;
        mp->page_present[page_index] |= ph;

        if(mempage_find_total_size(mp) > MEMORY_THRESHOLD) {
            size_t pindex = mempage_find_least_used_page(mp);

            #ifdef mempagedebug
                printf("Cache size: %ld/%ld: Saving page %ld to file\n", mempage_find_total_size(mp), MEMORY_THRESHOLD, pindex);
            #endif

            save_mempage_page(mp, pindex, mp->swap_directory);
        }
    }

    return mp;
}

void destroy_mempage(mempage mp) {
    if(mp) {
        for(__uint128_t p = 0; p < mp->page_count; p++) {
            for(size_t b = 0; b < mp->count_per_page; b++) free(mp->pages[p][b]);
            free(mp->pages[p]);
            free(mp->bin_counts[p]);
        }
        free(mp->pages);
        free(mp->access_counts);
        free(mp->bin_counts);
        free(mp->page_present);
        free(mp->swap_directory);
        free(mp);
    }
}

// __uint128_t* mempage_get(mempage mp, __uint128_t index) {
//     if(index >= mp->num_bins) err(4, "Index out of bounds in mempage\n");

//     uint32_t page = index / mp->count_per_page, page_index = index % mp->count_per_page;

//     // Extract the page
//     __uint128_t** l = mp->pages[page];

//     // Extract the int
//     return l[page_index];
// }

// void mempage_put(mempage mp, __uint128_t index, void* data) {
//     if(index >= mp->num_bins) err(4, "Index out of bounds in mempage\n");
    
//     uint32_t page = index / mp->count_per_page, page_index = index % mp->count_per_page;

//     // Extract the page
//     ptr_arraylist l = (ptr_arraylist)mp->pages->data[page];

//     // Extract the int
//     (l)->data[page_index] = data;
// }

uint8_t mempage_page_exists(mempage mp, size_t page_index) {
    size_t byte = page_index >> 3, bit = page_index % 8;
    uint8_t ph = 1 << bit, result = 0;
    result = mp->page_present[byte] & ph;
    return (result) ? 1 : 0;
}

size_t mempage_find_least_used_page(mempage mp) {
    size_t min = 0, result;
    uint8_t found = 0, first_page = 1;
    for(size_t i = 0; i < mp->page_count; i++) {
        if(mempage_page_exists(mp, i) && (first_page || mp->access_counts[i] < min)) {
            if(first_page) first_page = 0;
            min = mp->access_counts[i];
            result = i;
            found = 1;
        }
    }
    return (found) ? result : 0;
}

size_t mempage_find_total_size(mempage mp) {
    size_t result = sizeof(void*) * mp->page_count * sizeof(size_t) * mp->count_per_page;
    uint8_t ph;
    for(size_t b = 0; b < ((mp->page_count >> 3) + 1); b++) {
        ph = 1;
        for(uint8_t bit = 0; bit < 8; bit++) {
            if(bit) ph = ph << 1;
            if(mp->page_present[b] & ph) {
                result += sizeof(void*) + sizeof(void*) * mp->count_per_page;
                for(size_t bn = 0; bn < mp->count_per_page; bn++) {
                    result += sizeof(__uint128_t) * mp->bin_size;
                }
            }
        }
    }
    // for(size_t p = 0; p < mp->page_count; p++) {
    //     if(mempage_page_exists(mp, p)) {
    //         result += sizeof(void*) + sizeof(void*) * mp->count_per_page;
    //         for(size_t b = 0; b < mp->count_per_page; b++) {
    //             result += sizeof(__uint128_t) * mp->bin_counts[p][b];
    //         }
    //     }
    // }
    return result + sizeof(mempage_str);
}

void mempage_append_bin(mempage mp, __uint128_t bin_index, __uint128_t value) {
    if(bin_index >= mp->num_bins) err(4, "Index %lu %lu out of bounds in mempage of size %lu %lu\n", 
                                      ((uint64_t*)&bin_index)[1], ((uint64_t*)&bin_index)[0], 
                                      ((uint64_t*)&mp->num_bins)[1], ((uint64_t*)&mp->num_bins)[0]);
    uint32_t page = bin_index / mp->count_per_page, page_index = bin_index % mp->count_per_page;

    // printf("Calculated page and bin %ld\n", clock());

    if(!mempage_page_exists(mp, page)) {
        size_t lpage = mempage_find_least_used_page(mp);
        if(mempage_page_exists(mp, lpage)) swap_mempage_page(mp, lpage, page, mp->swap_directory);
        else load_mempage_page(mp, page, mp->swap_directory);
    }

    // printf("Checked page existence %ld\n", clock());

    mp->access_counts[page]++;

    #ifdef mempagedebug
        // printf("Memory page has been accessed %ld times\n", mp->access_counts[page]);
    #endif

    __uint128_t** l = mp->pages[page];
    size_t bcount = mp->bin_counts[page][page_index];
    
    __uint128_t* bin = l[page_index];
    for(size_t iter = 0;; iter++) {
        if(!bin[iter]) {
            bin[iter] = value;
            break;
        }
        else if(iter == bcount) {
            // reallocate the bin
            #ifdef mempagedebug
                printf("Reallocating bin %lu %lu to %ld\n", ((uint64_t*)&bin_index)[1], ((uint64_t*)&bin_index)[0], bcount + 10);
            #endif

            bin = realloc(bin, sizeof(__uint128_t) * (bcount + 10));

            for(size_t b = bcount; b < (bcount + 10); b++) bin[b] = 0;

            bcount += 10;

            l[page_index] = bin;
            mp->bin_counts[page][page_index] = bcount;

            bin[iter] = value;
            break;
        }
    }

    // printf("Inserted element %ld\n", clock());

    if(++mp->save_interv_counter == CHECK_SWAP_INTERV) {
        if(mempage_find_total_size(mp) > MEMORY_THRESHOLD) {
            #ifdef mempagedebug
                printf("The cache is too big, attempting to swap\n");
            #endif

            size_t pindex = mempage_find_least_used_page(mp);

            if(mempage_page_exists(mp, pindex)) {
                #ifdef mempagedebug
                    printf("Cache size: %ld/%ld: Saving page %ld to file\n", mempage_find_total_size(mp), MEMORY_THRESHOLD, pindex);
                #endif

                save_mempage_page(mp, pindex, mp->swap_directory);
            }
        }
        mp->save_interv_counter = 0;
    }

    // printf("Checked for swap %ld\n", clock());
}

uint8_t mempage_value_in_bin(mempage mp, __uint128_t bin_index, __uint128_t value) {
    if(bin_index >= mp->num_bins) err(4, "Index out of bounds in mempage\n");
    uint32_t page = bin_index / mp->count_per_page, page_index = bin_index % mp->count_per_page;

    if(!mempage_page_exists(mp, page)) {
        size_t lpage = mempage_find_least_used_page(mp);
        swap_mempage_page(mp, lpage, page, mp->swap_directory);
    }

    mp->access_counts[page]++;

    #ifdef mempagedebug
        // printf("Memory page has been accessed %ld times\n", mp->access_counts[page]);
    #endif

    __uint128_t** l = mp->pages[page];
    size_t bcount = mp->bin_counts[page][page_index];
    
    __uint128_t* bin = l[page_index];
    for(size_t b = 0; b < bcount; b++) if(bin[b] == value) return 1;
    return 0;
}

void mempage_clear_all(mempage mp) {
    size_t last_known_mem_page = 0;
    for(size_t p = 0; p < mp->page_count; p++) {
        if(mempage_page_exists(mp, p)) {
            last_known_mem_page = p;
            break;
        }
    }

    for(size_t p = 0; p < mp->page_count; p++) {
        if(!mempage_page_exists(mp, p)) {
            if(mempage_page_exists(mp, last_known_mem_page)) swap_mempage_page(mp, last_known_mem_page, p, mp->swap_directory);
            else load_mempage_page(mp, p, mp->swap_directory);
            last_known_mem_page = p;
        }
        
        for(size_t b = 0; b < mp->count_per_page; b++) {
            free(mp->pages[p][b]);
            mp->pages[p][b] = calloc(mp->bin_size, sizeof(__uint128_t));
            if(!mp->pages[p][b]) err(1, "Memory error while allocating bin for mempage\n");
            mp->bin_counts[p][b] = mp->bin_size;
        }
    }
}

void mempage_realloc(mempage mp, __uint128_t bin_count) {
    __uint128_t pages = (bin_count / mp->count_per_page) + 1;
    __uint128_t diff = pages - mp->page_count;

    #ifdef mempagedebug
        printf("Creating a mempage with %lu %lu pages\n", ((uint64_t*)&pages)[1], ((uint64_t*)&pages)[0]);
    #endif

    if(diff) {
        // mp->pages = create_ll();
        mp->pages = realloc(mp->pages, sizeof(__uint128_t**) * pages);
        if(!mp->pages) err(1, "Memory error while allocating book for mempage\n");

        mp->access_counts = realloc(mp->access_counts, pages * sizeof(size_t));
        if(!mp->access_counts) err(1, "Memory error while allocating book for mempage\n");

        mp->page_present = realloc(mp->page_present, ((pages >> 3) + 1) * sizeof(uint8_t));
        if(!mp->page_present) err(1, "Memory error while allocating book for mempage\n");

        mp->bin_counts = realloc(mp->bin_counts, sizeof(size_t**) * pages);
        if(!mp->bin_counts) err(1, "Memory error while allocating book for mempage\n");

        for(size_t p = mp->page_count; p < pages; p++) {
            __uint128_t** bins = malloc(sizeof(__uint128_t*) * mp->count_per_page);
            if(!bins) err(1, "Memory error while allocating array for mempage page\n");

            size_t* sizes = malloc(sizeof(size_t) * mp->count_per_page);
            if(!sizes) err(1, "Memory error while allocating array for mempage page\n");

            for(size_t b = 0; b < mp->count_per_page; b++) {
                bins[b] = calloc(mp->bin_size, sizeof(__uint128_t));
                if(!bins[b]) err(1, "Memory error while allocating bin for mempage\n");
                sizes[b] = mp->bin_size;
            }

            mp->pages[p] = bins;
            mp->bin_counts[p] = sizes;
            mp->access_counts[p] = 0;

            // Mark the page as not present in RAM
            size_t byte = p >> 3, bit = p % 8;
            uint8_t ph = 1 << bit;
            mp->page_present[byte] |= ph;

            if(mempage_find_total_size(mp) > MEMORY_THRESHOLD) {
                size_t pindex = mempage_find_least_used_page(mp);

                #ifdef mempagedebug
                    printf("Cache size: %ld/%ld: Saving page %ld to file\n", mempage_find_total_size(mp), MEMORY_THRESHOLD, pindex);
                #endif

                save_mempage_page(mp, pindex, mp->swap_directory);
            }
        }

        mp->page_count = pages;
    }

    mp->num_bins = bin_count;
}

#pragma endregion
#pragma region mempage_buff

uint8_t mempage_buff_page_exists(mempage_buff mp, size_t page_index) {
    size_t byte = page_index >> 3, bit = page_index % 8;
    uint8_t ph = 1 << bit, result = 0;
    result = mp->page_present[byte] & ph;
    return (result) ? 1 : 0;
}

size_t mempage_buff_find_least_used_page(mempage_buff mp) {
    for(size_t i = 0; i < mp->page_count; i++) {
        if(mempage_buff_page_exists(mp, i)) {
            return i;
        }
    }
    return 0;
}

size_t mempage_buff_find_total_size(mempage_buff buff) {
    size_t result = sizeof(void*) * buff->page_count;
    for(size_t p = 0; p < buff->page_count; p++) {
        if(mempage_buff_page_exists(buff, p)) {
            result += sizeof(__uint128_t) * buff->count_per_page;
        }
    }
    return result + sizeof(mempage_buff_str);
}

mempage_buff create_mempage_buff(__uint128_t num_elements, size_t page_size) {
    mempage_buff buff = malloc(sizeof(mempage_buff_str));
    if(!buff) err(1, "Memory error while allocating mempage buffer\n");

    buff->count_per_page = page_size;
    buff->num_element = num_elements;
    buff->save_interv_counter = 0;

    size_t num_pages = num_elements / page_size + 1;

    buff->pages = malloc(sizeof(__uint128_t*) * num_pages);
    buff->page_present = calloc((num_pages >> 3) + 1, sizeof(uint8_t));
    if(!buff->pages | !buff->page_present) err(1, "Memory error while allocating mempage pages\n");

    for(size_t p = 0; p < num_pages; p++) {
        buff->pages[p] = calloc(page_size, sizeof(__uint128_t));
        if(!buff->pages[p]) err(1, "Memory error while allocating mempage page\n");

        // Mark the page as present in memory
        size_t page_index = p >> 3, byte_index = p % 8;
        uint8_t ph = 1 << byte_index;
        buff->page_present[page_index] |= ph;

        if(mempage_buff_find_total_size(buff) > MEMORY_THRESHOLD) {
            size_t pindex = mempage_buff_find_least_used_page(buff);

            #ifdef mempagedebug
                printf("Saving page %ld to file\n", pindex);
            #endif

            save_mempage_buff_page(buff, pindex, buff->swap_directory);
        }
    }

    buff->swap_directory = find_temp_directory();

    return buff;
}

void destroy_mempage_buff(mempage_buff buff) {
    size_t num_pages = buff->num_element / buff->count_per_page + 1;

    for(size_t p = 0; p < num_pages; p++) free(buff->pages[p]);
    free(buff->pages);

    free(buff);
}

void mempage_buff_put(mempage_buff buff, __uint128_t index, __uint128_t value) {
    if(index >= buff->num_element) err(4, "Index %lu %lu is out of bounds in mempage buffer\n", ((uint64_t*)&index)[1], ((uint64_t*)&index)[0]);
    uint32_t page = index / buff->count_per_page, page_index = index % buff->count_per_page;

    if(!mempage_buff_page_exists(buff, page)) {
        size_t lpage = mempage_buff_find_least_used_page(buff);
        if(mempage_buff_page_exists(buff, lpage)) swap_mempage_buff_page(buff, lpage, page, buff->swap_directory);
        else load_mempage_buff_page(buff, page, buff->swap_directory);
    }

    __uint128_t* l = buff->pages[page];
    l[page_index] = value;

    if(++buff->save_interv_counter == CHECK_SWAP_INTERV) {
        if(mempage_buff_find_total_size(buff) > MEMORY_THRESHOLD) {
            size_t pindex = mempage_buff_find_least_used_page(buff);

            #ifdef mempagedebug
                printf("Saving page %ld to file\n", pindex);
            #endif

            save_mempage_buff_page(buff, pindex, buff->swap_directory);
        }
        buff->save_interv_counter = 0;
    }
}

__uint128_t mempage_buff_get(mempage_buff buff, __uint128_t index) {
    if(index >= buff->num_element) err(4, "Index %lu %lu is out of bounds in mempage buffer\n", ((uint64_t*)&index)[1], ((uint64_t*)&index)[0]);
    uint32_t page = index / buff->count_per_page, page_index = index % buff->count_per_page;

    if(!mempage_buff_page_exists(buff, page)) {
        size_t lpage = mempage_buff_find_least_used_page(buff);
        if(mempage_buff_page_exists(buff, lpage)) swap_mempage_buff_page(buff, lpage, page, buff->swap_directory);
        else load_mempage_buff_page(buff, page, buff->swap_directory);
    }

    return buff->pages[page][page_index];
}

#pragma endregion
#pragma region bit mempage region

bit_mempage create_bit_mempage(__uint128_t num_bits, size_t page_size) {
    bit_mempage mp = malloc(sizeof(bit_mempage_str));
    if(!mp) err(1, "Memory error while allocating bit cache mempage\n");

    mp->count_per_page = page_size;
    mp->num_elements = num_bits;
    mp->page_count = num_bits / page_size + 1;

    mp->mmap_directory = find_temp_directory();

    mp->mpages = malloc(sizeof(mmap_page) * mp->page_count);
    mp->pages = malloc(sizeof(uint8_t*) * mp->page_count);
    if(!mp->mpages || !mp->pages) err(1, "Memory error while allocating pages for bit cache mempage\n");

    for(size_t i = 0; i < mp->page_count; i++) {
        char* filename = find_abs_path(i, mp->mmap_directory);
        mp->mpages[i] = create_mmap_page(filename, page_size);
        mp->pages[i] = (uint8_t*)(mp->mpages[i]->map);
        for(size_t b = 0; b < mp->count_per_page; b++) mp->pages[i][b] = 0; // Initialize the data
        free(filename);
    }

    return mp;
}

void destroy_bit_mempage(bit_mempage mp) {
    if(mp) {
        free(mp->pages);
        for(size_t i = 0; i < mp->page_count; i++) destroy_mmap_page(mp->mpages[i], mp->count_per_page);
        free(mp->mpages);
        free(mp->mmap_directory);
        free(mp);
    }
}

void bit_mempage_put(bit_mempage buff, __uint128_t index, uint8_t value) {
    size_t page = index / buff->count_per_page, byte_index = index / 8, bit_index = index % 8;
    if(value) buff->pages[page][byte_index] |= 1 << bit_index;
    else {
        uint8_t ph = 254, lower;
        ph = ph << bit_index;

        lower = 1 << bit_index;
        ph += --lower;

        buff->pages[page][byte_index] &= ph;
    }
}

uint8_t bit_mempage_get(bit_mempage buff, __uint128_t index) {
    size_t page = index / buff->count_per_page, byte_index = index / 8, bit_index = index % 8;
    uint8_t ph = 1 << bit_index;
    return buff->pages[page][byte_index] & ph;
}

#pragma endregion