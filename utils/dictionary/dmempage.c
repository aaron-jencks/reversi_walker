#include "dmempage.h"

#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <time.h>

#pragma region dmempage

dmempage create_dmempage(size_t page_max, __uint128_t num_bins, size_t bin_size) {
    dmempage mp = malloc(sizeof(dmempage_str));
    if(!mp) err(1, "Memory Error while allocating memory page manager\n");

    __uint128_t pages = (num_bins / page_max) + 1;

    #ifdef dmempagedebug
        printf("Creating a dmempage with %lu %lu pages\n", ((uint64_t*)&pages)[1], ((uint64_t*)&pages)[0]);
    #endif

    // mp->pages = create_ll();
    mp->pages = malloc(sizeof(dict_pair_t**) * pages);
    if(!mp->pages) err(1, "Memory error while allocating book for dmempage\n");

    mp->bin_counts = malloc(sizeof(size_t*) * pages);
    if(!mp->bin_counts) err(1, "Memory error while allocating book for dmempage\n");

    mp->page_count = pages;
    mp->count_per_page = page_max;
    mp->num_bins = num_bins;
    mp->bin_size = bin_size;

    for(__uint128_t p = 0; p < pages; p++) {
        dict_pair_t** bins = malloc(sizeof(dict_pair_t*) * page_max);
        if(!(bins)) err(1, "Memory error while allocating array for dmempage page\n");

        size_t* sizes = malloc(sizeof(size_t) * page_max);
        if(!sizes) err(1, "Memory error while allocating array for dmempage page\n");

        for(size_t b = 0; b < page_max; b++) {
            bins[b] = calloc(bin_size, sizeof(dict_pair_t));
            if(!(bins[b])) err(1, "Memory error while allocating bin for dmempage\n");

            sizes[b] = mp->bin_size;
        }

        mp->pages[p] = bins;
        mp->bin_counts[p] = sizes;

        size_t page_index = p >> 3, bit = p % 8;
        uint8_t ph = 1 << bit;
        mp->page_present[page_index] |= ph;
    }

    return mp;
}

void destroy_dmempage(dmempage mp) {
    if(mp) {
        for(__uint128_t p = 0; p < mp->page_count; p++) {
            for(size_t b = 0; b < mp->count_per_page; b++) free(mp->pages[p][b]);
            free(mp->pages[p]);
            free(mp->bin_counts[p]);
        }
        free(mp->pages);
        free(mp->bin_counts);
        free(mp->page_present);
        free(mp);
    }
}

void dmempage_append_bin(dmempage mp, __uint128_t bin_index, dict_pair_t value) {
    if(bin_index >= mp->num_bins) err(4, "Index %lu %lu out of bounds in dmempage of size %lu %lu\n", 
                                      ((uint64_t*)&bin_index)[1], ((uint64_t*)&bin_index)[0], 
                                      ((uint64_t*)&mp->num_bins)[1], ((uint64_t*)&mp->num_bins)[0]);
    uint32_t page = bin_index / mp->count_per_page, page_index = bin_index % mp->count_per_page;

    dict_pair_t** l = mp->pages[page];
    size_t bcount = mp->bin_counts[page][page_index];
    
    dict_pair_t* bin = l[page_index];
    for(size_t iter = 0;; iter++) {
        if(!bin[iter].key) {
            bin[iter] = value;
            break;
        }
        else if(iter == bcount) {
            // reallocate the bin
            #ifdef dmempagedebug
                printf("Reallocating bin %lu %lu to %ld\n", ((uint64_t*)&bin_index)[1], ((uint64_t*)&bin_index)[0], bcount + 10);
            #endif

            bin = realloc(bin, sizeof(dict_pair_t) * (bcount + 10));

            for(size_t b = bcount; b < (bcount + 10); b++) bin[b].key = 0;

            bcount += 10;

            l[page_index] = bin;
            mp->bin_counts[page][page_index] = bcount;

            bin[iter] = value;
            break;
        }
    }
}

uint8_t dmempage_value_in_bin(dmempage mp, __uint128_t bin_index, __uint128_t value) {
    if(bin_index >= mp->num_bins) err(4, "Index out of bounds in dmempage\n");
    uint32_t page = bin_index / mp->count_per_page, page_index = bin_index % mp->count_per_page;

    dict_pair_t** l = mp->pages[page];
    size_t bcount = mp->bin_counts[page][page_index];
    
    dict_pair_t* bin = l[page_index];
    for(size_t b = 0; b < bcount; b++) if(bin[b].key == value) return 1;
    return 0;
}

void dmempage_clear_all(dmempage mp) {
    for(size_t p = 0; p < mp->page_count; p++) {
        for(size_t b = 0; b < mp->count_per_page; b++) {
            free(mp->pages[p][b]);
            mp->pages[p][b] = calloc(mp->bin_size, sizeof(dict_pair_t));
            if(!(mp->pages[p][b])) err(1, "Memory error while allocating bin for dmempage\n");
            mp->bin_counts[p][b] = mp->bin_size;
        }
    }
}

void dmempage_realloc(dmempage mp, __uint128_t bin_count) {
    __uint128_t pages = (bin_count / mp->count_per_page) + 1;
    __uint128_t diff = pages - mp->page_count;

    #ifdef dmempagedebug
        printf("Creating a dmempage with %lu %lu pages\n", ((uint64_t*)&pages)[1], ((uint64_t*)&pages)[0]);
    #endif

    if(diff) {
        // mp->pages = create_ll();
        mp->pages = realloc(mp->pages, sizeof(dict_pair_t**) * pages);
        if(!(mp->pages)) err(1, "Memory error while allocating book for dmempage\n");

        mp->bin_counts = realloc(mp->bin_counts, sizeof(size_t**) * pages);
        if(!mp->bin_counts) err(1, "Memory error while allocating book for dmempage\n");

        for(size_t p = mp->page_count; p < pages; p++) {
            dict_pair_t** bins = malloc(sizeof(dict_pair_t*) * mp->count_per_page);
            if(!bins) err(1, "Memory error while allocating array for dmempage page\n");

            size_t* sizes = malloc(sizeof(size_t) * mp->count_per_page);
            if(!sizes) err(1, "Memory error while allocating array for dmempage page\n");

            for(size_t b = 0; b < mp->count_per_page; b++) {
                bins[b] = calloc(mp->bin_size, sizeof(dict_pair_t));
                if(!bins[b]) err(1, "Memory error while allocating bin for dmempage\n");
                sizes[b] = mp->bin_size;
            }

            mp->pages[p] = bins;
            mp->bin_counts[p] = sizes;

            // Mark the page as not present in RAM
            size_t byte = p >> 3, bit = p % 8;
            uint8_t ph = 1 << bit;
            mp->page_present[byte] |= ph;
        }

        mp->page_count = pages;
    }

    mp->num_bins = bin_count;
}

uint8_t* dmempage_get(dmempage mp, __uint128_t bin_index, __uint128_t key) {
    if(bin_index >= mp->num_bins) err(4, "Index out of bounds in dmempage\n");
    uint32_t page = bin_index / mp->count_per_page, page_index = bin_index % mp->count_per_page;

    dict_pair_t** l = mp->pages[page];
    size_t bcount = mp->bin_counts[page][page_index];
    
    dict_pair_t* bin = l[page_index];
    for(size_t b = 0; b < bcount; b++) if(bin[b].key == key) return bin[b].value;
    return 0;
}

#pragma endregion
#pragma region dmempage_buff

dmempage_buff create_dmempage_buff(__uint128_t num_elements, size_t page_size) {
    dmempage_buff buff = malloc(sizeof(dmempage_buff_str));
    if(!buff) err(1, "Memory error while allocating dmempage buffer\n");

    buff->count_per_page = page_size;
    buff->num_element = num_elements;

    size_t num_pages = num_elements / page_size + 1;

    buff->pages = malloc(sizeof(dict_pair_t*) * num_pages);
    buff->page_present = calloc((num_pages >> 3) + 1, sizeof(uint8_t));
    if(!buff->pages | !buff->page_present) err(1, "Memory error while allocating dmempage pages\n");

    for(size_t p = 0; p < num_pages; p++) {
        buff->pages[p] = calloc(page_size, sizeof(__uint128_t));
        if(!buff->pages[p]) err(1, "Memory error while allocating dmempage page\n");

        // Mark the page as present in memory
        size_t page_index = p >> 3, byte_index = p % 8;
        uint8_t ph = 1 << byte_index;
        buff->page_present[page_index] |= ph;
    }

    return buff;
}

void destroy_dmempage_buff(dmempage_buff buff) {
    size_t num_pages = buff->num_element / buff->count_per_page + 1;

    for(size_t p = 0; p < num_pages; p++) free(buff->pages[p]);
    free(buff->pages);

    free(buff);
}

void dmempage_buff_put(dmempage_buff buff, __uint128_t index, dict_pair_t value) {
    if(index >= buff->num_element) err(4, "Index %lu %lu is out of bounds in dmempage buffer\n", ((uint64_t*)&index)[1], ((uint64_t*)&index)[0]);
    uint32_t page = index / buff->count_per_page, page_index = index % buff->count_per_page;

    dict_pair_t* l = buff->pages[page];
    l[page_index] = value;
}

dict_pair_t dmempage_buff_get(dmempage_buff buff, __uint128_t index) {
    if(index >= buff->num_element) err(4, "Index %lu %lu is out of bounds in dmempage buffer\n", ((uint64_t*)&index)[1], ((uint64_t*)&index)[0]);
    uint32_t page = index / buff->count_per_page, page_index = index % buff->count_per_page;

    return buff->pages[page][page_index];
}

#pragma endregion