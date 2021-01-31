#include "./heir_swapper.h"

bin_dict create_bin_dict(size_t num_bins);
void destroy_bin_dict(bin_dict d);

uint8_t* bin_dict_get(bin_dict d, __uint128_t k);
void bin_dict_put(bin_dict d, __uint128_t k);