#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <err.h>
#include <stdint.h>

#define LEN 64000000000


int main() {
    int fp = open("/home/aaron/Workspace/github/mine/reversi_walker/mmap_test/swapfile", O_RDWR);
    void* res = mmap(0, LEN, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, fp, 0);
    if(res == MAP_FAILED) err(1, "Mapping failed!\n");
    // for(size_t i = 0; i < (LEN / sizeof(__uint128_t)); i++) {
    //     printf("\r%ld/%ld", i, LEN / sizeof(__uint128_t));
    //     ((__uint128_t*)res)[i] = 2147483647;
    // }
    for(size_t i = 0; i < (LEN / 4096); i++) {
        printf("\r%ld/%ld", i, LEN / 4096);
        ((char*)res)[i * 4096] = 255;
    }
    printf("\n");
    close(fp);
}
