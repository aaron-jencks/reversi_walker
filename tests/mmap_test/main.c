#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <err.h>
#include <stdint.h>

#define LEN 64000000


int main() {
    int fp = open("./swapfile", O_RDWR);
    
    posix_fallocate(fp, 0, LEN);

    void* res = mmap(0, LEN, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fp, 0);
    if(res == MAP_FAILED) err(1, "Mapping failed!\n");
    
    for(size_t i = 0; i < (LEN / 4096); i++) {
        printf("\r%ld/%ld", i, LEN / 4096);
        ((char*)res)[i * 4095] = 255;
    }
    printf("\n");
    
    int fp1 = open("./swapfile_1", O_RDWR);
    
    posix_fallocate(fp1, 0, LEN);

    res = mmap(0, LEN, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fp1, 0);
    if(res == MAP_FAILED) err(1, "Mapping failed!\n");
    
    for(size_t i = 0; i < (LEN / 4096); i++) {
        printf("\r%ld/%ld", i, LEN / 4096);
        ((char*)res)[i * 4095] = 255;
    }
    printf("\n");

    close(fp);
    close(fp1);
}
