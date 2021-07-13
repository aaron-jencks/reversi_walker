#include "path_util.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

/**
 * @brief Generates a unique filename in the /tmp directory for checkpoint saving
 * 
 * @param filename The name of the file to save the data to in the /tmp directory (ex. "checkpoint.bin"), must be null-terminated
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary filename. Must be free'd by the user
 */
char* find_temp_filename(const char* filename) {
    // Setup the checkpoint saving system
    char temp_str[] = "/tmp/reversi.XXXXXX", *final_result = (char*)calloc(35, sizeof(char));
    char* dir_ptr = mkdtemp(temp_str);
    strncat(final_result, dir_ptr, 20);
    final_result[19] = '/';
    strncat(final_result + 20, filename, 14);
    #ifdef checkpointdebug
        printf("Saving to %s\n", final_result);
    #endif
    return final_result;
}

/**
 * @brief Generates a unique directory in the /home/$USER/Temp directory for mempage swapping
 * 
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary directory. Must be free'd by the user
 */
char* find_temp_directory() {
    // Setup the checkpoint saving system
    char* temp_str;
    if(!getenv("TEMP_DIR")) {
        struct passwd *p = getpwuid(getuid());  // Check for NULL!
        if(!p) err(1, "Memory Error while getting username for page swap\n");

        temp_str = (char*)malloc(sizeof(char) * strlen(p->pw_name) + 27);
        if(!temp_str) err(1, "Memory Error while allocating temporary directory for swap\n");
        snprintf(temp_str, strlen(p->pw_name) + 27, "/home/%s/Temp/reversi.XXXXXX", p->pw_name);
    }
    else {
        temp_str = (char*)malloc(sizeof(char) * (strlen(getenv("TEMP_DIR")) + 16));
        if(!temp_str) err(1, "Memory Error while allocating temporary directory for swap\n");
        snprintf(temp_str, strlen(getenv("TEMP_DIR")) + 16, "%s/reversi.XXXXXX", getenv("TEMP_DIR"));
    }

    char* dir_ptr = mkdtemp(temp_str), *result = (char*)malloc(sizeof(char) * (strlen(dir_ptr) + 1));
    memcpy(result, dir_ptr, strlen(dir_ptr) + 1);

    #ifdef checkpointdebug
        printf("Swapping to %s\n", dir_ptr);
    #endif

    free(temp_str);

    return result;
}

char* find_abs_path(size_t page_index, const char* swap_directory) {
    char *filename = (char*)malloc(sizeof(char) * 16), 
         *abs_path = (char*)malloc(sizeof(char) * (17 + strlen(swap_directory)));
    if(!filename) err(1, "Memory Error while allocating filename for swap page\n");
    filename[15] = 0;
    snprintf(filename, 15, "p%lu.bin", page_index);

    #ifdef checkpointdebug
        printf("The checkpoint filename is %s\n", filename);
    #endif

    abs_path[0] = 0;
    abs_path = strcat(abs_path, swap_directory);
    abs_path[strlen(swap_directory)] = '/';
    abs_path[strlen(swap_directory) + 1] = 0;

    #ifdef checkpointdebug
        // abs_path[strlen(swap_directory) + 1] = 0;
        printf("Swap directory is %s\n", abs_path);
    #endif

    strcat(abs_path, filename);
    // abs_path[strlen(filename) + strlen(swap_directory) + 1] = 0;
    free(filename);

    #ifdef checkpointdebug
        printf("Saving index %ld to %s\n", page_index, abs_path);
    #endif

    return abs_path;
}

char* temp_dir = 0;

char* get_temp_path() {
    if(temp_dir) return temp_dir;
    else {
        temp_dir = (char*)((getenv("TEMP_DIR")) ? getenv("TEMP_DIR") : "/tmp");
        return temp_dir;
    }
}

char* checkpoint_path = 0;

char* get_checkpoint_filepath() {
    if(!checkpoint_path) {
        if(!getenv("CHECKPOINT_PATH")) {
            checkpoint_path = (char*)malloc(sizeof(char) * (strlen(temp_dir) + 16));
            snprintf(checkpoint_path, strlen(temp_dir) + 16, "%s/checkpoint.bin", temp_dir);
        }
        else checkpoint_path = getenv("CHECKPOINT_PATH");
    }
    return checkpoint_path;
}