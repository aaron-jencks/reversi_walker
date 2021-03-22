#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generates a unique filename in the /tmp directory for checkpoint saving
 * 
 * @param filename The name of the file to save the data to in the /tmp directory (ex. "checkpoint.bin")
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary filename.
 */
char* find_temp_filename(const char* filename);

/**
 * @brief Generates a unique directory in the /home/$USER/Temp directory for mempage swapping
 * 
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary directory. Must be free'd by the user
 */
char* find_temp_directory();

/**
 * @brief Generates a unique filename in the given directory
 * 
 * @param directory The directory to prepend the filename with
 * @return char* Returns a null terminated string that contains the absolute path of the generated temporary file. Must be free'd by the user
 */
char* find_abs_path(size_t page_index, const char* directory);

#ifdef __cplusplus
}
#endif