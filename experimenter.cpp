#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include <err.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>

#include "./utils/tarraylist.hpp"


#define BUILD_PATH "/home/aaron/Workspace/github/mine/reversi_walker"
#define RUNTIME 86400


// TODO add way to make the log files auto move into a general folder.


typedef struct _experiment_t {
    char* executable;
    char* log_directory;
    char* output_directory;
} experiment_t;


char* setup_build(const char* cflags) {
    char* temp_dir = (char*)((getenv("TEMP_DIR")) ? getenv("TEMP_DIR") : "/tmp");

    // Create a folder for the build
    char* temp_result = (char*)malloc(sizeof(char) * (strlen(temp_dir) + 18));
    if(!temp_result) err(1, "Memory error while allocating build string\n");
    snprintf(temp_result, strlen(temp_dir) + 18, "%s/experiment.XXXXXX", temp_dir);
    temp_dir = mkdtemp(temp_result);

    // Execute make cflags='cflags'
    size_t pid = fork();

    if(pid == 0) {
        if(chdir(BUILD_PATH)) err(15, "Changing to build directory failed\n");

        char* cflag_str_temp = "cflags='%s'", *cflag_str = (char*)malloc(sizeof(char) * (10 + strlen(cflags)));
        if(!cflag_str) err(1, "Memory error while allocating build string\n");
        snprintf(cflag_str, strlen(cflags) + 10, cflag_str_temp, cflags);

        printf("Executing build for cflags='%s'\n", cflags);

        execlp("make", "make", cflag_str, NULL);
    }
    else {
        waitpid(pid, 0, 0);
        printf("Build is complete\n");

        char* exec_file = (char*)malloc(sizeof(char) * (strlen(temp_dir) + 6)), *target_file = (char*)malloc(sizeof(char) * (strlen(BUILD_PATH) + 6));
        if(!(exec_file && target_file)) err(1, "Memory error while allocating build string\n");
        snprintf(exec_file, strlen(temp_dir) + 6, "%s/main", temp_dir);
        snprintf(target_file, strlen(BUILD_PATH) + 6, "%s/main", BUILD_PATH);
        
        if(rename(exec_file, target_file)) err(16, "Error while moving executable\n");
        free(exec_file);

        // return the path of the filename of the main executable
        return target_file;
    }
}


int main(int argc, char* argv[]) {
    Arraylist<size_t> pids(argc + 1);

    if(argc > 1) {
        size_t pid = fork();

        if(pid) {
            // Perform the experiments
            for(size_t a = 1; a < argc; a++) {
                char* executable = setup_build(argv[a]);

                size_t child_pid = fork();

                if(child_pid) { pids.append(child_pid); free(executable); }
                else execlp(executable, executable, "", NULL);
            }
        }
        else {
            // Perform the default process
            char* executable = setup_build("");

            size_t child_pid = fork();

            if(child_pid) { pids.append(child_pid); free(executable); }
            else execlp(executable, executable, "", NULL);

            exit(0);
        }

        // Wait for the children to finish, or maybe just run them for a certain amount of time and then finish them?
        time_t start = time(0), current;
        size_t tdiff;

        printf("Waiting for experiments to complete\n");

        do {
            sched_yield();
            current = time(0);
            tdiff = current - start;
        } while(tdiff < RUNTIME);

        for(size_t p = 0; p < pids.pointer; p++) {
            kill(pids.data[p], SIGINT);
        }

        printf("Experiments complete!\n");
    }
    else {
        printf("To run experiments, please supply cflags argument for each instance\n");
    }
}