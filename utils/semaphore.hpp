#pragma once

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <semaphore.h>

class MultiReadSemaphore {
    private:
        size_t read_count;
        uint8_t can_read;
        pthread_mutex_t write_flag_mutex;
        pthread_mutex_t write_mutex;
        pthread_mutex_t read_mutex;
        sem_t read_semaphore;
    public:
        MultiReadSemaphore(size_t max_readers);
        ~MultiReadSemaphore();

        void signal_read();
        void signal_read_finish();

        void signal_read_write();

        void signal_write();
        void signal_write_finish();
};