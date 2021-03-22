#include "semaphore.hpp"

MultiReadSemaphore::MultiReadSemaphore(size_t max_readers) {
    read_count = 0;
    can_read = 1;
    pthread_mutex_init(&write_mutex, 0);
    pthread_mutex_init(&write_flag_mutex, 0);
    pthread_mutex_init(&read_mutex, 0);
    sem_init(&read_semaphore, 0, max_readers);
}

MultiReadSemaphore::~MultiReadSemaphore() {
    pthread_mutex_destroy(&write_mutex);
    pthread_mutex_destroy(&write_flag_mutex);
    pthread_mutex_destroy(&read_mutex);
    sem_destroy(&read_semaphore);
}

void MultiReadSemaphore::signal_read() {
    while(!can_read || pthread_mutex_trylock(&read_mutex)) sched_yield();
    if(++read_count == 1) while(pthread_mutex_trylock(&write_mutex)) sched_yield();
    pthread_mutex_unlock(&read_mutex);
    while(sem_trywait(&read_semaphore)) sched_yield();
}

void MultiReadSemaphore::signal_read_finish() {
    while(pthread_mutex_trylock(&read_mutex)) sched_yield();
    if(!--read_count) pthread_mutex_unlock(&write_mutex);
    pthread_mutex_unlock(&read_mutex);
    sem_post(&read_semaphore);
}

void MultiReadSemaphore::signal_read_write() {
    while(pthread_mutex_trylock(&write_flag_mutex)) sched_yield();
    can_read = 0;
    while(read_count > 1) sched_yield();
}

void MultiReadSemaphore::signal_write() {
    while(pthread_mutex_trylock(&write_flag_mutex)) sched_yield();
    can_read = 0;
    while(pthread_mutex_trylock(&write_mutex)) sched_yield();
}

void MultiReadSemaphore::signal_write_finish() {
    can_read = 1;
    pthread_mutex_unlock(&write_mutex);
    pthread_mutex_unlock(&write_flag_mutex);
}