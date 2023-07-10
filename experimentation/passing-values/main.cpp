#include <iostream>
#include <chrono>
#include <stdint.h>

using namespace std;

size_t counter = 0;

template <typename T> T* value_test(T v) {
    counter++;
    return &v;
}

struct board_size_t {
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t* d;
};

template <typename T> T t_constructor();

template <> 
board_size_t t_constructor<board_size_t>() {
    return board_size_t{};
}

template <> 
board_size_t* t_constructor<board_size_t*>() {
    return (board_size_t*)0;
}

template <typename T> void test_func(size_t n) {
    volatile T* result;
    for(size_t i = 0; i++ < n;) {
        result = value_test(t_constructor<T>());
    }
}

#define ELAPSED_MICRO(s) chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - (s)).count()
#define N 100000000

int main() {
    counter = 0;
    auto start = chrono::steady_clock::now();
    test_func<board_size_t>(N);
    auto microseconds = ELAPSED_MICRO(start);
    cout << "ran simulation: " << counter << " times.\n";

    cout << "struct time: " << microseconds / N << endl;

    counter = 0;
    start = chrono::steady_clock::now();
    test_func<board_size_t*>(N);
    microseconds = ELAPSED_MICRO(start);
    cout << "ran simulation: " << counter << " times.\n";

    cout << "pointer time: " << microseconds / N << endl;
    return 0;
}