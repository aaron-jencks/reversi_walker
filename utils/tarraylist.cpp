#include "tarraylist.hpp"
#include "./dictionary/dict_def.h"

template<> void Arraylist<dict_usage_pair_t>::insert(dict_usage_pair_t element, size_t index) {
	if(index >= size) {
			// reallocate the array
			size = (size) ? size << 1 : 1;
			data = (dict_usage_pair_t*)std::realloc(data, size * sizeof(dict_usage_pair_t));
			if(!data) err(1, "Memory error while allocating arraylist");
			for(size_t e = pointer; e < index; e++) { data[e] = {0, 0, 0}; }
			pointer = index + 1;
	}
	else {
		if(pointer + 1 >= size) {
			size = (size) ? size << 1 : 1;
			data = (dict_usage_pair_t*)std::realloc(data, size * sizeof(dict_usage_pair_t));
			if(!data) err(1, "Memory error while allocating arraylist");
		}

		for(size_t e = index; e < pointer; e++) { data[e + 1] = data[e]; }
		data[index] = element;
		pointer++;
	}
}

template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop(size_t index) {
	if(size && index < size) {
		dict_usage_pair_t d = data[index];
		for(size_t e = index + 1; e < pointer; e++) data[e - 1] = data[e];
		pointer--;
		return d;
	}
	return {0, 0, 0};
}

template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_back() {
	if(size) { return data[--pointer]; }
	return {0, 0, 0};
}

template<> dict_usage_pair_t Arraylist<dict_usage_pair_t>::pop_front() {
	if(size) {
		dict_usage_pair_t d = data[0];
		for(size_t e = 0; e < pointer; e++) data[e] = data[e + 1];
		pointer--;
		return d;
	}
	return {0, 0, 0};
}
