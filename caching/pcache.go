package caching

type PCacheInitializer[T any] func() T

type PointerCache[T any] struct {
	buff        []T
	available   []int
	initializer PCacheInitializer[T]
}

func CreatePointerCache[T any](capacity int, initializer PCacheInitializer[T]) PointerCache[T] {
	cache := PointerCache[T]{
		buff:        make([]T, capacity),
		available:   make([]int, capacity),
		initializer: initializer,
	}
	for bi := 0; bi < capacity; bi++ {
		cache.buff[bi] = initializer()
		cache.available[bi] = bi
	}
	return cache
}

// Get returns an initialized element from the cache
//
// If there are no elements available in the cache,
// then a new one is allocated and initialized
func (pc *PointerCache[T]) Get() (int, *T) {
	if len(pc.available) == 0 {
		pc.buff = append(pc.buff, pc.initializer())
		lbuff := len(pc.buff) - 1
		return lbuff, &pc.buff[lbuff]
	}
	last := len(pc.available) - 1
	index := pc.available[last]
	pc.available = pc.available[:last]
	return index, &pc.buff[index]
}

// Free marks the given index as available for retrieval by Get
//
// if the index is out of bounds, or is already free, then this is a no op
func (pc *PointerCache[T]) Free(index int) {
	if index < 0 || index >= len(pc.buff) {
		return
	}

	for _, i := range pc.available {
		if i == index {
			return
		}
	}

	pc.available = append(pc.available, index)
}
