package caching

type PCacheInitializer[T any] func() T

type PointerCache[T any] struct {
	Buff    []T
	Pointer int
}

func CreatePointerCache[T any](capacity int, initializer PCacheInitializer[T]) PointerCache[T] {
	cache := PointerCache[T]{
		Buff:    make([]T, 0, capacity),
		Pointer: 0,
	}
	for bi := 0; bi < capacity; bi++ {
		cache.Buff[bi] = initializer()
	}
	return cache
}
