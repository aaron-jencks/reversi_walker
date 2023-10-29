package caching

type ArrayStack[T any] struct {
	buff    []T
	pointer int
}

func CreateArrayStack[T any](capacity int) ArrayStack[T] {
	return ArrayStack[T]{
		buff:    make([]T, capacity),
		pointer: 0,
	}
}

func (s ArrayStack[T]) Index(index int) T {
	return s.buff[index]
}

func (s ArrayStack[T]) Len() int {
	return s.pointer
}

func (s *ArrayStack[T]) Clear() {
	s.pointer = 0
}

func (s *ArrayStack[T]) Push(e T) int {
	if s.pointer < len(s.buff) {
		s.buff[s.pointer] = e
	} else {
		s.buff = append(s.buff, e)
	}
	s.pointer++
	return s.pointer - 1
}

func (s *ArrayStack[T]) Pop() T {
	s.pointer--
	return s.buff[s.pointer]
}
