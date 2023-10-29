package utils

import "github.com/aaron-jencks/reversi/utils/uint128"

// Uint64ToBytes encodes a uint64 to an array of bytes
func Uint64ToBytes(i uint64) []byte {
	return []byte{
		byte(i >> 56),
		byte(i >> 48),
		byte(i >> 40),
		byte(i >> 32),
		byte(i >> 24),
		byte(i >> 16),
		byte(i >> 8),
		byte(i),
	}
}

// Uint64FromBytes decodes an array of bytes into a uint64
func Uint64FromBytes(b []byte) uint64 {
	result := uint64(0)
	result |= uint64(b[0]) << 56
	result |= uint64(b[1]) << 48
	result |= uint64(b[2]) << 40
	result |= uint64(b[3]) << 32
	result |= uint64(b[4]) << 24
	result |= uint64(b[5]) << 16
	result |= uint64(b[6]) << 8
	result |= uint64(b[7])
	return result
}

// Uint128ToBytes encodes a uint128 to an array of bytes
func Uint128ToBytes(i uint128.Uint128) []byte {
	return append(Uint64ToBytes(i.H), Uint64ToBytes(i.L)...)
}

// Uint128FromBytes decodes an array of bytes into a uint128
func Uint128FromBytes(b []byte) uint128.Uint128 {
	result := uint128.Uint128{}
	result.H = Uint64FromBytes(b)
	result.L = Uint64FromBytes(b[8:])
	return result
}
