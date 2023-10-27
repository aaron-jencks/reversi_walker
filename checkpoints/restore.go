package checkpoints

// import (
// 	"fmt"
// 	"os"
// 	"time"

// 	"github.com/aaron-jencks/reversi/walking"
// )

// func RestoreSimulation(filename string, size uint8, counter, explored, repeated *uint64, tstart *time.Time) ([]walking.BoardWalker, error) {
// 	fmt.Printf("Starting restore from %s\n", filename)
// 	fp, err := os.OpenFile(filename, os.O_RDONLY, 0777)
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer fp.Close()

// 	uint64FromBytes := func(b []byte) uint64 {
// 		result := uint64(0)
// 		result |= uint64(b[0]) << 56
// 		result |= uint64(b[1]) << 48
// 		result |= uint64(b[2]) << 40
// 		result |= uint64(b[3]) << 32
// 		result |= uint64(b[4]) << 24
// 		result |= uint64(b[5]) << 16
// 		result |= uint64(b[6]) << 8
// 		result |= uint64(b[7])
// 		return result
// 	}

// 	// reusable buffer for reading uint64 data
// 	i64buff := make([]byte, 8)

// 	_, err = fp.Read(i64buff)
// 	if err != nil {
// 		return nil, err
// 	}
// 	*counter = uint64FromBytes(i64buff)

// 	_, err = fp.Read(i64buff)
// 	if err != nil {
// 		return nil, err
// 	}
// 	*explored = uint64FromBytes(i64buff)

// 	_, err = fp.Read(i64buff)
// 	if err != nil {
// 		return nil, err
// 	}
// 	*repeated = uint64FromBytes(i64buff)

// 	// read start time
// 	_, err = fp.Read(i64buff)
// 	if err != nil {
// 		return nil, err
// 	}
// 	tbinlen := uint64FromBytes(i64buff)
// 	tbinbuff := make([]byte, tbinlen)
// 	_, err = fp.Read(tbinbuff)
// 	if err != nil {
// 		return nil, err
// 	}
// 	err = tstart.UnmarshalBinary(tbinbuff)
// 	if err != nil {
// 		return nil, err
// 	}

// 	fmt.Println("Restored counters and start time, reading boards...")
// 	// the rest of the file is boards
// 	i128buff := make([]byte, 16)
// 	_, err = fp.Read(i128buff)
// 	for err == nil {

// 	}

// }
