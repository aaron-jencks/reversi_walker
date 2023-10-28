package checkpoints

import (
	"fmt"
	"os"
	"time"

	"github.com/aaron-jencks/reversi/utils"
)

// represents the values stored in a checkpoint file
type FileStats struct {
	Version      string
	Counter      uint64
	Explored     uint64
	Repeated     uint64
	StartTime    time.Time
	CacheSize    uint64
	WalkerBoards uint64
}

func CheckpointStats(filename string) (FileStats, error) {
	fmt.Printf("Parsing checkpoint %s\n", filename)

	result := FileStats{}

	fp, err := os.OpenFile(filename, os.O_RDONLY, 0777)
	if err != nil {
		return result, err
	}
	defer fp.Close()

	// reusable buffer for reading uint64 data
	i64buff := make([]byte, 8)

	_, err = fp.Read(i64buff)
	if err != nil {
		return result, err
	}
	vlen := utils.Uint64FromBytes(i64buff)
	vbuff := make([]byte, vlen)
	_, err = fp.Read(vbuff)
	if err != nil {
		return result, err
	}
	fmt.Printf("Checkpoint version %s\n", string(vbuff))
	result.Version = string(vbuff)

	_, err = fp.Read(i64buff)
	if err != nil {
		return result, err
	}
	result.Counter = utils.Uint64FromBytes(i64buff)

	_, err = fp.Read(i64buff)
	if err != nil {
		return result, err
	}
	result.Explored = utils.Uint64FromBytes(i64buff)

	_, err = fp.Read(i64buff)
	if err != nil {
		return result, err
	}
	result.Repeated = utils.Uint64FromBytes(i64buff)

	// read start time
	_, err = fp.Read(i64buff)
	if err != nil {
		return result, err
	}
	tbinlen := utils.Uint64FromBytes(i64buff)
	tbinbuff := make([]byte, tbinlen)
	_, err = fp.Read(tbinbuff)
	if err != nil {
		return result, err
	}
	err = result.StartTime.UnmarshalBinary(tbinbuff)
	if err != nil {
		return result, err
	}

	_, err = fp.Read(i64buff)
	if err != nil {
		return result, err
	}
	result.CacheSize = utils.Uint64FromBytes(i64buff)

	noff, err := fp.Seek(int64(result.CacheSize)*16, 1)
	if err != nil {
		return result, err
	}

	stats, err := fp.Stat()
	if err != nil {
		return result, err
	}
	bsize := stats.Size()
	rfs := bsize - noff
	result.WalkerBoards = uint64(rfs / 16)

	return result, nil
}
