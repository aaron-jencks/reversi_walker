package checkpoints

import (
	"fmt"
	"os"
	"time"

	"github.com/aaron-jencks/reversi/utils"
	"github.com/aaron-jencks/reversi/visiting"
)

func SaveSimulation(fname string, fchans []chan *os.File, rchans []chan bool, cache visiting.VisitedCache, counted, explored, repeated uint64,
	elapsed time.Duration) error {

	fp, err := os.OpenFile(fname, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0777)
	if err != nil {
		return err
	}
	defer fp.Close()

	const versionString = "v1"
	_, err = fp.Write(utils.Uint64ToBytes(uint64(len(versionString))))
	if err != nil {
		return err
	}
	_, err = fp.Write([]byte(versionString))
	if err != nil {
		return err
	}

	_, err = fp.Write(utils.Uint64ToBytes(counted))
	if err != nil {
		return err
	}
	_, err = fp.Write(utils.Uint64ToBytes(explored))
	if err != nil {
		return err
	}
	_, err = fp.Write(utils.Uint64ToBytes(repeated))
	if err != nil {
		return err
	}
	_, err = fp.Write(utils.Uint64ToBytes(uint64(elapsed)))
	if err != nil {
		return err
	}

	err = cache.ToFile(fp)
	if err != nil {
		return err
	}

	for wi, fchan := range fchans {
		fchan <- fp
		v := <-rchans[wi]
		if !v {
			return fmt.Errorf("failed to save writer %d", wi)
		}
	}
	return nil
}
