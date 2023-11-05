package reportbuilding

import (
	"io"

	"github.com/aaron-jencks/reversi/visiting"
)

var CurrentReportBuilder ReportBuilder = BuildCSVReport

// Represents an interface for constructing a report from the set of final board states
// found by the walkers
type ReportBuilder func(io.Writer, visiting.VisitedCache, uint8) error
