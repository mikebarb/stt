package main

import (
	"io"

	"github.com/mikebarb/stt/internal/format"
	"github.com/mikebarb/stt/internal/whisper"
)

// toFormatSegments converts the whisper layer's segments into the shared
// renderer type in internal/format.
//
// The two are structurally identical. The conversion exists because
// internal/format must stay free of CGo, so that cmd/sttpack can be built and
// run without a C toolchain.
func toFormatSegments(segs []whisper.Segment) []format.Segment {
	out := make([]format.Segment, 0, len(segs))
	for _, s := range segs {
		fs := format.Segment{Start: s.Start, End: s.End, Text: s.Text}
		if len(s.Tokens) > 0 {
			fs.Tokens = make([]format.Token, 0, len(s.Tokens))
			for _, t := range s.Tokens {
				fs.Tokens = append(fs.Tokens, format.Token{
					Text:  t.Text,
					Prob:  t.Prob,
					Start: t.Start,
					End:   t.End,
				})
			}
		}
		out = append(out, fs)
	}
	return out
}

// writeOutput is the single entry point for transcript rendering.
func writeOutput(w io.Writer, name string, segs []whisper.Segment,
	sourceFile, modelUsed string, durationSeconds float64) error {
	return format.WriteOutput(w, name, toFormatSegments(segs), sourceFile, modelUsed, durationSeconds)
}

// writeAudit writes the low-confidence report to a file.
func writeAudit(path string, segs []whisper.Segment, threshold float64, modelUsed string) (int, error) {
	return format.WriteAuditFile(path, toFormatSegments(segs), threshold, modelUsed)
}

// writeAuditWriter writes the low-confidence report to an arbitrary writer.
func writeAuditWriter(w io.Writer, segs []whisper.Segment, threshold float64, modelUsed string) (int, error) {
	return format.WriteAudit(w, toFormatSegments(segs), threshold, modelUsed)
}
