package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/mikebarb/stt/internal/whisper"
)

// ---------------------------------------------------------------------------
// Time formatting. All segment times are in units of 10 ms (centiseconds),
// as returned by the whisper layer.
// ---------------------------------------------------------------------------

func csHMS(cs int64) (h, m, s, ms int64) {
	h = cs / 360000
	cs -= h * 360000
	m = cs / 6000
	cs -= m * 6000
	s = cs / 100
	ms = (cs - s*100) * 10
	return
}

// "HH:MM:SS,mmm" — SRT style (comma before milliseconds).
func csSRT(cs int64) string {
	h, m, s, ms := csHMS(cs)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}

// "HH:MM:SS.mmm" — VTT and timestamp-line style.
func csStamp(cs int64) string {
	h, m, s, ms := csHMS(cs)
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}

// "hh:mm" — for embedded minute markers.
func csMinute(cs int64) string {
	return fmt.Sprintf("%02d:%02d", cs/360000, (cs/6000)%60)
}

// ---------------------------------------------------------------------------
// Format writers.
// ---------------------------------------------------------------------------

// writeText: bare transcript, one segment per line.
func writeText(w io.Writer, segs []whisper.Segment) error {
	for _, s := range segs {
		if _, err := fmt.Fprintln(w, strings.TrimSpace(s.Text)); err != nil {
			return err
		}
	}
	return nil
}

// writeTimestamped: "[hh:mm:ss.mmm --> hh:mm:ss.mmm] text" — the form the
// upstream CLI and the golden references use.
func writeTimestamped(w io.Writer, segs []whisper.Segment) error {
	for _, s := range segs {
		_, err := fmt.Fprintf(w, "[%s --> %s]  %s\n",
			csStamp(cs(s.Start)), csStamp(cs(s.End)), strings.TrimSpace(s.Text))
		if err != nil {
			return err
		}
	}
	return nil
}

// writeMarkers: flowing text with a (hh:mm) marker inserted before the
// first segment that begins in each new minute. Designed for
// fast-forward navigation while reading.
func writeMarkers(w io.Writer, segs []whisper.Segment) error {
	lastMinute := int64(-1)
	var b strings.Builder
	for _, s := range segs {
		minute := cs(s.Start) / 6000
		if minute != lastMinute {
			if b.Len() > 0 {
				if _, err := fmt.Fprintln(w, strings.TrimSpace(b.String())); err != nil {
					return err
				}
			}
			b.Reset()
			if _, err := fmt.Fprintf(w, "(%s)\n", csMinute(cs(s.Start))); err != nil {
				return err
			}
			lastMinute = minute
		}
		b.WriteString(strings.TrimSpace(s.Text))
		b.WriteString(" ")
	}
	if b.Len() > 0 {
		_, err := fmt.Fprintln(w, strings.TrimSpace(b.String()))
		return err
	}
	return nil
}

// writeSRT: SubRip subtitles.
func writeSRT(w io.Writer, segs []whisper.Segment) error {
	for i, s := range segs {
		_, err := fmt.Fprintf(w, "%d\n%s --> %s\n%s\n\n",
			i+1, csSRT(cs(s.Start)), csSRT(cs(s.End)), strings.TrimSpace(s.Text))
		if err != nil {
			return err
		}
	}
	return nil
}

// writeVTT: WebVTT subtitles.
func writeVTT(w io.Writer, segs []whisper.Segment) error {
	if _, err := fmt.Fprintln(w, "WEBVTT"); err != nil {
		return err
	}
	for i, s := range segs {
		_, err := fmt.Fprintf(w, "%d\n%s --> %s\n%s\n\n",
			i+1, csStamp(cs(s.Start)), csStamp(cs(s.End)), strings.TrimSpace(s.Text))
		if err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// JSON — the contract shared with the Phase 2 GPU service and the Phase 3
// summariser. Times are seconds (float). This schema is the interchange
// point between the Go tool and the Python stack; do not change it without
// updating docs/ARCHITECTURE.md.
// ---------------------------------------------------------------------------

type jsonSegment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

type jsonTranscript struct {
	SourceFile      string        `json:"source_file"`
	ModelUsed       string        `json:"model_used"`
	DurationSeconds float64       `json:"duration_seconds"`
	Segments        []jsonSegment `json:"segments"`
}

func writeJSON(w io.Writer, segs []whisper.Segment, sourceFile, modelUsed string, durationSeconds float64) error {
	out := jsonTranscript{
		SourceFile:      sourceFile,
		ModelUsed:       modelUsed,
		DurationSeconds: durationSeconds,
		Segments:        make([]jsonSegment, 0, len(segs)),
	}
	for _, s := range segs {
		out.Segments = append(out.Segments, jsonSegment{
			Start: float64(cs(s.Start) / 100.0),
			End:   float64(cs(s.End) / 100.0),
			Text:  strings.TrimSpace(s.Text),
		})
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(&out)
}

// dispatch writes the requested format. Returns an error for unknown names
// so a typo fails before transcription, not after.
func writeOutput(w io.Writer, format string, segs []whisper.Segment, sourceFile, modelUsed string, durationSeconds float64) error {
	switch strings.ToLower(format) {
	case "text":
		return writeText(w, segs)
	case "timestamp":
		return writeTimestamped(w, segs)
	case "markers":
		return writeMarkers(w, segs)
	case "srt":
		return writeSRT(w, segs)
	case "vtt":
		return writeVTT(w, segs)
	case "json":
		return writeJSON(w, segs, sourceFile, modelUsed, durationSeconds)
	default:
		return fmt.Errorf("unknown output format %q (want text|timestamp|markers|srt|vtt|json)", format)
	}
}

// cs converts a Duration to whisper's centisecond units.
func cs(d time.Duration) int64 { return int64(d / (10 * time.Millisecond)) }
