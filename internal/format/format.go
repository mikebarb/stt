package format

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Token is one decoded token within a segment, matching whisper.Token.
// Start and End are in units of 10 ms (centiseconds).
type Token struct {
	Text  string
	Prob  float32
	Start int64
	End   int64
}

// Segment is one transcript segment with optional per-token detail.
type Segment struct {
	Start  time.Duration
	End    time.Duration
	Text   string
	Tokens []Token
}

// ---------------------------------------------------------------------------
// Time formatting (units of 10 ms / centiseconds).
// ---------------------------------------------------------------------------

// Cs converts a Duration to whisper's centisecond units.
func Cs(d time.Duration) int64 { return int64(d / (10 * time.Millisecond)) }

func CsHMS(cs int64) (h, m, s, ms int64) {
	h = cs / 360000
	cs -= h * 360000
	m = cs / 6000
	cs -= m * 6000
	s = cs / 100
	ms = (cs - s*100) * 10
	return
}

// CsSRT formats "HH:MM:SS,mmm" — SRT style (comma before milliseconds).
func CsSRT(cs int64) string {
	h, m, s, ms := CsHMS(cs)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}

// CsStamp formats "HH:MM:SS.mmm" — VTT, timestamp-line, and audit style.
func CsStamp(cs int64) string {
	h, m, s, ms := CsHMS(cs)
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}

// CsMinute formats "hh:mm" — for embedded minute markers.
func CsMinute(cs int64) string {
	return fmt.Sprintf("%02d:%02d", cs/360000, (cs/6000)%60)
}

// ---------------------------------------------------------------------------
// Format writers.
// ---------------------------------------------------------------------------

// WriteText writes bare transcript text, one segment per line.
func WriteText(w io.Writer, segs []Segment) error {
	for _, s := range segs {
		if _, err := fmt.Fprintln(w, strings.TrimSpace(s.Text)); err != nil {
			return err
		}
	}
	return nil
}

// RenderText returns the exact string written by WriteText.
func RenderText(segs []Segment) string {
	var b strings.Builder
	_ = WriteText(&b, segs)
	return b.String()
}

// WriteTimestamped writes "[hh:mm:ss.mmm --> hh:mm:ss.mmm]  text".
func WriteTimestamped(w io.Writer, segs []Segment) error {
	for _, s := range segs {
		_, err := fmt.Fprintf(w, "[%s --> %s]  %s\n",
			CsStamp(Cs(s.Start)), CsStamp(Cs(s.End)), strings.TrimSpace(s.Text))
		if err != nil {
			return err
		}
	}
	return nil
}

// WriteMarkers writes flowing text with a (hh:mm) marker inserted before the
// first segment beginning in each new minute.
func WriteMarkers(w io.Writer, segs []Segment) error {
	lastMinute := int64(-1)
	var b strings.Builder
	for _, s := range segs {
		minute := Cs(s.Start) / 6000
		if minute != lastMinute {
			if b.Len() > 0 {
				if _, err := fmt.Fprintln(w, strings.TrimSpace(b.String())); err != nil {
					return err
				}
			}
			b.Reset()
			if _, err := fmt.Fprintf(w, "(%s)\n", CsMinute(Cs(s.Start))); err != nil {
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

// WriteSRT writes SubRip subtitles.
func WriteSRT(w io.Writer, segs []Segment) error {
	for i, s := range segs {
		_, err := fmt.Fprintf(w, "%d\n%s --> %s\n%s\n\n",
			i+1, CsSRT(Cs(s.Start)), CsSRT(Cs(s.End)), strings.TrimSpace(s.Text))
		if err != nil {
			return err
		}
	}
	return nil
}

// WriteVTT writes WebVTT subtitles.
func WriteVTT(w io.Writer, segs []Segment) error {
	if _, err := fmt.Fprintln(w, "WEBVTT"); err != nil {
		return err
	}
	for i, s := range segs {
		_, err := fmt.Fprintf(w, "%d\n%s --> %s\n%s\n\n",
			i+1, CsStamp(Cs(s.Start)), CsStamp(Cs(s.End)), strings.TrimSpace(s.Text))
		if err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// JSON — Phase 2/3 interchange contract.
// ---------------------------------------------------------------------------

type JSONSegment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

type JSONTranscript struct {
	SourceFile      string        `json:"source_file"`
	ModelUsed       string        `json:"model_used"`
	DurationSeconds float64       `json:"duration_seconds"`
	Segments        []JSONSegment `json:"segments"`
}

func WriteJSON(w io.Writer, segs []Segment, sourceFile, modelUsed string, durationSeconds float64) error {
	out := JSONTranscript{
		SourceFile:      filepath.Base(sourceFile),
		ModelUsed:       modelUsed,
		DurationSeconds: durationSeconds,
		Segments:        make([]JSONSegment, 0, len(segs)),
	}
	for _, s := range segs {
		out.Segments = append(out.Segments, JSONSegment{
			Start: float64(Cs(s.Start)) / 100.0,
			End:   float64(Cs(s.End)) / 100.0,
			Text:  strings.TrimSpace(s.Text),
		})
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(&out)
}

// WriteOutput dispatches by format name.
func WriteOutput(w io.Writer, name string, segs []Segment, sourceFile, modelUsed string, durationSeconds float64) error {
	switch strings.ToLower(name) {
	case "text", "txt":
		return WriteText(w, segs)
	case "timestamp":
		return WriteTimestamped(w, segs)
	case "markers":
		return WriteMarkers(w, segs)
	case "srt":
		return WriteSRT(w, segs)
	case "vtt":
		return WriteVTT(w, segs)
	case "json":
		return WriteJSON(w, segs, sourceFile, modelUsed, durationSeconds)
	default:
		return fmt.Errorf("unknown output format %q (want text|timestamp|markers|srt|vtt|json)", name)
	}
}

// ---------------------------------------------------------------------------
// Audit report writer.
// ---------------------------------------------------------------------------

// WriteAudit writes every token with probability strictly below threshold.
func WriteAudit(w io.Writer, segs []Segment, threshold float64, modelUsed string) (int, error) {
	flagged := 0
	if _, err := fmt.Fprintf(w, "# audit threshold p < %.2f  model %s\n", threshold, modelUsed); err != nil {
		return 0, err
	}
	for _, s := range segs {
		for _, t := range s.Tokens {
			if float64(t.Prob) >= threshold {
				continue
			}
			flagged++
			if _, err := fmt.Fprintf(w, "[%s -> %s] p=%.3f %q\n",
				CsStamp(t.Start), CsStamp(t.End), t.Prob, strings.TrimSpace(t.Text)); err != nil {
				return flagged, err
			}
		}
	}
	if _, err := fmt.Fprintf(w, "# %d tokens flagged\n", flagged); err != nil {
		return flagged, err
	}
	return flagged, nil
}

// WriteAuditFile writes the audit report to path.
func WriteAuditFile(path string, segs []Segment, threshold float64, modelUsed string) (int, error) {
	f, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return WriteAudit(f, segs, threshold, modelUsed)
}
