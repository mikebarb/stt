package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"

	"github.com/mikebarb/stt/internal/whisper"
)

// defaultConfThreshold is used for the low-confidence count when -audit is
// not set, so the metric is comparable across runs regardless of flags.
const defaultConfThreshold = 0.35

// runStats holds per-run diagnostics. Deliberately separate from the
// transcript JSON: that file is the Phase 2/3 contract and must not drift,
// while these fields are free to change as analysis needs evolve.
type runStats struct {
	SourceFile     string  `json:"source_file"`
	ModelUsed      string  `json:"model_used"`
	AudioSeconds   float64 `json:"audio_seconds"`
	WallSeconds    float64 `json:"wall_seconds"`
	RealtimeFactor float64 `json:"realtime_factor"`

	Segments     int     `json:"segments"`
	Words        int     `json:"words"`
	Characters   int     `json:"characters"`
	Tokens       int     `json:"tokens"`
	WordsPerMin  float64 `json:"words_per_minute"`
	SegmentsPerM float64 `json:"segments_per_minute"`

	MeanSegmentS    float64 `json:"mean_segment_seconds"`
	LongestSegmentS float64 `json:"longest_segment_seconds"`

	MeanTokenConf float64 `json:"mean_token_confidence"`
	MinTokenConf  float64 `json:"min_token_confidence"`
	LowConfTokens int     `json:"low_confidence_tokens"`
	ConfThreshold float64 `json:"confidence_threshold"`

	StartSec  float64 `json:"start_seconds"`
	EndSec    float64 `json:"end_seconds"`
	Truncated bool    `json:"truncated"`
}

func round3(f float64) float64 {
	return float64(int64(f*1000+0.5)) / 1000
}

// isWordToken reports whether a token carries lexical content. Punctuation
// and spacing tokens have systematically low probabilities and would drag
// the mean confidence down for reasons unrelated to recognition quality.
func isWordToken(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

// computeStats derives the per-run diagnostics from the transcript.
//
// Token-level figures require Segment.Tokens to be populated, which means
// Options.Audit must be true when the model is opened (see the note in
// main.go / batchConfig handling).
func computeStats(segs []whisper.Segment, inputPath, tag string,
	audioSec float64, wall time.Duration, startSec, endSec, confThreshold float64) runStats {

	if confThreshold <= 0 {
		confThreshold = defaultConfThreshold
	}

	s := runStats{
		SourceFile:    filepath.Base(inputPath),
		ModelUsed:     tag,
		AudioSeconds:  round3(audioSec),
		WallSeconds:   round3(wall.Seconds()),
		Segments:      len(segs),
		StartSec:      startSec,
		EndSec:        endSec,
		Truncated:     startSec > 0 || endSec > 0,
		ConfThreshold: confThreshold,
	}

	if wall > 0 && audioSec > 0 {
		s.RealtimeFactor = round3(audioSec / wall.Seconds())
	}

	var segSecondsTotal, longest float64
	var confSum float64
	var confCount int
	minConf := 1.0

	for _, seg := range segs {
		text := strings.TrimSpace(seg.Text)
		s.Characters += len(text)

		fields := strings.Fields(text)
		s.Words += len(fields)

		d := (seg.End - seg.Start).Seconds()
		segSecondsTotal += d
		if d > longest {
			longest = d
		}

		for _, t := range seg.Tokens {
			s.Tokens++
			if !isWordToken(t.Text) {
				continue
			}
			p := float64(t.Prob)
			confSum += p
			confCount++
			if p < minConf {
				minConf = p
			}
			if p < confThreshold {
				s.LowConfTokens++
			}
		}
	}

	if s.Segments > 0 {
		s.MeanSegmentS = round3(segSecondsTotal / float64(s.Segments))
	}
	s.LongestSegmentS = round3(longest)

	if audioSec > 0 {
		s.WordsPerMin = round3(float64(s.Words) / audioSec * 60.0)
		s.SegmentsPerM = round3(float64(s.Segments) / audioSec * 60.0)
	}

	if confCount > 0 {
		s.MeanTokenConf = round3(confSum / float64(confCount))
		s.MinTokenConf = round3(minConf)
	}

	return s
}

// statsPath returns the sidecar path for a transcript stem.
func statsPath(stem, tag string) string {
	return stem + "_" + tag + ".stats.json"
}

// writeStatsAtomic writes the stats sidecar via .part + rename, matching the
// transcript writers so a killed run leaves nothing that looks complete.
func writeStatsAtomic(path string, s runStats) error {
	buf, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	buf = append(buf, '\n')
	return writeFileAtomic(path, buf)
}

// appendIndex adds one line to the corpus index. Append-only so a
// re-transcription with a different model adds a row rather than replacing
// the previous experiment.
func appendIndex(dir string, s runStats) error {
	path := filepath.Join(dir, "transcripts.index.jsonl")

	line, err := json.Marshal(s)
	if err != nil {
		return err
	}

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(append(line, '\n')); err != nil {
		return err
	}
	return nil
}

// recordStats writes the sidecar and appends to the index. Failures are
// non-fatal to the transcription itself: the transcript is the product,
// the statistics are a convenience.
func recordStats(stem, dir string, s runStats) error {
	if err := writeStatsAtomic(statsPath(stem, s.ModelUsed), s); err != nil {
		return fmt.Errorf("stats sidecar: %w", err)
	}
	if err := appendIndex(dir, s); err != nil {
		return fmt.Errorf("stats index: %w", err)
	}
	return nil
}
