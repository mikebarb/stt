// Command sttpack reads and writes transcript packs (stt-pack/1).
//
// A pack is the complete record of one transcription: the transcript text,
// the segment timings, the per-token confidences, the run parameters and the
// statistics. Every other artefact — the plain text, the SRT, the audit
// report — is a rendering of the pack and can be regenerated from it.
//
// Usage:
//
//	sttpack info         <pack.json>
//	sttpack extract      <pack.json> --format text|timestamp|markers|srt|vtt|json [--out FILE]
//	sttpack audit        <pack.json> [--threshold 0.20] [--out FILE]
//	sttpack summary      <pack.json> [--out FILE]
//	sttpack set-summary  <pack.json> --abstract FILE [--model NAME] [--prompt-version V]
//	sttpack verify       <pack.json> [--format ...]
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/mikebarb/stt/internal/pack"
)

func usage() {
	fmt.Fprintf(os.Stderr, `usage: sttpack <command> <pack.json> [flags]

commands:
  info         show a summary of the pack
  extract      render a standard output format from the pack
  audit        render a low-confidence report from the pack
  summary      show the attached summary, if any
  set-summary  attach or replace the pack's summary block
  verify       check that every format regenerates identically

flags:
  --format NAME       text|timestamp|markers|srt|vtt|json
  --threshold FLOAT   audit probability threshold (default: the pack's own)
  --out FILE          write to FILE instead of stdout
  --abstract FILE     file containing the abstract text
  --model NAME        summarisation model that produced the abstract
  --prompt-version V  prompt version that produced the abstract
`)
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "sttpack: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return fmt.Errorf("no command given")
	}

	cmd := args[0]
	rest := args[1:]

	// Pull the pack path out: the first non-flag argument after the command.
	packPath := ""
	for _, a := range rest {
		if !strings.HasPrefix(a, "-") {
			packPath = a
			break
		}
	}
	if packPath == "" {
		usage()
		return fmt.Errorf("%s requires a pack file", cmd)
	}

	flags := parseFlags(rest)

	switch cmd {
	case "info":
		return cmdInfo(packPath)
	case "extract":
		return cmdExtract(packPath, flags)
	case "audit":
		return cmdAudit(packPath, flags)
	case "summary":
		return cmdSummary(packPath, flags)
	case "set-summary":
		return cmdSetSummary(packPath, flags)
	case "verify":
		return cmdVerify(packPath)
	case "recheck":
		return cmdRecheck(packPath, flags)
	default:
		usage()
		return fmt.Errorf("unknown command %q", cmd)
	}
}

// parseFlags collects --key value pairs. Unknown flags are ignored so the
// command set can grow without breaking older invocations.
func parseFlags(args []string) map[string]string {
	flags := map[string]string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !strings.HasPrefix(a, "--") {
			continue
		}
		key := strings.TrimPrefix(a, "--")
		if eq := strings.Index(key, "="); eq >= 0 {
			flags[key[:eq]] = key[eq+1:]
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
			flags[key] = args[i+1]
			i++
		} else {
			flags[key] = ""
		}
	}
	return flags
}

func cmdInfo(path string) error {
	p, err := pack.Load(path)
	if err != nil {
		return err
	}

	fmt.Printf("pack:        %s\n", path)
	fmt.Printf("schema:      %s\n", p.Schema)
	fmt.Printf("created:     %s\n", p.CreatedUTC)
	if p.UpdatedUTC != "" {
		fmt.Printf("updated:     %s\n", p.UpdatedUTC)
	}
	fmt.Printf("source:      %s\n", p.Source.File)
	if p.Source.AudioHash != "" {
		fmt.Printf("audio hash:  %s\n", p.Source.AudioHash)
	}
	fmt.Printf("duration:    %.1f s (%.1f min)\n",
		p.Source.DurationSeconds, p.Source.DurationSeconds/60)
	if p.Run.Truncated {
		fmt.Printf("truncated:   yes (%.0f–%.0f s of the source)\n",
			p.Run.StartOffsetSeconds, p.Run.StartOffsetSeconds+p.Run.AudioSeconds)
	}
	fmt.Printf("model:       %s\n", p.Model.Name)
	fmt.Printf("run:         %.1f s audio in %.1f s (%.2fx), %d threads\n",
		p.Run.AudioSeconds, p.Run.ElapsedSeconds, p.Run.RealtimeRatio, p.Run.Threads)
	fmt.Printf("segments:    %d\n", p.Statistics.Segments)
	fmt.Printf("words:       %d (%.1f wpm)\n",
		p.Statistics.Words, p.Statistics.WordsPerMinute)
	fmt.Printf("confidence:  mean %.3f, min %.3f, %d word tokens below %.2f\n",
		p.Statistics.MeanTokenConfidence, p.Statistics.MinTokenConfidence,
		p.Statistics.LowConfidenceTokens, p.Statistics.ConfidenceThreshold)
	if p.Audit.FlaggedCount != p.Statistics.LowConfidenceTokens {
		fmt.Printf("audit:       %d tokens flagged (all tokens, incl. punctuation)\n", p.Audit.FlaggedCount)
	}
	fmt.Printf("corrections: %d applied\n", p.Corrections.Total)

	if p.Summary != nil {
		state := "current"
		if p.IsSummaryStale() {
			state = "STALE (transcript has changed since it was written)"
		}
		fmt.Printf("summary:     %s\n", state)
		fmt.Printf("  model:     %s\n", p.Summary.Model)
		fmt.Printf("  prompt:    %s\n", p.Summary.PromptVersion)
		fmt.Printf("  generated: %s\n", p.Summary.GeneratedUTC)
		fmt.Printf("  abstract:  %s\n", oneLine(p.Summary.Abstract))
	} else {
		fmt.Printf("summary:     none\n")
	}
	return nil
}

func cmdExtract(path string, flags map[string]string) error {
	p, err := pack.Load(path)
	if err != nil {
		return err
	}

	name := flags["format"]
	if name == "" {
		name = "text"
	}

	out := flags["out"]
	if out == "" {
		return p.ExtractFormat(os.Stdout, name)
	}

	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	return p.ExtractFormat(f, name)
}

func cmdAudit(path string, flags map[string]string) error {
	p, err := pack.Load(path)
	if err != nil {
		return err
	}

	threshold := 0.0
	if s := flags["threshold"]; s != "" {
		var v float64
		if _, err := fmt.Sscanf(s, "%g", &v); err != nil {
			return fmt.Errorf("bad --threshold %q", s)
		}
		threshold = v
	}

	out := flags["out"]
	if out == "" {
		_, err := p.ExtractAudit(os.Stdout, threshold)
		return err
	}

	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = p.ExtractAudit(f, threshold)
	return err
}

func cmdSummary(path string, flags map[string]string) error {
	p, err := pack.Load(path)
	if err != nil {
		return err
	}
	if p.Summary == nil {
		return fmt.Errorf("no summary attached to this pack")
	}

	if p.IsSummaryStale() {
		fmt.Fprintf(os.Stderr, "warning: this summary was written against an earlier transcript\n")
	}

	text := p.Summary.Abstract
	if out := flags["out"]; out != "" {
		return os.WriteFile(out, []byte(text), 0o644)
	}
	fmt.Println(text)
	return nil
}

func cmdSetSummary(path string, flags map[string]string) error {
	abstractFile := flags["abstract"]
	if abstractFile == "" {
		return fmt.Errorf("--abstract FILE is required")
	}

	data, err := os.ReadFile(abstractFile)
	if err != nil {
		return err
	}

	p, err := pack.Load(path)
	if err != nil {
		return err
	}

	p.SetSummary(pack.SummaryInfo{
		Abstract:      strings.TrimSpace(string(data)),
		Model:         flags["model"],
		PromptVersion: flags["prompt-version"],
		Reviewed:      false,
	})

	if err := pack.WriteAtomic(path, p); err != nil {
		return err
	}
	fmt.Printf("summary written to %s\n", path)
	return nil
}

// cmdVerify re-renders every format from the pack and confirms the pack's
// stored transcript text matches the segment list. It cannot compare against
// the original files, but it proves the pack is internally consistent.
func cmdVerify(path string) error {
	p, err := pack.Load(path)
	if err != nil {
		return err
	}

	problems := 0

	if p.Schema != pack.SchemaVersion {
		fmt.Printf("schema:      %s (expected %s)\n", p.Schema, pack.SchemaVersion)
		problems++
	}
	if p.Transcript.SegmentCount != len(p.Transcript.Segments) {
		fmt.Printf("segments:    %d declared, %d present\n",
			p.Transcript.SegmentCount, len(p.Transcript.Segments))
		problems++
	}
	if p.Statistics.Segments != len(p.Transcript.Segments) {
		fmt.Printf("statistics:  %d segments, transcript has %d\n",
			p.Statistics.Segments, len(p.Transcript.Segments))
		problems++
	}

	// The stored text must equal the text rendered from the segments.
	var rendered strings.Builder
	if err := pack.RenderSegments(&rendered, p.ToFormatSegments()); err != nil {
		return err
	}
	if rendered.String() != p.Transcript.Text {
		fmt.Println("text:        stored transcript text does not match the segment list")
		problems++
	}

	// Every format must render without error.
	for _, name := range []string{"text", "timestamp", "markers", "srt", "vtt", "json"} {
		var b strings.Builder
		if err := p.ExtractFormat(&b, name); err != nil {
			fmt.Printf("format %-10s FAILED: %v\n", name, err)
			problems++
			continue
		}
		fmt.Printf("format %-10s ok (%d bytes)\n", name, b.Len())
	}

	if problems == 0 {
		fmt.Println("verify:      pack is internally consistent")
		return nil
	}
	return fmt.Errorf("%d problem(s) found", problems)
}

// cmdRecheck recomputes the repetition analysis for a pack and rewrites the
// pack in place.
//
// This exists so a diagnostic added after a corpus was produced can be
// applied retrospectively: the segments are already in the pack, so no
// re-transcription is needed.
func cmdRecheck(path string, flags map[string]string) error {
	p, err := pack.Load(path)
	if err != nil {
		return err
	}

	n := pack.DefaultLoopN
	minCount := pack.DefaultLoopMinCount
	minDensity := pack.DefaultLoopMinDensity

	if s := flags["n"]; s != "" {
		fmt.Sscanf(s, "%d", &n)
	}
	if s := flags["min-count"]; s != "" {
		fmt.Sscanf(s, "%d", &minCount)
	}
	if s := flags["min-density"]; s != "" {
		fmt.Sscanf(s, "%g", &minDensity)
	}

	loop := pack.DetectLoops(p.ToFormatSegments(), n, minCount, minDensity)

	p.Statistics.LoopDensity = loop.Density
	p.Statistics.RepetitionSeconds = loop.RepetitionSeconds

	if loop.IsLoop() || len(loop.Spans) > 0 {
		lp := loop
		p.Loop = &lp
	} else {
		p.Loop = nil
	}

	if err := pack.WriteAtomic(path, p); err != nil {
		return err
	}

	status := "ok"
	if loop.IsLoop() {
		status = "LOOP"
	}
	fmt.Printf("%-8s density %5.1f%%  count %4d  %s\n",
		status, loop.Density*100, loop.Count, filepath.Base(path))
	for _, s := range loop.Spans {
		fmt.Printf("         %s - %s  (%.0fs, %.0f wpm, trigger %s)\n",
			formatClock(s.Start), formatClock(s.End),
			s.End-s.Start, s.WordsPerMin, s.Trigger)
	}
	return nil
}

func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.TrimSpace(s)
	if len(s) > 120 {
		return s[:117] + "..."
	}
	return s
}

// formatClock renders seconds as HH:MM:SS.
func formatClock(seconds float64) string {
	total := int(seconds)
	h := total / 3600
	m := (total % 3600) / 60
	s := total % 60
	return fmt.Sprintf("%d:%02d:%02d", h, m, s)
}

// unused keeps the json import honest if the file is trimmed later.
var _ = json.Marshal
