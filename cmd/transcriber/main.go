package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/mikebarb/stt/internal/audio"
	"github.com/mikebarb/stt/internal/whisper"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// Default (no -o, no -fmt): write the standard trio beside the input.
// -o <path>           : single file, format from -fmt (default text)
// -fmt <f> with no -o : single format to stdout
// -all                : write all six formats (adds srt, vtt, timestamp)

func run() error {
	var (
		modelPath   = flag.String("m", "models\\ggml-base.en.bin", "path to the ggml model file")
		outPath     = flag.String("o", "", "write transcript to this file (single output)")
		writeTxt    = flag.Bool("txt", false, "write the standard trio beside the input")
		all         = flag.Bool("all", false, "write all formats beside the input (implies -txt)")
		timestamps  = flag.Bool("ts", false, "include segment timestamps (deprecated: use -fmt timestamp)")
		threads     = flag.Int("t", 0, "worker threads (0 = number of CPUs)")
		quiet       = flag.Bool("q", false, "suppress whisper.cpp native logging")
		startSec    = flag.Float64("start", 0, "start offset in seconds (default 0)")
		endSec      = flag.Float64("end", 0, "stop after this many seconds of audio (0 = whole file)")
		fmtFlag     = flag.String("fmt", "text", "single-output format: text|timestamp|markers|srt|vtt|json")
		audit       = flag.Float64("audit", 0, "flag tokens with probability below this (0 = off, try 0.2)")
		doStats     = flag.Bool("stats", false, "write a .stats.json sidecar (always on with -txt/-all)")
		dirFlag     = flag.String("dir", "", "process every audio file in this directory")
		force       = flag.Bool("force", false, "re-transcribe files that already have outputs")
		onInterrupt = flag.String("on-interrupt", "ask", "Ctrl+C behaviour: ask|finish|abort|now")
		corrFile    = flag.String("corrections", "", "corrections rules file (default: corrections.txt beside the exe)")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s [flags] <audio-file>\n\nflags:\n",
			filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	// -corrections given explicitly means a missing file is a hard error;
	// when it is the default path, a missing file simply means no rules.
	corrExplicit := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "corrections" {
			corrExplicit = true
		}
	})

	corrPath := *corrFile
	if corrPath == "" {
		corrPath = defaultCorrectionsPath()
	}

	// -dir takes no positional argument; handle before the arity check.
	if *dirFlag != "" {
		if flag.NArg() != 0 {
			return fmt.Errorf("-dir takes no positional argument (unexpected: %q)", flag.Args())
		}
		return runBatch(batchConfig{
			dir:         *dirFlag,
			modelPath:   *modelPath,
			tag:         modelTag(*modelPath),
			threads:     *threads,
			quiet:       true,
			audit:       *audit,
			all:         *all,
			force:       *force,
			onInterrupt: *onInterrupt,
			startSec:    *startSec,
			endSec:      *endSec,
			corrections: corrPath,
		})
	}

	if flag.NArg() != 1 {
		flag.Usage()
		return fmt.Errorf("expected exactly one audio file")
	}
	inputPath := flag.Arg(0)

	if *outPath != "" && (*writeTxt || *all) {
		return fmt.Errorf("-o is mutually exclusive with -txt and -all")
	}

	// -ts maps onto the timestamp format for backwards compatibility.
	effFmt := strings.ToLower(*fmtFlag)
	if *timestamps && effFmt == "text" {
		effFmt = "timestamp"
	}
	switch effFmt {
	case "text", "timestamp", "markers", "srt", "vtt", "json":
	default:
		return fmt.Errorf("unknown -fmt %q", *fmtFlag)
	}

	tag := modelTag(*modelPath)

	// Rules before the model: a malformed file fails in milliseconds.
	corr, err := loadCorrections(corrPath, corrExplicit)
	if err != nil {
		return err
	}
	if n := len(corr.rules); n > 0 {
		fmt.Printf("corrections: %d rule(s) from %s\n", n, corrPath)
	}

	model, err := whisper.New(*modelPath, whisper.Options{
		Threads: *threads,
		UseGPU:  false,
		Quiet:   *quiet,
		Audit:   *audit > 0,
		Tokens:  *doStats || *audit > 0 || *writeTxt || *all,
	})
	if err != nil {
		return err
	}
	defer model.Close()

	samples, err := audio.Load(inputPath)
	if err != nil {
		return err
	}
	samples, err = sliceSamples(samples, *startSec, *endSec)
	if err != nil {
		return err
	}

	audioSeconds := float64(len(samples)) / 16000.0

	start := time.Now()
	segs, err := model.Transcribe(samples)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	// Corrections run before stats and before any writer, so every artefact
	// and every statistic describes the delivered text.
	if n := corr.ApplyToSegments(segs); n > 0 {
		fmt.Printf("corrections: %d replacement(s)\n", n)
	}

	switch {
	case *outPath != "":
		// --- single explicit file ---
		var buf bytes.Buffer
		if err := writeOutput(&buf, effFmt, segs, inputPath, tag, audioSeconds); err != nil {
			return err
		}
		if err := os.WriteFile(*outPath, buf.Bytes(), 0o644); err != nil {
			return err
		}
		fmt.Printf("wrote %s (%d segments, %d chars)\n", *outPath, len(segs), buf.Len())

		outStem := strings.TrimSuffix(*outPath, filepath.Ext(*outPath))

		if *audit > 0 {
			sidecar := outStem + ".audit.txt"
			n, err := writeAudit(sidecar, segs, *audit, tag)
			if err != nil {
				return err
			}
			fmt.Printf("audit: %d tokens below %.2f -> %s\n", n, *audit, sidecar)
		}

		if err := writeCorrectionsLog(outStem+".corrections.log",
			corr, tag, inputPath); err != nil {
			fmt.Fprintf(os.Stderr, "warning: corrections log: %v\n", err)
		}

		if *doStats {
			rs := computeStats(segs, inputPath, tag, audioSeconds, elapsed,
				*startSec, *endSec, *audit)
			if err := writeStatsAtomic(statsPath(outStem, tag), rs); err != nil {
				fmt.Fprintf(os.Stderr, "warning: stats sidecar: %v\n", err)
			}
			if err := appendIndex(filepath.Dir(inputPath), rs); err != nil {
				fmt.Fprintf(os.Stderr, "warning: stats index: %v\n", err)
			}
		}

	case *writeTxt || *all:
		// --- standard set beside the input ---
		stem := outputStem(inputPath, *startSec, *endSec)
		paths := trioPaths(stem, tag)
		if *all {
			paths[stem+"_"+tag+".srt"] = "srt"
			paths[stem+"_"+tag+".vtt"] = "vtt"
			paths[stem+"_"+tag+".timestamp.txt"] = "timestamp"
		}

		for path, format := range paths {
			var buf bytes.Buffer
			if err := writeOutput(&buf, format, segs, inputPath, tag, audioSeconds); err != nil {
				return err
			}
			if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
				return err
			}
			fmt.Printf("wrote %s (%d chars)\n", path, buf.Len())
		}

		if *audit > 0 {
			sidecar := stem + "_" + tag + ".audit.txt"
			n, err := writeAudit(sidecar, segs, *audit, tag)
			if err != nil {
				return err
			}
			fmt.Printf("audit: %d tokens below %.2f -> %s\n", n, *audit, sidecar)
		}

		if err := writeCorrectionsLog(stem+"_"+tag+".corrections.log",
			corr, tag, inputPath); err != nil {
			fmt.Fprintf(os.Stderr, "warning: corrections log: %v\n", err)
		}

		rs := computeStats(segs, inputPath, tag, audioSeconds, elapsed,
			*startSec, *endSec, *audit)
		if err := recordStats(stem, filepath.Dir(inputPath), rs); err != nil {
			fmt.Fprintf(os.Stderr, "warning: stats: %v\n", err)
		} else {
			fmt.Printf("wrote stats %s\n", statsPath(stem, tag))
		}

	default:
		// --- stdout: no side files, report to stderr ---
		var buf bytes.Buffer
		if err := writeOutput(&buf, effFmt, segs, inputPath, tag, audioSeconds); err != nil {
			return err
		}
		os.Stdout.Write(buf.Bytes())

		var rep strings.Builder
		corr.Report(&rep, tag, filepath.Base(inputPath))
		if rep.Len() > 0 {
			fmt.Fprint(os.Stderr, rep.String())
		}
	}

	fmt.Printf("%.1fs audio in %.2fs (%.2fx realtime)\n",
		audioSeconds, elapsed.Seconds(), audioSeconds/elapsed.Seconds())

	return nil
}

var (
	// Matches "ggml-<name>.bin", case-insensitively.
	modelNameRe = regexp.MustCompile(`(?i)^ggml-(.+)\.bin$`)

	// Anything that is not alphanumeric, dot, underscore or hyphen.
	unsafeFileCharsRe = regexp.MustCompile(`[^A-Za-z0-9._-]+`)
)

// modelTag returns a short, filesystem-safe identifier for a model file.
//
//	"models\ggml-base.en.bin"        -> "base.en"
//	"models\ggml-small.en-q5_1.bin"  -> "small.en-q5_1"
//	"/path/to/custom-model.bin"      -> "custom-model"
//	"C:\Models\ggml-medium.en.bin"   -> "medium.en"
func modelTag(modelPath string) string {
	base := filepath.Base(modelPath)

	// Preferred: recognise the upstream naming convention and drop the
	// "ggml-" prefix and ".bin" suffix.
	if m := modelNameRe.FindStringSubmatch(base); m != nil {
		base = m[1]
	} else {
		// Fallback: keep whatever precedes the final extension.
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}

	// Defensive: model files can be named arbitrarily, and a name is about
	// to become part of a path component.
	tag := unsafeFileCharsRe.ReplaceAllString(base, "-")
	tag = strings.Trim(tag, "-.")

	if tag == "" {
		return "model"
	}
	return tag
}

// outputs maps default-trio filenames to their format names.
func trioPaths(stem, tag string) map[string]string {
	base := stem + "_" + tag
	return map[string]string{
		base + ".txt":         "text",
		base + ".markers.txt": "markers",
		base + ".json":        "json",
	}
}

// defaultCorrectionsPath looks for corrections.txt beside the executable,
// then in the working directory. Missing files are tolerated.
func defaultCorrectionsPath() string {
	exe, err := os.Executable()
	if err == nil {
		p := filepath.Join(filepath.Dir(exe), "corrections.txt")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "corrections.txt"
}

// writeAudit emits a review queue: every token whose probability fell
// below the threshold, with its own timestamps so a reviewer can jump
// straight to that word in Audacity.
func writeAuditWriter(w io.Writer, segs []whisper.Segment, threshold float64, modelUsed string) (int, error) {
	flagged := 0
	fmt.Fprintf(w, "# audit threshold p < %.2f  model %s\n", threshold, modelUsed)
	for _, s := range segs {
		for _, t := range s.Tokens {
			if float64(t.Prob) >= threshold {
				continue
			}
			flagged++
			fmt.Fprintf(w, "[%s -> %s] p=%.3f %q\n",
				csStamp(t.Start), csStamp(t.End), t.Prob, strings.TrimSpace(t.Text))
		}
	}
	fmt.Fprintf(w, "# %d tokens flagged\n", flagged)
	return flagged, nil
}

// keep the file-based form for the single-file path
func writeAudit(path string, segs []whisper.Segment, threshold float64, modelUsed string) (int, error) {
	f, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return writeAuditWriter(f, segs, threshold, modelUsed)
}
