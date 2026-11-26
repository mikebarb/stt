package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mikebarb/stt/internal/audio"
	"github.com/mikebarb/stt/internal/whisper"
)

// audioExts are the container formats the miniaudio decode path handles.
var audioExts = []string{".mp3", ".wav", ".flac", ".ogg", ".m4a"}

// batchConfig is the whole set of knobs for a directory run.
type batchConfig struct {
	dir         string
	modelPath   string
	tag         string
	threads     int
	quiet       bool
	audit       float64
	all         bool
	force       bool
	onInterrupt string // finish | abort | ask | now
	startSec    float64
	endSec      float64
	corrections string // path to the rules file; empty means none
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

// expectedOutputs lists the artefacts a completed run leaves behind. Used both
// to decide whether a file is already done (resume) and by the writer, so the
// two can never disagree about filenames.
func expectedOutputs(stem, tag string, all bool) map[string]string {
	base := stem + "_" + tag
	m := map[string]string{
		base + ".txt":         "text",
		base + ".markers.txt": "markers",
		base + ".json":        "json",
	}
	if all {
		m[base+".srt"] = "srt"
		m[base+".vtt"] = "vtt"
		m[base+".timestamp.txt"] = "timestamp"
	}
	return m
}

// discover returns the audio files in cfg.dir that still need transcribing.
//
// A file counts as complete only when *every* expected artefact for the
// current model tag exists. That makes resume safe and makes changing models
// a genuine re-run rather than a silent skip. -force overrides.
func discover(cfg batchConfig) ([]string, error) {
	entries, err := os.ReadDir(cfg.dir)
	if err != nil {
		return nil, fmt.Errorf("read directory: %w", err)
	}

	// Remove partials left behind by a killed run. They are never output.
	if stale, _ := filepath.Glob(filepath.Join(cfg.dir, "*.part")); len(stale) > 0 {
		for _, p := range stale {
			os.Remove(p)
		}
	}

	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if !contains(audioExts, ext) {
			continue
		}
		path := filepath.Join(cfg.dir, e.Name())
		stem := outputStem(path, cfg.startSec, cfg.endSec)

		if !cfg.force && allExist(expectedOutputs(stem, cfg.tag, cfg.all)) {
			continue
		}
		out = append(out, path)
	}
	sort.Strings(out)
	return out, nil
}

func outputStem(inputPath string, startSec, endSec float64) string {
	stem := strings.TrimSuffix(inputPath, filepath.Ext(inputPath)) // <-- this is the line you already have
	if startSec > 0 || endSec > 0 {
		if endSec > 0 {
			stem = fmt.Sprintf("%s_%04.0f-%04.0f", stem, startSec, startSec+endSec)
		} else {
			stem = fmt.Sprintf("%s_%04.0f-end", stem, startSec)
		}
	}
	return stem
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

func allExist(m map[string]string) bool {
	for p := range m {
		if _, err := os.Stat(p); err != nil {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Atomic write
// ---------------------------------------------------------------------------

// writeFileAtomic writes to <path>.part and renames on success. A killed run
// therefore never leaves a file that looks like output.
func writeFileAtomic(path string, data []byte) error {
	tmp := path + ".part"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// ---------------------------------------------------------------------------
// Shared state for the interrupt handler
// ---------------------------------------------------------------------------

type batchState struct {
	mu       sync.Mutex
	current  string
	started  time.Time
	audioSec float64
	avgRatio float64 // elapsed seconds per audio second, from finished files
	haveAvg  bool
	model    *whisper.Model
}

func (s *batchState) begin(path string, audioSec float64) {
	s.mu.Lock()
	s.current = filepath.Base(path)
	s.started = time.Now()
	s.audioSec = audioSec
	s.mu.Unlock()
}

func (s *batchState) finish(audioSec, elapsedSec float64) {
	s.mu.Lock()
	if audioSec > 0 {
		r := elapsedSec / audioSec
		if !s.haveAvg {
			s.avgRatio, s.haveAvg = r, true
		} else {
			s.avgRatio = 0.7*s.avgRatio + 0.3*r // smoothed
		}
	}
	s.current = ""
	s.audioSec = 0
	s.mu.Unlock()
}

// status describes the current file: elapsed, expected total, and how far
// through the expected duration we are.
func (s *batchState) status() (name string, elapsed, expected time.Duration, pct float64, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current == "" {
		return "", 0, 0, 0, false
	}
	elapsed = time.Since(s.started)
	if s.audioSec > 0 && s.haveAvg {
		expected = time.Duration(s.audioSec*s.avgRatio*1000) * time.Millisecond
		if expected > 0 {
			pct = float64(elapsed) / float64(expected) * 100
		}
	}
	return s.current, elapsed, expected, pct, true
}

// ---------------------------------------------------------------------------
// Batch runner
// ---------------------------------------------------------------------------

func runBatch(cfg batchConfig) error {
	logPath := filepath.Join(cfg.dir,
		"transcribe-"+time.Now().Format("20060102-150405")+".log")
	lf, err := os.Create(logPath)
	if err != nil {
		return fmt.Errorf("create log: %w", err)
	}
	defer lf.Close()

	// Console output is mirrored verbatim into the log.
	logger := log.New(io.MultiWriter(os.Stdout, lf), "", log.LstdFlags)

	files, err := discover(cfg)
	if err != nil {
		return err
	}
	logger.Printf("batch: model=%s threads=%d files=%d dir=%s",
		cfg.tag, cfg.threads, len(files), cfg.dir)
	logger.Printf("batch: log file %s", logPath)

	if len(files) == 0 {
		logger.Printf("batch: nothing to do")
		return nil
	}

	corr, err := loadCorrections(cfg.corrections, false)
	if err != nil {
		return err
	}
	if n := len(corr.rules); n > 0 {
		logger.Printf("batch: %d correction rule(s) from %s", n, cfg.corrections)
	}

	model, err := whisper.New(cfg.modelPath, whisper.Options{
		Threads: cfg.threads,
		Quiet:   cfg.quiet,
		Audit:   true, // tokens needed for stats, not just the audit file
	})
	if err != nil {
		return err
	}
	defer model.Close()

	st := &batchState{model: model}
	var stopAfterCurrent int32
	var abortNow int32

	sigs := make(chan os.Signal, 4)
	signal.Notify(sigs, os.Interrupt)
	defer signal.Stop(sigs)
	go interruptHandler(sigs, logger, cfg, st, &stopAfterCurrent, &abortNow)

	total := time.Now()
	var done, skipped, failed int

	for _, path := range files {
		if atomic.LoadInt32(&abortNow) != 0 {
			break
		}
		if atomic.LoadInt32(&stopAfterCurrent) != 0 {
			logger.Printf("batch: stopping before %s (interrupt received)", filepath.Base(path))
			break
		}

		logger.Printf("start %s", path)
		model.ClearAbort()

		samples, err := audio.Load(path)
		if err != nil {
			logger.Printf("error %s: %v", filepath.Base(path), err)
			failed++
			continue
		}

		// honours -start / -end
		samples, err = sliceSamples(samples, cfg.startSec, cfg.endSec)
		if err != nil {
			logger.Printf("skip %s: %v", filepath.Base(path), err)
			failed++
			continue
		}

		audioSec := float64(len(samples)) / 16000.0
		st.begin(path, audioSec)

		if _, _, expected, _, ok := st.status(); ok && expected > 0 {
			logger.Printf("  audio %.1f min, estimated %s at %.2fx",
				audioSec/60.0, roundMin(expected), 1.0/st.avgRatioOr(1))
		} else {
			logger.Printf("  audio %.1f min", audioSec/60.0)
		}

		t0 := time.Now()
		segs, err := model.Transcribe(samples)
		elapsed := time.Since(t0)
		st.finish(audioSec, elapsed.Seconds())

		if errors.Is(err, whisper.ErrAborted) {
			logger.Printf("aborted %s after %s — no output written",
				filepath.Base(path), roundSec(elapsed))
			break
		}
		if err != nil {
			logger.Printf("error %s: %v", filepath.Base(path), err)
			failed++
			continue
		}

		corr.Reset()
		if n := corr.ApplyToSegments(segs); n > 0 {
			logger.Printf("  corrections: %d replacement(s)", n)
		}

		// --- new: compute statistics here, where elapsed and logger live ---
		rs := computeStats(segs, path, cfg.tag, audioSec, elapsed,
			cfg.startSec, cfg.endSec, cfg.audit)

		written, err := writeAll(segs, path, cfg, audioSec, rs, corr)
		if err != nil {
			logger.Printf("error writing %s: %v", filepath.Base(path), err)
			failed++
			continue
		}

		done++
		logger.Printf("done %s: %d segs, %.1f min audio in %s (%.2fx), wpm %.0f, %.1f seg/min, conf %.3f, %d files",
			filepath.Base(path), len(segs), audioSec/60.0, roundSec(elapsed),
			audioSec/elapsed.Seconds(), rs.WordsPerMin, rs.SegmentsPerM, rs.MeanTokenConf, written)
	}

	logger.Printf("batch complete: %d done, %d skipped, %d failed, total %s",
		done, skipped, failed, roundSec(time.Since(total)))
	return nil
}

// sliceSamples applies -start/-end, in seconds, to decoded 16 kHz mono audio.
func sliceSamples(samples []float32, startSec, endSec float64) ([]float32, error) {
	const rate = 16000

	if startSec > 0 {
		off := int(startSec * rate)
		if off >= len(samples) {
			return nil, fmt.Errorf("start offset %.1fs is beyond the end of the audio", startSec)
		}
		samples = samples[off:]
	}
	if endSec > 0 {
		n := int(endSec * rate)
		if n < len(samples) {
			samples = samples[:n]
		}
	}
	if len(samples) < rate {
		return nil, fmt.Errorf("audio segment is too short to transcribe")
	}
	return samples, nil
}

// avgRatioOr returns the smoothed ratio, or a plausible default.
func (s *batchState) avgRatioOr(def float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.haveAvg {
		return s.avgRatio
	}
	return def
}

// writeAll renders every expected artefact for one file, atomically.
func writeAll(segs []whisper.Segment, inputPath string, cfg batchConfig,
	audioSec float64, rs runStats, corr *corrector) (int, error) {

	stem := outputStem(inputPath, cfg.startSec, cfg.endSec)
	outputs := expectedOutputs(stem, cfg.tag, cfg.all)

	if err := writeFormats(segs, inputPath, cfg, audioSec, outputs); err != nil {
		return 0, err
	}

	if cfg.audit > 0 {
		sidecar := stem + "_" + cfg.tag + ".audit.txt"
		var b strings.Builder
		if _, err := writeAuditWriter(&b, segs, cfg.audit, cfg.tag); err != nil {
			return 0, err
		}
		if err := writeFileAtomic(sidecar, []byte(b.String())); err != nil {
			return 0, err
		}
	}

	// Corrections log — only written when at least one rule fired.
	if err := writeCorrectionsLog(stem+"_"+cfg.tag+".corrections.log", corr, cfg.tag, inputPath); err != nil {
		return len(outputs), err
	}

	// Diagnostics last: a stats failure must never cost a completed transcript.
	if err := recordStats(stem, cfg.dir, rs); err != nil {
		return len(outputs), err
	}
	return len(outputs), nil
}

// writeFormats is the previous loop body, extracted so writeAll stays readable.
func writeFormats(segs []whisper.Segment, inputPath string,
	cfg batchConfig, audioSec float64, outputs map[string]string) error {

	for path, format := range outputs {
		var b strings.Builder
		if err := writeOutput(&b, format, segs, inputPath, cfg.tag, audioSec); err != nil {
			return err
		}
		if err := writeFileAtomic(path, []byte(b.String())); err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Interrupt handling
// ---------------------------------------------------------------------------

func interruptHandler(sigs <-chan os.Signal, logger *log.Logger, cfg batchConfig,
	st *batchState, stopAfterCurrent, abortNow *int32) {

	<-sigs

	mode := cfg.onInterrupt
	interactive := isTerminal(os.Stdin)

	name, elapsed, expected, pct, active := st.status()
	if active {
		logger.Printf("interrupt: %s has run %s of an estimated %s (%.0f%% of expected time)",
			name, roundSec(elapsed), roundMin(expected), pct)
	} else {
		logger.Printf("interrupt: between files")
	}

	if mode == "ask" && !interactive {
		logger.Printf("interrupt: stdin is not a terminal, defaulting to 'finish'")
		mode = "finish"
	}

	switch mode {
	case "now":
		logger.Printf("interrupt: exiting immediately")
		os.Exit(130)

	case "abort":
		logger.Printf("interrupt: aborting current file and stopping")
		atomic.StoreInt32(abortNow, 1)
		st.requestAbort()

	case "ask":
		fmt.Fprint(os.Stderr, "\n[c]omplete this file then stop  [a]bort it and stop  [q]uit now > ")
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "a":
			logger.Printf("interrupt: aborting current file and stopping")
			atomic.StoreInt32(abortNow, 1)
			st.requestAbort()
		case "q":
			logger.Printf("interrupt: exiting immediately")
			os.Exit(130)
		default:
			logger.Printf("interrupt: completing current file, then stopping")
			atomic.StoreInt32(stopAfterCurrent, 1)
		}

	default: // finish
		logger.Printf("interrupt: completing current file, then stopping")
		atomic.StoreInt32(stopAfterCurrent, 1)
	}

	// A second Ctrl+C always exits, whatever was chosen above.
	<-sigs
	logger.Printf("interrupt: forcing exit")
	os.Exit(130)
}

func (s *batchState) requestAbort() {
	s.mu.Lock()
	m := s.model
	s.mu.Unlock()
	if m != nil {
		m.RequestAbort()
	}
}

func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func roundSec(d time.Duration) string {
	d = d.Round(time.Second)
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	sec := int(d.Seconds()) % 60
	if h > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", h, m, sec)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, sec)
	}
	return fmt.Sprintf("%ds", sec)
}

func roundMin(d time.Duration) string {
	if d <= 0 {
		return "unknown"
	}
	return fmt.Sprintf("%.0f min", d.Minutes())
}
