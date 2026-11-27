package main

import (
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/mikebarb/stt/internal/audiohash"
	"github.com/mikebarb/stt/internal/pack"
	"github.com/mikebarb/stt/internal/whisper"
)

// packContext holds the per-run values that are identical for every file in a
// batch: the model, its digest, and the corrections file.
type packContext struct {
	modelPath string
	tag       string
	corrPath  string
	threads   int
	noContext bool

	modelSize int64
	modelSHA  string
	corrSHA   string
	digests   bool
}

// packFile carries the per-file values needed to build one pack.
type packFile struct {
	InputPath    string
	FullAudioSec float64
	AudioSec     float64
	StartSec     float64
	EndSec       float64
	AuditThresh  float64
	Started      time.Time
	Finished     time.Time
	Segments     []whisper.Segment
	Stats        runStats
	Applied      []pack.CorrectionApplied
}

// resolvedThreads mirrors what whisper.New does internally, so the pack
// records the thread count actually used rather than the flag value.
func resolvedThreads(n int) int {
	if n > 0 {
		return n
	}
	return runtime.NumCPU()
}

func newPackContext(modelPath, tag, corrPath string, threads int, noContext bool) *packContext {
	return &packContext{
		modelPath: modelPath,
		tag:       tag,
		corrPath:  corrPath,
		threads:   resolvedThreads(threads),
		noContext: noContext,
	}
}

// ensureDigests computes the model and corrections digests once per run.
//
// They are provenance rather than correctness, so a failure leaves the field
// empty instead of failing the run. Hashing a 1.5 GB model takes a few
// seconds; doing it once per batch rather than once per file matters.
func (c *packContext) ensureDigests() {
	if c.digests {
		return
	}
	c.digests = true

	if sha, size, err := pack.FileDigest(c.modelPath); err == nil {
		c.modelSHA = sha
		c.modelSize = size
	}
	if c.corrPath != "" {
		if sha, _, err := pack.FileDigest(c.corrPath); err == nil {
			c.corrSHA = sha
		}
	}
}

// packStats maps the run statistics onto the pack's statistics block, so the
// pack and the .stats.json sidecar can never disagree.
func packStats(rs runStats) pack.StatisticsInfo {
	s := pack.StatisticsInfo{
		Words:                 rs.Words,
		Characters:            rs.Characters,
		Segments:              rs.Segments,
		Tokens:                rs.Tokens,
		WordsPerMinute:        rs.WordsPerMin,
		SegmentsPerMinute:     rs.SegmentsPerM,
		MeanSegmentSeconds:    rs.MeanSegmentS,
		LongestSegmentSeconds: rs.LongestSegmentS,
		MeanTokenConfidence:   rs.MeanTokenConf,
		MinTokenConfidence:    rs.MinTokenConf,
		LowConfidenceTokens:   rs.LowConfTokens,
		ConfidenceThreshold:   rs.ConfThreshold,
	}
	if s.Words > 0 {
		s.LowConfidencePer1k = round3(float64(s.LowConfidenceTokens) / float64(s.Words) * 1000)
	}
	return s
}

// write builds the pack for one file and writes it atomically.
//
// The audio hash is taken from the source MP3 before any -start/-end slicing,
// so it matches the value held by the catalogue for that lecture.
func (c *packContext) write(path string, f packFile) (*pack.Pack, error) {
	c.ensureDigests()

	audioHash, err := audiohash.CalculateAudioHash(f.InputPath)
	if err != nil {
		return nil, fmt.Errorf("audio hash: %w", err)
	}

	var size int64
	if fi, err := os.Stat(f.InputPath); err == nil {
		size = fi.Size()
	}

	finished := f.Finished
	if finished.IsZero() {
		finished = f.Started
	}

	p := pack.Build(pack.BuildInput{
		SourcePath:          f.InputPath,
		SourceAudioHash:     audioHash,
		SourceSizeBytes:     size,
		FullDurationSeconds: f.FullAudioSec,

		ModelName:      c.tag,
		ModelPath:      c.modelPath,
		ModelSizeBytes: c.modelSize,
		ModelSHA256:    c.modelSHA,

		StartedAt:          f.Started,
		FinishedAt:         finished,
		AudioSeconds:       f.AudioSec,
		Threads:            c.threads,
		StartOffsetSec:     f.StartSec,
		EndOffsetSec:       f.EndSec,
		AuditThreshold:     f.AuditThresh,
		CorrectionsFile:    c.corrPath,
		CorrectionsSHA256:  c.corrSHA,
		CorrectionsApplied: f.Applied,

		Segments:   toFormatSegments(f.Segments),
		Statistics: statsPtr(f.Stats),
		NoContext:  c.noContext,
	})

	if err := pack.WriteAtomic(path, p); err != nil {
		return nil, err
	}
	return p, nil
}

func statsPtr(rs runStats) *pack.StatisticsInfo {
	s := packStats(rs)
	return &s
}
