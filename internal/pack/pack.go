package pack

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"

	"github.com/mikebarb/stt/internal/format"
)

const (
	SchemaVersion = "stt-pack/1"
	// VersionTimeLayout is UTC yyyymmddhhmmss (14 digits): fixed-width,
	// lexicographically sortable, and stateless across machines.
	VersionTimeLayout    = "20060102150405"
	ToolName             = "github.com/mikebarb/stt"
	ToolVersion          = "0.2.0"
	DefaultConfThreshold = 0.35
)

type Pack struct {
	Schema      string          `json:"schema"`
	CreatedUTC  string          `json:"created_utc"`
	UpdatedUTC  string          `json:"updated_utc"`
	Tool        ToolInfo        `json:"tool"`
	Source      SourceInfo      `json:"source"`
	Model       ModelInfo       `json:"model"`
	Run         RunInfo         `json:"run"`
	Corrections CorrectionsInfo `json:"corrections"`
	Statistics  StatisticsInfo  `json:"statistics"`
	Audit       AuditInfo       `json:"audit"`
	Transcript  TranscriptInfo  `json:"transcript"`
	Loop        *LoopInfo       `json:"loop,omitempty"`
	Summary     *SummaryInfo    `json:"summary,omitempty"`
}

type ToolInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type SourceInfo struct {
	File            string  `json:"file"`
	AudioHash       string  `json:"audio_hash,omitempty"`
	SizeBytes       int64   `json:"size_bytes"`
	DurationSeconds float64 `json:"duration_seconds"`
}

type ModelInfo struct {
	Name      string `json:"name"`
	File      string `json:"file"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256,omitempty"`
}

type RunInfo struct {
	StartedUTC              string  `json:"started_utc"`
	FinishedUTC             string  `json:"finished_utc"`
	ElapsedSeconds          float64 `json:"elapsed_seconds"`
	AudioSeconds            float64 `json:"audio_seconds"`
	RealtimeRatio           float64 `json:"realtime_ratio"`
	Threads                 int     `json:"threads"`
	Sampling                string  `json:"sampling"`
	BeamSize                int     `json:"beam_size"`
	BestOf                  int     `json:"best_of"`
	Language                string  `json:"language"`
	Translate               bool    `json:"translate"`
	ConditionOnPreviousText bool    `json:"condition_on_previous_text"`
	StartOffsetSeconds      float64 `json:"start_offset_seconds"`
	EndOffsetSeconds        float64 `json:"end_offset_seconds"`
	Truncated               bool    `json:"truncated"`
}

type CorrectionApplied struct {
	Segment int    `json:"segment"`
	From    string `json:"from"`
	To      string `json:"to"`
	Rule    string `json:"rule"`
}

type CorrectionsInfo struct {
	File    string              `json:"file,omitempty"`
	SHA256  string              `json:"sha256,omitempty"`
	Total   int                 `json:"total"`
	Applied []CorrectionApplied `json:"applied,omitempty"`
}

type StatisticsInfo struct {
	Words                 int     `json:"words"`
	Characters            int     `json:"characters"`
	Segments              int     `json:"segments"`
	Tokens                int     `json:"tokens"`
	WordsPerMinute        float64 `json:"words_per_minute"`
	SegmentsPerMinute     float64 `json:"segments_per_minute"`
	MeanSegmentSeconds    float64 `json:"mean_segment_seconds"`
	LongestSegmentSeconds float64 `json:"longest_segment_seconds"`
	MeanTokenConfidence   float64 `json:"mean_token_confidence"`
	MinTokenConfidence    float64 `json:"min_token_confidence"`
	LowConfidenceTokens   int     `json:"low_confidence_tokens"`
	LowConfidencePer1k    float64 `json:"low_confidence_per_1k"`
	ConfidenceThreshold   float64 `json:"confidence_threshold"`
	LoopDensity           float64 `json:"loop_density"`
	RepetitionSeconds     float64 `json:"repetition_seconds"`
}

type FlaggedToken struct {
	Segment     int     `json:"segment"`
	Token       int     `json:"token"`
	Text        string  `json:"text"`
	Probability float32 `json:"probability"`
	Start       int64   `json:"t0"` // centiseconds
	End         int64   `json:"t1"` // centiseconds
}

type AuditInfo struct {
	Threshold    float64        `json:"threshold"`
	FlaggedCount int            `json:"flagged_count"`
	Flagged      []FlaggedToken `json:"flagged,omitempty"`
}

type Token struct {
	Text  string  `json:"t"`
	Prob  float32 `json:"p"`
	Start int64   `json:"t0"` // centiseconds
	End   int64   `json:"t1"` // centiseconds
}

type Segment struct {
	Index  int     `json:"i"`
	Start  float64 `json:"start"`
	End    float64 `json:"end"`
	Text   string  `json:"text"`
	Tokens []Token `json:"tokens,omitempty"`
}

type TranscriptInfo struct {
	Text         string    `json:"text"`
	TextSHA256   string    `json:"text_sha256"`
	SegmentCount int       `json:"segment_count"`
	Segments     []Segment `json:"segments"`
}

// SummaryInfo holds the Phase 3 summarisation output attached after
// transcription. SourceTextSHA256 links it to Transcript.TextSHA256 so stale
// abstracts are detected automatically after a model upgrade.
type SummaryInfo struct {
	Abstract         string   `json:"abstract"`
	KeyPoints        []string `json:"key_points,omitempty"`
	Topics           []string `json:"topics,omitempty"`
	Model            string   `json:"model,omitempty"`
	PromptVersion    string   `json:"prompt_version,omitempty"`
	GeneratedUTC     string   `json:"generated_utc,omitempty"`
	SourceTextSHA256 string   `json:"source_text_sha256,omitempty"`
	Reviewed         bool     `json:"reviewed"`
}

// BuildInput collects everything the transcriber knows at the end of a run.
type BuildInput struct {
	SourcePath          string
	SourceAudioHash     string
	SourceSizeBytes     int64
	FullDurationSeconds float64

	ModelName      string
	ModelPath      string
	ModelSizeBytes int64
	ModelSHA256    string

	StartedAt      time.Time
	FinishedAt     time.Time
	AudioSeconds   float64
	Threads        int
	StartOffsetSec float64
	EndOffsetSec   float64
	AuditThreshold float64

	CorrectionsFile    string
	CorrectionsSHA256  string
	CorrectionsApplied []CorrectionApplied

	Segments []format.Segment

	// Statistics optionally supplies pre-computed statistics so the pack and
	// the .stats.json sidecar can never disagree. When nil, Build computes
	// them from the segments.
	Statistics *StatisticsInfo

	NoContext bool
}

// Build assembles a complete stt-pack/1 document from a transcription run.
func Build(in BuildInput) *Pack {
	now := in.FinishedAt
	if now.IsZero() {
		now = time.Now().UTC()
	}
	started := in.StartedAt
	if started.IsZero() {
		started = now
	}
	elapsed := now.Sub(started).Seconds()
	if elapsed < 0 {
		elapsed = 0
	}

	fullDur := in.FullDurationSeconds
	if fullDur <= 0 {
		fullDur = in.AudioSeconds
	}

	ratio := 0.0
	if elapsed > 0 && in.AudioSeconds > 0 {
		ratio = round3(in.AudioSeconds / elapsed)
	}

	text := format.RenderText(in.Segments)
	textSum := sha256.Sum256([]byte(text))
	textSHA := hex.EncodeToString(textSum[:])

	segs := make([]Segment, 0, len(in.Segments))
	for i, s := range in.Segments {
		ps := Segment{
			Index: i,
			Start: float64(format.Cs(s.Start)) / 100.0,
			End:   float64(format.Cs(s.End)) / 100.0,
			Text:  strings.TrimSpace(s.Text),
		}
		if len(s.Tokens) > 0 {
			ps.Tokens = make([]Token, 0, len(s.Tokens))
			for _, t := range s.Tokens {
				ps.Tokens = append(ps.Tokens, Token{
					Text:  t.Text,
					Prob:  t.Prob,
					Start: t.Start,
					End:   t.End,
				})
			}
		}
		segs = append(segs, ps)
	}

	var stats StatisticsInfo
	if in.Statistics != nil {
		stats = *in.Statistics
	} else {
		stats = computeStatistics(in.Segments, in.AudioSeconds, in.AuditThreshold)
	}

	// Repetition analysis. Computed from the segments regardless of where the
	// statistics came from: the transcriber supplies a pre-computed
	// StatisticsInfo (from runStats) which cannot know about loops, so setting
	// these inside the else branch would leave transcriber-built packs
	// recording loop_density = 0 always.
	loop := DetectLoops(in.Segments, DefaultLoopN, DefaultLoopMinCount, DefaultLoopMinDensity)
	stats.LoopDensity = loop.Density
	stats.RepetitionSeconds = loop.RepetitionSeconds

	audit := computeAudit(in.Segments, in.AuditThreshold)

	var loopPtr *LoopInfo
	if loop.IsLoop() || len(loop.Spans) > 0 {
		lp := loop
		loopPtr = &lp
	}

	return &Pack{
		Schema:     SchemaVersion,
		CreatedUTC: now.UTC().Format(time.RFC3339),
		UpdatedUTC: now.UTC().Format(VersionTimeLayout),
		Tool: ToolInfo{
			Name:    ToolName,
			Version: ToolVersion,
		},
		Source: SourceInfo{
			File:            filepath.Base(in.SourcePath),
			AudioHash:       in.SourceAudioHash,
			SizeBytes:       in.SourceSizeBytes,
			DurationSeconds: round3(fullDur),
		},
		Model: ModelInfo{
			Name:      in.ModelName,
			File:      in.ModelPath,
			SizeBytes: in.ModelSizeBytes,
			SHA256:    in.ModelSHA256,
		},
		Run: RunInfo{
			StartedUTC:              started.UTC().Format(time.RFC3339),
			FinishedUTC:             now.UTC().Format(time.RFC3339),
			ElapsedSeconds:          round3(elapsed),
			AudioSeconds:            in.AudioSeconds,
			RealtimeRatio:           ratio,
			Threads:                 in.Threads,
			Sampling:                "beam_search",
			BeamSize:                5,
			BestOf:                  5,
			Language:                "en",
			Translate:               false,
			ConditionOnPreviousText: !in.NoContext,
			StartOffsetSeconds:      in.StartOffsetSec,
			EndOffsetSeconds:        in.EndOffsetSec,
			Truncated:               in.StartOffsetSec > 0 || in.EndOffsetSec > 0,
		},
		Corrections: CorrectionsInfo{
			File:    in.CorrectionsFile,
			SHA256:  in.CorrectionsSHA256,
			Total:   len(in.CorrectionsApplied),
			Applied: in.CorrectionsApplied,
		},
		Statistics: stats,
		Audit:      audit,
		Loop:       loopPtr,
		Transcript: TranscriptInfo{
			Text:         text,
			TextSHA256:   textSHA,
			SegmentCount: len(segs),
			Segments:     segs,
		},
	}
}

// ToFormatSegments converts the pack's stored segments back into the shared
// renderer type, preserving centisecond precision without float drift.
func (p *Pack) ToFormatSegments() []format.Segment {
	out := make([]format.Segment, 0, len(p.Transcript.Segments))
	for _, s := range p.Transcript.Segments {
		fs := format.Segment{
			Start: time.Duration(math.Round(s.Start*100)) * 10 * time.Millisecond,
			End:   time.Duration(math.Round(s.End*100)) * 10 * time.Millisecond,
			Text:  s.Text,
		}
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

// ExtractFormat renders any standard output format from the pack.
// For plain text, writes Transcript.Text directly (byte copy per D2).
func (p *Pack) ExtractFormat(w io.Writer, name string) error {
	switch strings.ToLower(name) {
	case "text", "txt":
		if p.Transcript.Text != "" {
			_, err := io.WriteString(w, p.Transcript.Text)
			return err
		}
	}
	dur := p.Run.AudioSeconds
	if dur <= 0 {
		dur = p.Source.DurationSeconds
	}
	return format.WriteOutput(w, name, p.ToFormatSegments(), p.Source.File, p.Model.Name, dur)
}

// ExtractAudit renders an audit report from the pack's retained tokens at
// the requested threshold (defaults to the pack's recorded threshold, or 0.20).
func (p *Pack) ExtractAudit(w io.Writer, threshold float64) (int, error) {
	if threshold <= 0 {
		threshold = p.Audit.Threshold
	}
	if threshold <= 0 {
		threshold = 0.20
	}
	return format.WriteAudit(w, p.ToFormatSegments(), threshold, p.Model.Name)
}

// SetSummary attaches or replaces the pack's summary block, linking it to
// the current transcript text checksum.
func (p *Pack) SetSummary(s SummaryInfo) {
	if s.SourceTextSHA256 == "" {
		s.SourceTextSHA256 = p.Transcript.TextSHA256
	}
	if s.GeneratedUTC == "" {
		s.GeneratedUTC = time.Now().UTC().Format(time.RFC3339)
	}
	p.Summary = &s
}

// IsSummaryStale reports whether an attached summary was generated against a
// different transcript text than the pack currently holds.
func (p *Pack) IsSummaryStale() bool {
	if p == nil || p.Summary == nil || p.Summary.SourceTextSHA256 == "" {
		return false
	}
	return p.Summary.SourceTextSHA256 != p.Transcript.TextSHA256
}

// Load reads and parses a pack JSON file from disk.
func Load(path string) (*Pack, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p Pack
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("pack %s: %w", path, err)
	}
	return &p, nil
}

// WriteAtomic writes the pack as indented JSON via a .part temp file + rename.
func WriteAtomic(path string, p *Pack) error {
	// Every tool-driven write advances UpdatedUTC automatically so no
	// mutation path can forget to bump the version stamp.
	p.UpdatedUTC = time.Now().UTC().Format(VersionTimeLayout)

	buf, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	buf = append(buf, '\n')

	tmp := path + ".part"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(buf); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// FileDigest returns the SHA-256 hex digest and byte size of path. Call once
// per model/corrections file in batch mode rather than per audio file.
func FileDigest(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return "", 0, err
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), info.Size(), nil
}

// ---------------------------------------------------------------------------
// Internal helpers (mirroring stats.go).
// ---------------------------------------------------------------------------

func round3(f float64) float64 {
	return float64(int64(f*1000+0.5)) / 1000
}

func isWordToken(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

func computeStatistics(segs []format.Segment, audioSec, confThreshold float64) StatisticsInfo {
	if confThreshold <= 0 {
		confThreshold = DefaultConfThreshold
	}
	s := StatisticsInfo{
		Segments:            len(segs),
		ConfidenceThreshold: confThreshold,
	}

	var segSecTotal, longest, confSum float64
	var confCount int
	minConf := 1.0

	for _, seg := range segs {
		txt := strings.TrimSpace(seg.Text)
		s.Characters += len(txt)
		s.Words += len(strings.Fields(txt))

		d := (seg.End - seg.Start).Seconds()
		segSecTotal += d
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
				s.LowConfidenceTokens++
			}
		}
	}

	if s.Segments > 0 {
		s.MeanSegmentSeconds = round3(segSecTotal / float64(s.Segments))
	}
	s.LongestSegmentSeconds = round3(longest)

	if audioSec > 0 {
		s.WordsPerMinute = round3(float64(s.Words) / audioSec * 60.0)
		s.SegmentsPerMinute = round3(float64(s.Segments) / audioSec * 60.0)
	}
	if s.Words > 0 {
		s.LowConfidencePer1k = round3(float64(s.LowConfidenceTokens) / float64(s.Words) * 1000.0)
	}
	if confCount > 0 {
		s.MeanTokenConfidence = round3(confSum / float64(confCount))
		s.MinTokenConfidence = round3(minConf)
	}
	return s
}

func computeAudit(segs []format.Segment, threshold float64) AuditInfo {
	if threshold <= 0 {
		return AuditInfo{}
	}
	var flagged []FlaggedToken
	for si, s := range segs {
		for ti, t := range s.Tokens {
			if float64(t.Prob) >= threshold {
				continue
			}
			flagged = append(flagged, FlaggedToken{
				Segment:     si,
				Token:       ti,
				Text:        strings.TrimSpace(t.Text),
				Probability: t.Prob,
				Start:       t.Start,
				End:         t.End,
			})
		}
	}
	return AuditInfo{
		Threshold:    threshold,
		FlaggedCount: len(flagged),
		Flagged:      flagged,
	}
}

// RenderSegments writes the plain-text rendering of the segments. This is
// the canonical text form; Transcript.Text must equal it.
func RenderSegments(w io.Writer, segs []format.Segment) error {
	return format.WriteText(w, segs)
}
