package pack

import (
	"bytes"
	"path/filepath"
	"testing"
	"time"

	"github.com/mikebarb/stt/internal/format"
)

func sampleSegments() []format.Segment {
	return []format.Segment{
		{
			Start: 0,
			End:   11320 * time.Millisecond,
			Text:  " And so my fellow Americans, ask not.",
			Tokens: []format.Token{
				{Text: " And", Prob: 0.98, Start: 0, End: 50},
				{Text: " fellow", Prob: 0.15, Start: 200, End: 260},
				{Text: " Americans", Prob: 0.42, Start: 260, End: 400},
			},
		},
		{
			Start: 62500 * time.Millisecond,
			End:   71040 * time.Millisecond,
			Text:  "Second minute segment.",
			Tokens: []format.Token{
				{Text: "Second", Prob: 0.91, Start: 6250, End: 6350},
				{Text: " segment", Prob: 0.18, Start: 6900, End: 7104},
			},
		},
	}
}

func TestPackRoundTripByteIdentical(t *testing.T) {
	segs := sampleSegments()
	src := `C:\audio\talk___1.mp3`
	model := "medium.en"
	audioSec := 71.04

	p := Build(BuildInput{
		SourcePath:          src,
		SourceAudioHash:     "deadbeef",
		SourceSizeBytes:     123456,
		FullDurationSeconds: audioSec,
		ModelName:           model,
		ModelPath:           `models\ggml-medium.en.bin`,
		StartedAt:           time.Now().UTC().Add(-10 * time.Second),
		FinishedAt:          time.Now().UTC(),
		AudioSeconds:        audioSec,
		Threads:             6,
		AuditThreshold:      0.20,
		Segments:            segs,
	})

	packPath := filepath.Join(t.TempDir(), "talk___1_medium.en.stt.json")
	if err := WriteAtomic(packPath, p); err != nil {
		t.Fatalf("WriteAtomic: %v", err)
	}

	loaded, err := Load(packPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	for _, fmtName := range []string{"text", "timestamp", "markers", "srt", "vtt", "json"} {
		var want, got bytes.Buffer
		if err := format.WriteOutput(&want, fmtName, segs, src, model, audioSec); err != nil {
			t.Fatalf("direct %s: %v", fmtName, err)
		}
		if err := loaded.ExtractFormat(&got, fmtName); err != nil {
			t.Fatalf("extract %s: %v", fmtName, err)
		}
		if !bytes.Equal(want.Bytes(), got.Bytes()) {
			t.Errorf("format %s mismatch:\n--- want ---\n%s\n--- got ---\n%s",
				fmtName, want.String(), got.String())
		}
	}

	// Audit byte-identical at both original threshold (0.20) and a new threshold (0.50).
	for _, thr := range []float64{0.20, 0.50} {
		var want, got bytes.Buffer
		if _, err := format.WriteAudit(&want, segs, thr, model); err != nil {
			t.Fatal(err)
		}
		if _, err := loaded.ExtractAudit(&got, thr); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(want.Bytes(), got.Bytes()) {
			t.Errorf("audit (%.2f) mismatch:\n--- want ---\n%s\n--- got ---\n%s",
				thr, want.String(), got.String())
		}
	}
}

func TestPackSummaryAndStaleness(t *testing.T) {
	p := Build(BuildInput{
		SourcePath:   "talk.mp3",
		ModelName:    "base.en",
		AudioSeconds: 12,
		Segments:     sampleSegments(),
	})

	if p.Summary != nil {
		t.Fatal("expected nil summary on fresh pack")
	}

	p.SetSummary(SummaryInfo{
		Abstract:      "A concise overview of the lecture.",
		Model:         "qwen2.5:7b",
		PromptVersion: "abstract-v1",
	})
	if p.IsSummaryStale() {
		t.Fatal("freshly attached summary should not be stale")
	}

	path := filepath.Join(t.TempDir(), "p.stt.json")
	if err := WriteAtomic(path, p); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Summary == nil || loaded.Summary.Abstract != "A concise overview of the lecture." {
		t.Fatalf("summary did not survive round-trip: %+v", loaded.Summary)
	}
	if loaded.IsSummaryStale() {
		t.Fatal("loaded summary should not be stale")
	}

	// Simulate transcript text changing on a model upgrade.
	loaded.Transcript.TextSHA256 = "different-hash"
	if !loaded.IsSummaryStale() {
		t.Fatal("summary should be flagged stale when transcript hash changes")
	}
}

func TestPackUpdatedUTCStampedOnWrite(t *testing.T) {
	p := Build(BuildInput{
		SourcePath:   "talk.mp3",
		ModelName:    "base.en",
		AudioSeconds: 12,
		Segments:     sampleSegments(),
	})

	path := filepath.Join(t.TempDir(), "p.stt.json")
	if err := WriteAtomic(path, p); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}

	if len(loaded.UpdatedUTC) != 14 {
		t.Fatalf("UpdatedUTC = %q, want 14-digit yyyymmddhhmmss", loaded.UpdatedUTC)
	}
	if _, err := time.Parse(VersionTimeLayout, loaded.UpdatedUTC); err != nil {
		t.Fatalf("UpdatedUTC %q failed parse: %v", loaded.UpdatedUTC, err)
	}
	if loaded.CreatedUTC != p.CreatedUTC {
		t.Errorf("CreatedUTC changed on write: got %q, want %q", loaded.CreatedUTC, p.CreatedUTC)
	}
}
