package format

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func testSegments() []Segment {
	return []Segment{
		{
			Start: 0,
			End:   11320 * time.Millisecond,
			Text:  " First segment. ",
			Tokens: []Token{
				{Text: " First", Prob: 0.95, Start: 0, End: 50},
				{Text: " segment", Prob: 0.15, Start: 50, End: 1132},
			},
		},
		{
			Start: 62500 * time.Millisecond,
			End:   71040 * time.Millisecond,
			Text:  "Second minute.",
			Tokens: []Token{
				{Text: "Second", Prob: 0.20, Start: 6250, End: 6400}, // exactly at threshold 0.20 -> not flagged
				{Text: " minute", Prob: 0.19, Start: 6400, End: 7104},
			},
		},
	}
}

func TestWriteJSONPreservesCentisecondsAndBaseName(t *testing.T) {
	var buf bytes.Buffer
	err := WriteJSON(&buf, testSegments(), `C:\audio\talk___1.mp3`, "medium.en", 71.04)
	if err != nil {
		t.Fatal(err)
	}

	var got JSONTranscript
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.SourceFile != "talk___1.mp3" {
		t.Errorf("SourceFile = %q, want talk___1.mp3", got.SourceFile)
	}
	if got.Segments[0].End != 11.32 {
		t.Errorf("Segments[0].End = %v, want 11.32 (centisecond precision)", got.Segments[0].End)
	}
	if got.Segments[1].Start != 62.5 {
		t.Errorf("Segments[1].Start = %v, want 62.5", got.Segments[1].Start)
	}
}

func TestWriteFormats(t *testing.T) {
	segs := testSegments()

	// text
	if got := RenderText(segs); got != "First segment.\nSecond minute.\n" {
		t.Errorf("RenderText = %q", got)
	}

	// timestamp
	var ts bytes.Buffer
	if err := WriteTimestamped(&ts, segs); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(ts.String(), "[00:00:00.000 --> 00:00:11.320]  First segment.\n") {
		t.Errorf("unexpected timestamp output:\n%s", ts.String())
	}

	// markers
	var mk bytes.Buffer
	if err := WriteMarkers(&mk, segs); err != nil {
		t.Fatal(err)
	}
	wantMk := "(00:00)\nFirst segment.\n(00:01)\nSecond minute.\n"
	if mk.String() != wantMk {
		t.Errorf("markers:\n--- got ---\n%s--- want ---\n%s", mk.String(), wantMk)
	}

	// srt (comma for milliseconds)
	var srt bytes.Buffer
	if err := WriteSRT(&srt, segs); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(srt.String(), "1\n00:00:00,000 --> 00:00:11,320\nFirst segment.\n\n") {
		t.Errorf("unexpected SRT output:\n%s", srt.String())
	}

	// vtt (dot for milliseconds + WEBVTT header)
	var vtt bytes.Buffer
	if err := WriteVTT(&vtt, segs); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(vtt.String(), "WEBVTT\n1\n00:00:00.000 --> 00:00:11.320\n") {
		t.Errorf("unexpected VTT output:\n%s", vtt.String())
	}

	// unknown format rejected
	var bad bytes.Buffer
	if err := WriteOutput(&bad, "docx", segs, "a.mp3", "base.en", 10); err == nil {
		t.Error("expected error for unknown format")
	}
}

func TestWriteAudit(t *testing.T) {
	var buf bytes.Buffer
	n, err := WriteAudit(&buf, testSegments(), 0.20, "medium.en")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("flagged = %d, want 2 (strictly below 0.20)", n)
	}
	out := buf.String()
	if !strings.Contains(out, "# audit threshold p < 0.20  model medium.en\n") {
		t.Errorf("missing header:\n%s", out)
	}
	if !strings.Contains(out, "[00:00:00.500 -> 00:00:11.320] p=0.150 \"segment\"\n") {
		t.Errorf("missing flagged token:\n%s", out)
	}
	if !strings.Contains(out, "# 2 tokens flagged\n") {
		t.Errorf("missing footer:\n%s", out)
	}
}
