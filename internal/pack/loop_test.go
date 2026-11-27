package pack

import (
	"fmt"
	"testing"
	"time"

	"github.com/mikebarb/stt/internal/format"
)

func seg(start, end int, text string) format.Segment {
	return format.Segment{
		Start: time.Duration(start) * time.Second,
		End:   time.Duration(end) * time.Second,
		Text:  text,
	}
}

// loopingSegs builds a transcript that repeats one phrase over most of its
// length, with real speech either side.
func loopingSegs() []format.Segment {
	var out []format.Segment
	out = append(out, seg(0, 10,
		"The lecture begins here with the speaker talking normally about things."))
	for i := 0; i < 40; i++ {
		out = append(out, seg(10+i*10, 20+i*10,
			"This is at their son's birthday party. This is at their son's birthday party."))
	}
	out = append(out, seg(410, 420,
		"And then the speaker resumes with something quite different entirely."))
	return out
}

func TestDetectLoopsFindsRepetition(t *testing.T) {
	info := DetectLoops(loopingSegs(), DefaultLoopN, DefaultLoopMinCount, DefaultLoopMinDensity)

	if !info.IsLoop() {
		t.Fatalf("expected a loop, got density %.4f count %d", info.Density, info.Count)
	}
	if info.Count < 30 {
		t.Errorf("count = %d, expected at least 30", info.Count)
	}
	if info.Density < 0.30 {
		t.Errorf("density = %.4f, expected above 0.30", info.Density)
	}
	if len(info.Spans) == 0 {
		t.Fatal("expected at least one span")
	}

	s := info.Spans[0]
	// The span is quantised to 100-word windows, so it may begin before the
	// repetition does. Overshooting the start is the safe direction.
	if s.Start > 60 {
		t.Errorf("span start %.0fs is later than the repeated region began", s.Start)
	}
	if s.End < 250 {
		t.Errorf("span end %.0fs does not reach into the repeated region", s.End)
	}
}

// TestDetectLoopsLocalised covers the case the sliding window exists for: a
// short loop inside a long clean transcript, where the loop is too small a
// fraction of the whole to trip the overall density threshold.
func TestDetectLoopsLocalised(t *testing.T) {
	var segs []format.Segment
	n := 0

	// 1500 segments of 10 unique words each: 15,000 words with no repetition.
	for i := 0; i < 1500; i++ {
		words := ""
		for j := 0; j < 10; j++ {
			words += fmt.Sprintf("cleanword%d ", n)
			n++
		}
		segs = append(segs, seg(i*10, i*10+10, words))
	}

	// 25 segments of a twice-repeated phrase: 400 words.
	loopText := "This is at their son's birthday party. This is at their son's birthday party."
	for i := 0; i < 25; i++ {
		segs = append(segs, seg(15000+i*10, 15010+i*10, loopText))
	}

	info := DetectLoops(segs, DefaultLoopN, DefaultLoopMinCount, DefaultLoopMinDensity)

	// Overall density is under 5%, so this is not flagged globally.
	if info.Density >= DefaultLoopMinDensity {
		t.Fatalf("overall density %.4f unexpectedly above threshold", info.Density)
	}

	// But the windowed scan must still locate the region.
	if len(info.Spans) == 0 {
		t.Fatal("localised loop not detected by the window scan")
	}
	s := info.Spans[0]
	if s.Start > 15010 {
		t.Errorf("span start %.0fs is past the loop, expected at or before 15000", s.Start)
	}
	if s.End < 15200 {
		t.Errorf("span end %.0fs does not reach into the loop", s.End)
	}
	if s.Trigger != "unknown" && s.Trigger != "music" && s.Trigger != "laughter" {
		t.Errorf("unexpected trigger %q", s.Trigger)
	}
}

func TestDetectLoopsCleanTranscript(t *testing.T) {
	segs := []format.Segment{
		seg(0, 10, "First point about the subject of the lecture which is quite ordinary."),
		seg(10, 20, "Second point about a different aspect of the same subject matter."),
		seg(20, 30, "Third point which brings in some additional information entirely."),
		seg(30, 40, "Fourth point which closes the discussion in a considered fashion."),
	}
	info := DetectLoops(segs, DefaultLoopN, DefaultLoopMinCount, DefaultLoopMinDensity)

	if info.IsLoop() {
		t.Errorf("clean transcript flagged as a loop: density %.4f", info.Density)
	}
	if info.Count > 2 {
		t.Errorf("count = %d, expected no real repetition", info.Count)
	}
}

func TestDetectLoopsShortTranscript(t *testing.T) {
	info := DetectLoops([]format.Segment{seg(0, 5, "Too short to analyse.")},
		DefaultLoopN, DefaultLoopMinCount, DefaultLoopMinDensity)
	if info.IsLoop() {
		t.Error("short transcript flagged as a loop")
	}
}

func TestStripPunctKeepsWords(t *testing.T) {
	cases := map[string]string{
		"Hello,":        "hello",
		"world.":        "world",
		"♪":             "",
		"don't":         "don't",
		"(parenthesis)": "parenthesis",
	}
	for in, want := range cases {
		if got := stripPunct(in); got != want {
			t.Errorf("stripPunct(%q) = %q, want %q", in, got, want)
		}
	}
}
