package pack

import (
	"strings"

	"github.com/mikebarb/stt/internal/format"
)

// Loop detection parameters. These mirror scripts/loops.py so the two agree.
const (
	// DefaultLoopN is the n-gram length used for density measurement.
	DefaultLoopN = 10

	// DefaultLoopMinCount is the number of repetitions at which a phrase is
	// considered a candidate loop.
	DefaultLoopMinCount = 15

	// DefaultLoopMinDensity is the fraction of the transcript consumed by
	// the repeated phrase above which the file is flagged as looping.
	//
	// The separation this threshold relies on is measured, not guessed:
	// genuine loops occupy 19-30% of the transcript, while legitimate
	// repetition - hymn refrains, scripture quotations, a congregation
	// invited to sing - occupies under 2%.
	DefaultLoopMinDensity = 0.05

	// windowWords and stepWords drive span detection.
	windowWords = 200
	stepWords   = 100

	// windowThreshold is the density within a window above which that window
	// is considered stuck.
	windowThreshold = 0.5
)

// LoopSpan is one contiguous region of repeated text.
type LoopSpan struct {
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Phrase      string  `json:"phrase"`
	Density     float64 `json:"density"`
	WordsPerMin float64 `json:"words_per_minute"`
	Trigger     string  `json:"trigger"`
	ContentLoss bool    `json:"suspected_content_loss"`
}

// LoopInfo records the repetition analysis for one transcript.
//
// Density is the fraction of the transcript consumed by the most-repeated
// n-gram. It is the discriminating measure: mean token confidence cannot
// detect repetition, because repeated text is emitted confidently.
type LoopInfo struct {
	N                 int        `json:"n"`
	Count             int        `json:"count"`
	Density           float64    `json:"density"`
	Flagged           bool       `json:"flagged"`
	Phrase            string     `json:"phrase"`
	FirstSeconds      float64    `json:"first_seconds"`
	LastSeconds       float64    `json:"last_seconds"`
	RepetitionSeconds float64    `json:"repetition_seconds"`
	Spans             []LoopSpan `json:"spans,omitempty"`
}

// IsLoop reports whether the transcript is looping rather than merely
// containing legitimate repetition.
func (l *LoopInfo) IsLoop() bool {
	return l != nil && l.Flagged
	//if l == nil {
	//	return false
	//}
	//return l.Count >= DefaultLoopMinCount && l.Density >= DefaultLoopMinDensity
}

// word is one token of the transcript with its start time in seconds.
type word struct {
	text  string
	start float64
}

// stripPunct removes characters that should not participate in n-gram
// matching: punctuation, and the music-note symbol whisper emits.
func stripPunct(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r + ('a' - 'A'))
		case r == '’' || r == '\'':
			b.WriteRune(r) // apostrophes are common inside words
		}
	}
	return b.String()
}

// collectWords flattens segments into words with their start times.
func collectWords(segs []format.Segment) []word {
	var out []word
	for _, s := range segs {
		start := float64(format.Cs(s.Start)) / 100.0
		for _, f := range strings.Fields(s.Text) {
			w := stripPunct(f)
			if w == "" {
				continue
			}
			out = append(out, word{text: w, start: start})
		}
	}
	return out
}

// DetectLoops analyses a transcript for repetition.
//
// It returns two things:
//
//   - The overall density of the most-repeated n-gram. This is always
//     populated, so a clean file still carries a comparable number.
//   - The spans over which repetition occurred, which is what a reviewer
//     needs in order to know where to listen.
func DetectLoops(segs []format.Segment, n, minCount int, minDensity float64) LoopInfo {
	if n <= 0 {
		n = DefaultLoopN
	}
	if minCount <= 0 {
		minCount = DefaultLoopMinCount
	}
	if minDensity <= 0 {
		minDensity = DefaultLoopMinDensity
	}

	words := collectWords(segs)
	info := LoopInfo{N: n}
	if len(words) < n {
		return info
	}

	// --- overall n-gram density ---
	counts := make(map[string]int, len(words))
	grams := make([]string, len(words)-n+1)
	for i := 0; i+n <= len(words); i++ {
		g := joinWords(words[i : i+n])
		grams[i] = g
		counts[g]++
	}

	bestGram, bestCount := "", 0
	for g, c := range counts {
		if c > bestCount || (c == bestCount && g < bestGram) {
			bestGram, bestCount = g, c
		}
	}

	info.Count = bestCount
	info.Phrase = bestGram
	info.Density = round3(float64(bestCount*n) / float64(len(words)))

	for i, g := range grams {
		if g == bestGram {
			info.FirstSeconds = round3(words[i].start)
			info.LastSeconds = round3(words[i+n-1].start)
			break
		}
	}

	// --- spans: sliding windows in which one n-gram dominates ---
	if len(words) >= windowWords {
		type marked struct {
			from, to int
			phrase   string
			density  float64
		}
		var marks []marked

		for w0 := 0; w0+windowWords <= len(words); w0 += stepWords {
			wc := make(map[string]int, windowWords)
			for i := w0; i+n <= w0+windowWords; i++ {
				wc[grams[i]]++
			}
			bg, bc := "", 0
			for g, c := range wc {
				if c > bc {
					bg, bc = g, c
				}
			}
			if bc == 0 {
				continue
			}
			d := float64(bc*n) / float64(windowWords)
			if d >= windowThreshold {
				marks = append(marks, marked{w0, w0 + windowWords, bg, round3(d)})
			}
		}

		// Merge overlapping or adjacent windows.
		var merged []marked
		for _, m := range marks {
			if len(merged) > 0 && m.from <= merged[len(merged)-1].to {
				last := &merged[len(merged)-1]
				if m.to > last.to {
					last.to = m.to
				}
				if m.density > last.density {
					last.phrase = m.phrase
					last.density = m.density
				}
				continue
			}
			merged = append(merged, m)
		}

		for _, m := range merged {
			sp := LoopSpan{
				Start:   round3(words[m.from].start),
				End:     round3(words[min(m.to, len(words)-1)].start),
				Phrase:  m.phrase,
				Density: m.density,
			}
			spanSec := sp.End - sp.Start
			if spanSec <= 0 {
				spanSec = 1
			}
			sp.WordsPerMin = round3(float64(m.to-m.from) / spanSec * 60)
			sp.Trigger = classifyTrigger(segs, sp.Start, sp.End)
			info.Spans = append(info.Spans, sp)
			info.RepetitionSeconds = round3(info.RepetitionSeconds + (sp.End - sp.Start))
		}
	}

	info.Flagged = info.Count >= minCount && info.Density >= minDensity

	return info
}

// classifyTrigger guesses what kind of non-speech caused a span, from the
// annotation tags whisper emits inside it.
func classifyTrigger(segs []format.Segment, start, end float64) string {
	sawMusic, sawLaughter := false, false
	for _, s := range segs {
		ss := float64(format.Cs(s.Start)) / 100.0
		se := float64(format.Cs(s.End)) / 100.0
		if se < start || ss > end {
			continue
		}
		low := strings.ToLower(s.Text)
		if strings.Contains(low, "♪") || strings.Contains(low, "[music]") ||
			strings.Contains(low, "(music)") {
			sawMusic = true
		}
		if strings.Contains(low, "[laughter]") || strings.Contains(low, "(laughter)") ||
			strings.Contains(low, "[laughing]") {
			sawLaughter = true
		}
	}
	switch {
	case sawMusic:
		return "music"
	case sawLaughter:
		return "laughter"
	default:
		return "unknown"
	}
}

// joinWords renders an n-gram as a readable phrase.
func joinWords(ws []word) string {
	parts := make([]string, len(ws))
	for i, w := range ws {
		parts[i] = w.text
	}
	return strings.Join(parts, " ")
}
