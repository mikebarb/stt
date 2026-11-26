package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/mikebarb/stt/internal/whisper"
)

// correction is one rule from the corrections file.
type correction struct {
	from  string         // literal text, or the pattern for regex rules
	to    string         // replacement
	re    *regexp.Regexp // compiled matcher
	regex bool
	hits  int // applied count for the current file
}

// corrector holds the rule list and per-run hit counts.
type corrector struct {
	rules []correction
}

// loadCorrections reads the rules file. A missing file is not an error when
// required is false, which is the behaviour for the default path: the
// transcriber works without a corrections file.
func loadCorrections(path string, required bool) (*corrector, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) && !required {
			return &corrector{}, nil
		}
		return nil, fmt.Errorf("corrections: %w", err)
	}
	defer f.Close()

	c := &corrector{}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		idx := strings.Index(line, "=>")
		if idx < 0 {
			return nil, fmt.Errorf("%s:%d: missing '=>'", path, lineNo)
		}
		left := strings.TrimSpace(line[:idx])
		right := strings.TrimSpace(line[idx+2:])
		if left == "" {
			return nil, fmt.Errorf("%s:%d: empty pattern", path, lineNo)
		}

		rule := correction{from: left, to: right}

		if pat, ok := strings.CutPrefix(left, "regex:"); ok {
			pat = strings.TrimSpace(pat)
			re, err := regexp.Compile("(?i)" + pat)
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
			}
			rule.re = re
			rule.regex = true
			rule.from = pat
		} else {
			re, err := regexp.Compile(`(?i)\b` + regexp.QuoteMeta(left) + `\b`)
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
			}
			rule.re = re
		}

		c.rules = append(c.rules, rule)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("corrections: %w", err)
	}

	return c, nil
}

// Apply corrects one string. Rules run in file order, each exactly once, so
// later rules see earlier output; keep the list free of cascading pairs.
func (c *corrector) Apply(s string) string {
	if c == nil || s == "" || len(c.rules) == 0 {
		return s
	}

	for i := range c.rules {
		r := &c.rules[i]

		if r.regex {
			matches := r.re.FindAllString(s, -1)
			if len(matches) == 0 {
				continue
			}
			r.hits += len(matches)
			s = r.re.ReplaceAllString(s, r.to)
			continue
		}

		s = r.re.ReplaceAllStringFunc(s, func(m string) string {
			r.hits++
			return adaptCase(m, r.to)
		})
	}
	return s
}

// ApplyToSegments corrects every segment in place and returns the total
// number of replacements made since the last Reset.
func (c *corrector) ApplyToSegments(segs []whisper.Segment) int {
	if c == nil {
		return 0
	}
	for i := range segs {
		segs[i].Text = c.Apply(segs[i].Text)
	}
	total := 0
	for i := range c.rules {
		total += c.rules[i].hits
	}
	return total
}

// Reset clears per-file hit counts so a batch reports each file separately.
func (c *corrector) Reset() {
	if c == nil {
		return
	}
	for i := range c.rules {
		c.rules[i].hits = 0
	}
}

// Report writes the rules that fired, most frequent first. Silent when
// nothing matched, so clean transcripts produce no log.
func (c *corrector) Report(w io.Writer, modelUsed, sourceFile string) {
	if c == nil {
		return
	}

	type hit struct {
		from, to string
		n        int
	}
	var hits []hit
	total := 0
	for _, r := range c.rules {
		if r.hits > 0 {
			hits = append(hits, hit{r.from, r.to, r.hits})
			total += r.hits
		}
	}
	if len(hits) == 0 {
		return
	}

	sort.Slice(hits, func(i, j int) bool {
		if hits[i].n != hits[j].n {
			return hits[i].n > hits[j].n
		}
		return hits[i].from < hits[j].from
	})

	fmt.Fprintf(w, "# corrections %s  model %s  total %d\n",
		sourceFile, modelUsed, total)
	for _, h := range hits {
		fmt.Fprintf(w, "%s => %s  (%d)\n", h.from, h.to, h.n)
	}
}

// adaptCase copies the capitalisation of the match onto the replacement, so a
// rule authored in lower case still reads correctly at a sentence start.
func adaptCase(original, replacement string) string {
	if replacement == "" {
		return ""
	}

	// ALL CAPS in, ALL CAPS out.
	if isAllUpper(original) {
		return strings.ToUpper(replacement)
	}

	// Initial capital in, initial capital out.
	orig := []rune(original)
	rep := []rune(replacement)
	if len(orig) > 0 && unicode.IsUpper(orig[0]) {
		rep[0] = unicode.ToUpper(rep[0])
	}
	return string(rep)
}

func isAllUpper(s string) bool {
	hasLetter := false
	for _, r := range s {
		if unicode.IsLetter(r) {
			hasLetter = true
			if !unicode.IsUpper(r) {
				return false
			}
		}
	}
	return hasLetter
}

// writeCorrectionsLog writes a report of the rules that fired for this file.
// Silent (no file created) when nothing matched, so clean transcripts leave
// no clutter. A nil corrector is also silent.
func writeCorrectionsLog(path string, corr *corrector,
	modelUsed, sourceFile string) error {

	var b strings.Builder
	corr.Report(&b, modelUsed, filepath.Base(sourceFile))
	if b.Len() == 0 {
		return nil
	}
	return writeFileAtomic(path, []byte(b.String()))
}
