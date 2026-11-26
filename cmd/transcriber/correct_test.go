package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeRules creates a temporary corrections file and returns its path.
func writeRules(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "corrections.txt")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCorrectionsApply(t *testing.T) {
	c, err := loadCorrections(writeRules(t, `
# comment line
sensorous => censorious
at lab => at L'Abri
regex: (contracts? and )lessons\b => ${1}lessens
`), true)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct{ in, want string }{
		// literal, lower case
		{"some sensorous elite", "some censorious elite"},
		// case adaptation
		{"Sensorous at the start", "Censorious at the start"},
		{"SENSOROUS shouts", "CENSORIOUS shouts"},
		// multi-word literal
		{"he works at lab now", "he works at L'Abri now"},
		// regex literal-adjacent
		{"contracts and lessons", "contracts and lessens"},
		{"the contract and lessons", "the contract and lessens"},
		// must NOT fire: whole-word matching
		{"lessons in grammar", "lessons in grammar"},
		{"sensorously inclined", "sensorously inclined"},
		// no rules match
		{"nothing to change here", "nothing to change here"},
	}
	for _, tc := range cases {
		if got := c.Apply(tc.in); got != tc.want {
			t.Errorf("Apply(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestCorrectionsHitCount(t *testing.T) {
	c, err := loadCorrections(writeRules(t, "sensorous => censorious\n"), true)
	if err != nil {
		t.Fatal(err)
	}

	c.Apply("sensorous one")
	c.Apply("sensorous two and sensorous three")

	if c.rules[0].hits != 3 {
		t.Errorf("hits = %d, want 3", c.rules[0].hits)
	}

	c.Reset()
	if c.rules[0].hits != 0 {
		t.Errorf("after Reset, hits = %d, want 0", c.rules[0].hits)
	}
}

func TestCorrectionsReport(t *testing.T) {
	c, err := loadCorrections(writeRules(t, "sensorous => censorious\ntermith => termeth\n"), true)
	if err != nil {
		t.Fatal(err)
	}

	// Nothing fired yet: Report must stay silent.
	var b strings.Builder
	c.Report(&b, "medium.en", "talk.mp3")
	if b.Len() != 0 {
		t.Errorf("empty report expected, got %q", b.String())
	}

	c.Apply("sensorous and termith")
	b.Reset()
	c.Report(&b, "medium.en", "talk.mp3")
	out := b.String()

	if !strings.Contains(out, "sensorous => censorious") {
		t.Errorf("report missing first rule: %q", out)
	}
	if !strings.Contains(out, "total 2") {
		t.Errorf("report missing total: %q", out)
	}
}

func TestCorrectionsMissingFile(t *testing.T) {
	// required=false: a missing file is not an error (default path case).
	c, err := loadCorrections(filepath.Join(t.TempDir(), "absent.txt"), false)
	if err != nil {
		t.Fatalf("optional missing file should not error: %v", err)
	}
	if len(c.rules) != 0 {
		t.Errorf("expected no rules, got %d", len(c.rules))
	}

	// required=true: a missing file is a hard error (explicit flag case).
	if _, err := loadCorrections(filepath.Join(t.TempDir(), "absent.txt"), true); err == nil {
		t.Error("required missing file should error")
	}
}

func TestCorrectionsMalformed(t *testing.T) {
	cases := []string{
		"no arrow here\n",
		"=> missing pattern\n",
		"regex: (unclosed => x\n",
	}
	for _, body := range cases {
		if _, err := loadCorrections(writeRules(t, body), true); err == nil {
			t.Errorf("expected error for %q", body)
		}
	}
}

func TestCorrectionsNilSafe(t *testing.T) {
	var c *corrector
	if got := c.Apply("anything"); got != "anything" {
		t.Errorf("nil corrector changed text: %q", got)
	}

	var b strings.Builder
	c.Report(&b, "m", "f")
	if b.Len() != 0 {
		t.Errorf("nil corrector wrote a report")
	}
}
