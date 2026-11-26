package audiohash

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCalculateAudioHashIgnoresID3Tags(t *testing.T) {
	dir := t.TempDir()
	frames := make([]byte, 256)
	for i := range frames {
		frames[i] = byte(i)
	}

	// Bare audio frames
	barePath := filepath.Join(dir, "bare.mp3")
	if err := os.WriteFile(barePath, frames, 0o644); err != nil {
		t.Fatal(err)
	}

	// Same frames wrapped in ID3v2 header (20 bytes payload) + ID3v1 trailer (128 bytes)
	id3v2 := make([]byte, 10+20)
	copy(id3v2[:3], "ID3")
	id3v2[9] = 20 // synchsafe size = 20
	id3v1 := make([]byte, 128)
	copy(id3v1[:3], "TAG")

	tagged := append(append(id3v2, frames...), id3v1...)
	taggedPath := filepath.Join(dir, "tagged.mp3")
	if err := os.WriteFile(taggedPath, tagged, 0o644); err != nil {
		t.Fatal(err)
	}

	h1, err := CalculateAudioHash(barePath)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := CalculateAudioHash(taggedPath)
	if err != nil {
		t.Fatal(err)
	}
	if h1 != h2 {
		t.Fatalf("hash mismatch: bare=%s tagged=%s", h1, h2)
	}
}
