package main

import (
	"fmt"
	"io"
	"os"
	"time"

	mp3 "github.com/hajimehoshi/go-mp3"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: mp3info <file.mp3>")
		os.Exit(2)
	}

	for _, path := range os.Args[1:] {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}

		dec, err := mp3.NewDecoder(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, path, err)
			f.Close()
			continue
		}

		sr := dec.SampleRate()
		start := time.Now()
		n, err := io.Copy(io.Discard, dec)
		f.Close()

		fmt.Printf("%s\n  sample rate : %d Hz\n  decoded    : %d bytes in %s\n",
			path, sr, n, time.Since(start).Round(time.Millisecond))
	}
}
