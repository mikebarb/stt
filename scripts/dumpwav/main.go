package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/mikebarb/stt/internal/audio"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: dumpwav <in.mp3> <out.wav>")
		os.Exit(2)
	}
	in, out := os.Args[1], os.Args[2]

	// Decode through the SAME pipeline the transcriber uses:
	// go-mp3 decode + mono downmix + ResampleLinear to 16 kHz.
	samples, err := audio.Load(in)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var peak float32
	var sumSq float64
	for _, s := range samples {
		a := float32(math.Abs(float64(s)))
		if a > peak {
			peak = a
		}
		sumSq += float64(s) * float64(s)
	}
	rms := math.Sqrt(sumSq / float64(len(samples)))

	fmt.Printf("samples   : %d\n", len(samples))
	fmt.Printf("duration  : %.1f s\n", float64(len(samples))/16000.0)
	fmt.Printf("peak      : %.4f (%.1f dBFS)\n", peak, 20*math.Log10(float64(peak)+1e-12))
	fmt.Printf("rms       : %.4f (%.1f dBFS)\n", rms, 20*math.Log10(rms+1e-12))

	// Write a 16-bit mono 16 kHz WAV.
	var data []byte
	for _, s := range samples {
		v := int16(math.Round(float64(s) * 32767))
		data = binary.LittleEndian.AppendUint16(data, uint16(v))
	}

	var hdr []byte
	hdr = append(hdr, "RIFF"...)
	hdr = binary.LittleEndian.AppendUint32(hdr, uint32(36+len(data)))
	hdr = append(hdr, "WAVEfmt "...)
	hdr = binary.LittleEndian.AppendUint32(hdr, 16)
	hdr = binary.LittleEndian.AppendUint16(hdr, 1) // PCM
	hdr = binary.LittleEndian.AppendUint16(hdr, 1) // mono
	hdr = binary.LittleEndian.AppendUint32(hdr, 16000)
	hdr = binary.LittleEndian.AppendUint32(hdr, 32000)
	hdr = binary.LittleEndian.AppendUint16(hdr, 2)
	hdr = binary.LittleEndian.AppendUint16(hdr, 16)
	hdr = append(hdr, "data"...)
	hdr = binary.LittleEndian.AppendUint32(hdr, uint32(len(data)))

	f, err := os.Create(out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	f.Write(hdr)
	f.Write(data)
	fmt.Printf("wrote %s\n", out)
}
