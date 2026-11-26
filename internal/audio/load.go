// Package audio decodes audio files to float32 mono at 16 kHz.
package audio

/*
#cgo CXXFLAGS: -I${SRCDIR}/../../third_party/whisper.cpp/examples -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/whisper.cpp/build/examples -lcommon

#include <stdlib.h>

int stt_read_audio(const char * path, float ** out);
void stt_audio_free(float * samples);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Load decodes an audio file (MP3, WAV, FLAC, OGG, ...) to float32
// mono at 16 kHz, using whisper.cpp's miniaudio-based decoder.
func Load(path string) ([]float32, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var p *C.float
	n := C.stt_read_audio(cPath, &p)
	if n < 0 || p == nil {
		return nil, fmt.Errorf("audio: failed to decode %q", path)
	}
	defer C.stt_audio_free(p)

	samples := unsafe.Slice((*float32)(unsafe.Pointer(p)), int(n))
	out := make([]float32, len(samples))
	copy(out, samples)
	return out, nil
}
