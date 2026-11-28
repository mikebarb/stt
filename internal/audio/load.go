// Package audio decodes audio files to float32 mono at 16 kHz.
package audio

/*
#cgo CXXFLAGS: -I${SRCDIR}/../../third_party/whisper.cpp/examples -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/whisper.cpp/build/examples -lcommon

#include <stddef.h>
#include <stdlib.h>

int stt_read_audio_mem(const char * data, size_t data_size, float ** out);
void stt_audio_free(float * samples);
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

// Load reads an audio file (using Go's native Windows UTF-16 file I/O) and
// decodes it (MP3, WAV, FLAC, OGG, ...) to float32 mono at 16 kHz via
// whisper.cpp's in-memory miniaudio decoder.
func Load(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("audio: failed to read %q: %w", path, err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("audio: empty file %q", path)
	}

	var p *C.float
	n := C.stt_read_audio_mem((*C.char)(unsafe.Pointer(&data[0])), C.size_t(len(data)), &p)
	if n < 0 || p == nil {
		return nil, fmt.Errorf("audio: failed to decode %q", path)
	}
	defer C.stt_audio_free(p)

	samples := unsafe.Slice((*float32)(unsafe.Pointer(p)), int(n))
	out := make([]float32, len(samples))
	copy(out, samples)
	return out, nil
}
