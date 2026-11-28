// internal\audio\stt_audio.cpp
//
// C bridge over whisper.cpp's in-memory read_audio_data (miniaudio decode path).
// Produces float32 mono PCM at 16 kHz (COMMON_SAMPLE_RATE).
//
// Decoding from memory avoids Windows ANSI/codepage filename failures when
// paths contain non-ASCII UTF-8 characters (Go reads the file via UTF-16).

#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstddef>

#include "common-whisper.h"

extern "C" {

// Decodes an audio file from an in-memory byte buffer to float32 mono at 16 kHz.
// Returns the sample count, or -1 on failure.
// The caller frees the returned buffer with stt_audio_free().
int stt_read_audio_mem(const char * data, size_t data_size, float ** out) {
    if (data == nullptr || data_size == 0 || out == nullptr) {
        return -1;
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;

    if (!::read_audio_data(data, data_size, pcmf32, pcmf32s, /*stereo=*/false)) {
        return -1;
    }

    const size_t n = pcmf32.size();
    if (n == 0) {
        return -1;
    }

    float * buf = (float *)malloc(n * sizeof(float));
    if (buf == nullptr) {
        return -1;
    }
    memcpy(buf, pcmf32.data(), n * sizeof(float));
    *out = buf;
    return (int)n;
}

void stt_audio_free(float * samples) {
    free(samples);
}

} // extern "C"
