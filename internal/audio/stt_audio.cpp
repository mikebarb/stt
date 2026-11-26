// internal\audio\stt_audio.cpp
//
// C bridge over whisper.cpp's read_audio_data (miniaudio decode path).
// Produces float32 mono PCM at 16 kHz (COMMON_SAMPLE_RATE).

#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

#include "common-whisper.h"

extern "C" {

// Decodes an audio file to float32 mono at 16 kHz.
// Returns the sample count, or -1 on failure.
// The caller frees the returned buffer with stt_audio_free().
int stt_read_audio(const char * path, float ** out) {
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;

    if (!::read_audio_data(std::string(path), pcmf32, pcmf32s, /*stereo=*/false)) {
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
