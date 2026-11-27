#ifndef STT_WHISPER_H
#define STT_WHISPER_H

#include <stdint.h>

typedef struct stt_ctx stt_ctx;

/* Returns NULL on failure. n_threads <= 0 selects a default. */
stt_ctx * stt_init(const char * model_path, int n_threads, int use_gpu, int no_context);
void      stt_free(stt_ctx * h);

/* Returns segment count, or negative on error. */
int stt_transcribe(stt_ctx * h, const float * samples, int n_samples);

int         stt_n_segments(stt_ctx * h);
const char * stt_segment_text(stt_ctx * h, int i);
int64_t     stt_segment_t0(stt_ctx * h, int i);   /* units of 10 ms */
int64_t     stt_segment_t1(stt_ctx * h, int i);

/* Token-level access. Tokens are valid after stt_transcribe; i_seg and
 * i_tok are zero-based. Timestamps are in units of 10 ms and are only
 * meaningful because stt_transcribe enables token_timestamps. */
int      stt_n_tokens(stt_ctx * h, int i_seg);
const char * stt_token_text(stt_ctx * h, int i_seg, int i_tok);
float    stt_token_prob(stt_ctx * h, int i_seg, int i_tok);
int64_t  stt_token_t0(stt_ctx * h, int i_seg, int i_tok);
int64_t  stt_token_t1(stt_ctx * h, int i_seg, int i_tok);

/* Silences whisper.cpp's diagnostic logging (model load info, etc).
   Call once before stt_init if you want clean output. */
void      stt_silence_logs(void);

int      stt_n_tokens(stt_ctx * h, int i_seg);
const char * stt_token_text(stt_ctx * h, int i_seg, int i_tok);
float    stt_token_prob(stt_ctx * h, int i_seg, int i_tok);
int64_t  stt_token_t0(stt_ctx * h, int i_seg, int i_tok);
int64_t  stt_token_t1(stt_ctx * h, int i_seg, int i_tok);

/* Cooperative abort. stt_request_abort may be called from another thread
 * (e.g. a signal handler); whisper polls the flag between compute batches.
 * After an aborted run, call stt_clear_abort before transcribing again. */
void stt_request_abort(stt_ctx * h);
void stt_clear_abort(stt_ctx * h);
int  stt_was_aborted(stt_ctx * h);

#endif
