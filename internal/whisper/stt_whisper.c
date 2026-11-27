#include "stt_whisper.h"
#include <stdlib.h>
#include "whisper.h"

struct stt_ctx {
    struct whisper_context * wctx;
    int n_threads;
    int no_context;
    volatile int abort_requested;
    volatile int was_aborted;
};

static bool stt_abort_cb(void * user_data) {
    stt_ctx * h = (stt_ctx *) user_data;
    return (h != NULL && h->abort_requested != 0);
}

stt_ctx * stt_init(const char * model_path, int n_threads, int use_gpu, int no_context) {
    if (model_path == NULL) return NULL;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu != 0;

    struct whisper_context * wctx =
        whisper_init_from_file_with_params(model_path, cparams);
    if (wctx == NULL) return NULL;

    stt_ctx * h = (stt_ctx *) calloc(1, sizeof(*h));
    if (h == NULL) { whisper_free(wctx); return NULL; }

    h->wctx      = wctx;
    h->n_threads = (n_threads > 0) ? n_threads : 4;
    h->no_context = no_context;
    return h;
}

void stt_free(stt_ctx * h) {
    if (h == NULL) return;
    if (h->wctx != NULL) whisper_free(h->wctx);
    free(h);
}

int stt_transcribe(stt_ctx * h, const float * samples, int n_samples) {
    if (h == NULL || samples == NULL || n_samples <= 0) return -1;

    struct whisper_full_params p =
        whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

    p.strategy                   = WHISPER_SAMPLING_BEAM_SEARCH;
    p.n_threads                  = h->n_threads;
    p.print_realtime             = false;
    p.print_progress             = false;
    p.print_timestamps           = false;
    p.print_special              = false;
    p.translate                  = false;
    p.language                   = "en";
    p.single_segment             = false;
    p.beam_search.beam_size      = 5;
    p.token_timestamps           = true;    /* required for per-token p and t0/t1 */
    p.abort_callback             = stt_abort_cb;
    p.abort_callback_user_data   = h;
    p.no_context                 = true;    /* never bleed previous file into next in batch mode */
    if (h->no_context != 0) {
        p.n_max_text_ctx = 0;                /* disable rolling window-to-window context (-mc 0) */
    }
    
    int rc = whisper_full(h->wctx, p, samples, n_samples);
    
    if (rc != 0) {
        if (h->abort_requested != 0) {
            h->was_aborted = 1;
            return -2;              /* distinguished: aborted, not failed */
        }
        return rc;
    }
    return whisper_full_n_segments(h->wctx);
}

void stt_request_abort(stt_ctx * h) { if (h != NULL) h->abort_requested = 1; }
void stt_clear_abort(stt_ctx * h)   { if (h != NULL) { h->abort_requested = 0; h->was_aborted = 0; } }
int  stt_was_aborted(stt_ctx * h)   { return (h == NULL) ? 0 : h->was_aborted; }

int stt_n_segments(stt_ctx * h) {
    return (h == NULL) ? 0 : whisper_full_n_segments(h->wctx);
}

const char * stt_segment_text(stt_ctx * h, int i) {
    return (h == NULL) ? NULL : whisper_full_get_segment_text(h->wctx, i);
}

int64_t stt_segment_t0(stt_ctx * h, int i) {
    return (h == NULL) ? 0 : whisper_full_get_segment_t0(h->wctx, i);
}

int64_t stt_segment_t1(stt_ctx * h, int i) {
    return (h == NULL) ? 0 : whisper_full_get_segment_t1(h->wctx, i);
}

static void stt_log_ignore(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)text; (void)user_data;
}

void stt_silence_logs(void) {
    whisper_log_set(stt_log_ignore, NULL);
}

int stt_n_tokens(stt_ctx * h, int i_seg) {
    if (h == NULL || h->wctx == NULL || i_seg < 0) return 0;
    if (i_seg >= whisper_full_n_segments(h->wctx)) return 0;
    return whisper_full_n_tokens(h->wctx, i_seg);
}

const char * stt_token_text(stt_ctx * h, int i_seg, int i_tok) {
    if (h == NULL || h->wctx == NULL) return NULL;
    return whisper_full_get_token_text(h->wctx, i_seg, i_tok);
}

/* whisper_full_get_token_data returns a struct by value, which is awkward
 * across CGo; expose only the fields we need instead. */
float stt_token_prob(stt_ctx * h, int i_seg, int i_tok) {
    if (h == NULL || h->wctx == NULL) return 0.0f;
    struct whisper_token_data td =
        whisper_full_get_token_data(h->wctx, i_seg, i_tok);
    return td.p;
}

int64_t stt_token_t0(stt_ctx * h, int i_seg, int i_tok) {
    if (h == NULL || h->wctx == NULL) return 0;
    struct whisper_token_data td =
        whisper_full_get_token_data(h->wctx, i_seg, i_tok);
    return td.t0;
}

int64_t stt_token_t1(stt_ctx * h, int i_seg, int i_tok) {
    if (h == NULL || h->wctx == NULL) return 0;
    struct whisper_token_data td =
        whisper_full_get_token_data(h->wctx, i_seg, i_tok);
    return td.t1;
}
