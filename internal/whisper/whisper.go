package whisper

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/whisper.cpp/include
#cgo CFLAGS: -I${SRCDIR}/../../third_party/whisper.cpp/ggml/include

#cgo LDFLAGS: -Wl,--start-group
#cgo LDFLAGS: ${SRCDIR}/../../third_party/whisper.cpp/build/src/libwhisper.a
#cgo LDFLAGS: ${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src/ggml.a
#cgo LDFLAGS: ${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src/ggml-base.a
#cgo LDFLAGS: ${SRCDIR}/../../third_party/whisper.cpp/build/ggml/src/ggml-cpu.a
#cgo LDFLAGS: -Wl,--end-group
#cgo LDFLAGS: -lstdc++ -lgomp

#include "stt_whisper.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

type Options struct {
	Threads int
	UseGPU  bool
	Quiet   bool // suppress whisper.cpp's native stderr logging
	Audit   bool // write low-confidence report (caller-level concern)
	Tokens  bool // capture token detail (needed for stats)
}

// Token is one decoded token within a segment, with the model's confidence
// in it and its timestamps (units of 10 ms).
type Token struct {
	Text  string
	Prob  float32
	Start int64
	End   int64
}

type Segment struct {
	Start  time.Duration
	End    time.Duration
	Text   string
	Tokens []Token
}

type Model struct {
	ptr   *C.stt_ctx
	audit bool
}

var ErrAborted = errors.New("whisper: transcription aborted")

func (m *Model) RequestAbort() {
	if m != nil && m.ptr != nil {
		C.stt_request_abort(m.ptr)
	}
}

func (m *Model) ClearAbort() {
	if m != nil && m.ptr != nil {
		C.stt_clear_abort(m.ptr)
	}
}

func New(modelPath string, o Options) (*Model, error) {
	if o.Quiet {
		C.stt_silence_logs()
	}

	if o.Threads <= 0 {
		o.Threads = runtime.NumCPU()
	}

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	gpu := C.int(0)
	if o.UseGPU {
		gpu = 1
	}

	ptr := C.stt_init(cPath, C.int(o.Threads), gpu)
	if ptr == nil {
		return nil, fmt.Errorf("whisper: failed to load model %q", modelPath)
	}

	m := &Model{ptr: ptr, audit: o.Audit}
	runtime.SetFinalizer(m, func(x *Model) { _ = x.Close() })
	return m, nil
}

func (m *Model) Close() error {
	if m == nil || m.ptr == nil {
		return nil
	}
	C.stt_free(m.ptr)
	m.ptr = nil
	runtime.SetFinalizer(m, nil)
	return nil
}

func (m *Model) Transcribe(samples []float32) ([]Segment, error) {
	if m == nil || m.ptr == nil {
		return nil, fmt.Errorf("whisper: model is closed")
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("whisper: no audio samples")
	}

	n := C.stt_transcribe(
		m.ptr,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.int(len(samples)),
	)
	if n == -2 {
		return nil, ErrAborted
	}
	if n < 0 {
		return nil, fmt.Errorf("whisper: transcription failed (code %d)", int(n))
	}

	out := make([]Segment, 0, int(C.stt_n_segments(m.ptr)))
	for i := 0; i < int(C.stt_n_segments(m.ptr)); i++ {
		seg := Segment{
			Start: time.Duration(int64(C.stt_segment_t0(m.ptr, C.int(i)))) * 10 * time.Millisecond,
			End:   time.Duration(int64(C.stt_segment_t1(m.ptr, C.int(i)))) * 10 * time.Millisecond,
			Text:  C.GoString(C.stt_segment_text(m.ptr, C.int(i))),
		}
		if m.audit {
			seg.Tokens = m.segmentTokens(C.int(i))
		}
		out = append(out, seg)
	}
	return out, nil
}

// segmentTokens extracts the token-level detail for one segment. Special
// tokens (framed in square brackets, e.g. "[_BEG_]") are skipped, and
// whitespace-only tokens are kept so text reassembly stays possible.
func (m *Model) segmentTokens(iSeg C.int) []Token {
	n := C.stt_n_tokens(m.ptr, iSeg)
	if n <= 0 {
		return nil
	}
	toks := make([]Token, 0, n)
	for i := C.int(0); i < n; i++ {
		txt := C.GoString(C.stt_token_text(m.ptr, iSeg, i))
		if len(txt) >= 2 && txt[0] == '[' && txt[len(txt)-1] == ']' {
			continue // special token
		}
		toks = append(toks, Token{
			Text:  txt,
			Prob:  float32(C.stt_token_prob(m.ptr, iSeg, i)),
			Start: int64(C.stt_token_t0(m.ptr, iSeg, i)),
			End:   int64(C.stt_token_t1(m.ptr, iSeg, i)),
		})
	}
	return toks
}
