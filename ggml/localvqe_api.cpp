/**
 * LocalVQE C API implementation.
 *
 * Streaming graph analysis/synthesis uses an orthonormal DCT-II basis
 * (baked into the graph as mul_mat nodes). This file keeps a 256-sample
 * PCM history ring per channel to construct the 512-sample analysis window
 * and performs 50%-overlap synthesis with divide-by-2 normalization.
 */

#include "localvqe_api.h"
#include "daf_frontend.h"
#include "localvqe_graph.h"
#include "noise_gate.h"
#include "native_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// ── Internal context ─────────────────────────────────────────────────────────

struct localvqe_ctx {
    // Optional DAF front-end (203K cascade): present when the gguf embeds
    // daf.* tensors. Runs per hop before the mask graph; the graph then
    // consumes (e, yhat) instead of (mic, ref).
    daf_frontend daf;
    std::vector<float> daf_e, daf_yhat;

    dvqe_graph_model graph_model;
    dvqe_stream_graph stream_graph;
    std::string last_error;

    // Hand-written single-thread streaming engine (native_engine.cpp), a
    // drop-in for process_frame_graph on the CPU backend. Selected at
    // construction when the GGUF is one it implements (v1.2-family mask
    // model: version>=3, kw==4, no DAF); the ggml graph is still built so
    // the two can be switched per context (options "engine" / env
    // LOCALVQE_ENGINE=graph|native). The engine keeps its own copy of the
    // weights; the raw localvqe_model is dropped after init.
    native_engine ne;
    bool use_native = false;

    // Front-end-only build (2.7K release): mask graph never built; each
    // hop emits the adaptive filter's output `e` directly.
    bool daf_standalone = false;

    // 256-sample PCM history per channel (kept for next analysis window).
    std::vector<float> pcm_hist_mic;
    std::vector<float> pcm_hist_ref;

    // Per-frame scratch: the 512-sample analysis window fed to the graph.
    std::vector<float> mic_window;
    std::vector<float> ref_window;
    std::vector<float> enh_window;  // 512-sample graph output before OLA

    // Synthesis OLA accumulator (512 floats, shifts by hop each frame).
    std::vector<float> ola;

    // Residual-echo noise gate. Off by default; user opts in via
    // localvqe_set_noise_gate. Operates per-hop on the post-OLA output.
    bool noise_gate_enabled = false;
    float noise_gate_threshold_dbfs = -45.0f;

    // s16 conversion scratch (3 * hop for frame API).
    std::vector<float> s16_conv_buf;

    // Batch API scratch (grow-only).
    std::vector<float> batch_s16_a;
    std::vector<float> batch_s16_b;
    std::vector<float> batch_s16_out;
};

static void ensure_size(std::vector<float>& v, size_t n) {
    if (v.size() < n) v.resize(n);
}

// ── C API ────────────────────────────────────────────────────────────────────

struct localvqe_options {
    std::string model_path;
    std::string backend_name = "CPU";
    int device_index = 0;
    int n_threads = 0;  // 0 = auto / honour GGML_NTHREADS env var
    std::string engine;  // "" = auto (env LOCALVQE_ENGINE, else native when supported)
};

// True when the native engine implements this model exactly (it mirrors the
// v11/SiLU streaming graph builders; see native_engine.h).
static inline bool hop_length_ok(const localvqe_hparams& hp) { return hp.hop_length == 256; }
static bool native_engine_supported(const localvqe_hparams& hp) {
    return hp.version >= 3 && hp.kernel_size_w == 4 && hp.kernel_size_h >= 2 &&
           hp.n_fft == 512 && hop_length_ok(hp) && hp.n_freq_bins == 256 &&
           !hp.daf_standalone && hp.mic_channels.size() == 6 &&
           hp.far_channels.size() == 3;
}


static localvqe_ctx_t make_ctx(const char* model_path,
                               const char* backend_name,
                               int device_index,
                               int n_threads_override = 0,
                               const char* engine_pref = "") {
    auto* ctx = new (std::nothrow) localvqe_ctx;
    if (!ctx) return 0;

    int n_threads = n_threads_override;
    if (n_threads == 0) {
        // Fallback chain: explicit option (above) → env var → auto.
        if (const char* env_threads = std::getenv("GGML_NTHREADS")) {
            n_threads = std::atoi(env_threads);
        }
    }

    if (!load_graph_model_ex(model_path, ctx->graph_model,
                             backend_name, device_index, true, n_threads)) {
        delete ctx;
        return 0;
    }
    ctx->daf_standalone = ctx->graph_model.hparams.daf_standalone;
    if (!ctx->daf_standalone &&
        !build_stream_graph(ctx->graph_model, ctx->stream_graph)) {
        free_graph_model(ctx->graph_model);
        delete ctx;
        return 0;
    }

    int n_fft = ctx->graph_model.hparams.n_fft;
    int hop = ctx->graph_model.hparams.hop_length;

    bool daf_ok = daf_init(ctx->daf, ctx->graph_model);
    if (ctx->daf_standalone) {
        if (!daf_ok) {  // a standalone build with no filter tensors is invalid
            fprintf(stderr, "localvqe: daf.standalone set but no daf.* tensors\n");
            free_graph_model(ctx->graph_model);
            delete ctx;
            return 0;
        }
        fprintf(stderr, "localvqe: front-end-only build (2.7K adaptive filter)\n");
    } else if (daf_ok) {
        fprintf(stderr, "localvqe: DAF front-end active (203K cascade)\n");
    }
    // Engine selection: explicit option > LOCALVQE_ENGINE env > auto.
    {
        std::string pref = engine_pref ? engine_pref : "";
        if (pref.empty()) {
            if (const char* e = std::getenv("LOCALVQE_ENGINE")) pref = e;
        }
        const bool want_native = (pref != "graph");
        const bool cpu_backend = (std::string(backend_name) == "CPU");
        if (want_native && cpu_backend && !ctx->daf.loaded &&
            native_engine_supported(ctx->graph_model.hparams)) {
            localvqe_model raw;
            if (load_model(model_path, raw, false) && ne_init(ctx->ne, raw)) {
                ctx->use_native = true;
            } else {
                fprintf(stderr, "localvqe: native engine %s, using the ggml graph\n",
                        pref == "native" ? "requested but init failed" : "init failed");
            }
        } else if (pref == "native") {
            fprintf(stderr, "localvqe: native engine requested but unsupported for this model/backend\n");
        }
        if (!ctx->daf_standalone)
            fprintf(stderr, "localvqe: engine=%s\n", ctx->use_native ? "native" : "graph");
    }

    ctx->daf_e.assign(hop, 0.0f);
    ctx->daf_yhat.assign(hop, 0.0f);
    ctx->pcm_hist_mic.assign(hop, 0.0f);
    ctx->pcm_hist_ref.assign(hop, 0.0f);
    ctx->mic_window.assign(n_fft, 0.0f);
    ctx->ref_window.assign(n_fft, 0.0f);
    ctx->enh_window.assign(n_fft, 0.0f);
    ctx->ola.assign(n_fft, 0.0f);
    ctx->s16_conv_buf.assign(3 * hop, 0.0f);

    return reinterpret_cast<localvqe_ctx_t>(ctx);
}

extern "C" {

LOCALVQE_API localvqe_ctx_t localvqe_new(const char* model_path) {
    return make_ctx(model_path, "CPU", 0);
}

LOCALVQE_API localvqe_options_t localvqe_options_new(void) {
    return reinterpret_cast<localvqe_options_t>(new (std::nothrow) localvqe_options);
}

LOCALVQE_API void localvqe_options_free(localvqe_options_t handle) {
    delete reinterpret_cast<localvqe_options*>(handle);
}

LOCALVQE_API int localvqe_options_set_model_path(localvqe_options_t handle,
                                                 const char* model_path) {
    if (!handle) return -1;
    if (!model_path || !*model_path) return -2;
    reinterpret_cast<localvqe_options*>(handle)->model_path = model_path;
    return 0;
}

LOCALVQE_API int localvqe_options_set_backend(localvqe_options_t handle,
                                              const char* backend_name) {
    if (!handle) return -1;
    if (!backend_name || !*backend_name) return -2;
    reinterpret_cast<localvqe_options*>(handle)->backend_name = backend_name;
    return 0;
}

LOCALVQE_API int localvqe_options_set_device(localvqe_options_t handle,
                                             int device_index) {
    if (!handle) return -1;
    if (device_index < 0) return -2;
    reinterpret_cast<localvqe_options*>(handle)->device_index = device_index;
    return 0;
}

LOCALVQE_API int localvqe_options_set_threads(localvqe_options_t handle,
                                              int n_threads) {
    if (!handle) return -1;
    if (n_threads < 0 || n_threads > 32) return -2;
    reinterpret_cast<localvqe_options*>(handle)->n_threads = n_threads;
    return 0;
}

LOCALVQE_API int localvqe_options_set_engine(localvqe_options_t handle,
                                             const char* engine) {
    if (!handle) return -1;
    if (!engine) return -2;
    std::string e = engine;
    if (e != "" && e != "auto" && e != "graph" && e != "native") return -2;
    reinterpret_cast<localvqe_options*>(handle)->engine = (e == "auto") ? "" : e;
    return 0;
}

LOCALVQE_API const char* localvqe_engine_name(localvqe_ctx_t handle) {
    if (!handle) return "";
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (ctx->daf_standalone) return "daf";
    return ctx->use_native ? "native" : "graph";
}

LOCALVQE_API localvqe_ctx_t localvqe_new_with_options(localvqe_options_t handle) {
    if (!handle) return 0;
    auto* opts = reinterpret_cast<localvqe_options*>(handle);
    if (opts->model_path.empty()) {
        fprintf(stderr, "localvqe: options missing model_path\n");
        return 0;
    }
    return make_ctx(opts->model_path.c_str(),
                    opts->backend_name.c_str(),
                    opts->device_index,
                    opts->n_threads,
                    opts->engine.c_str());
}

LOCALVQE_API void localvqe_list_devices(void) {
    dvqe_list_devices(stderr);
}

LOCALVQE_API void localvqe_print_profile(localvqe_ctx_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    print_memory_budget(ctx->graph_model, ctx->stream_graph);
    putchar('\n');
    print_op_histogram(ctx->stream_graph.graph);
    putchar('\n');
}

LOCALVQE_API void localvqe_free(localvqe_ctx_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    free_stream_graph(ctx->stream_graph);
    free_graph_model(ctx->graph_model);
    delete ctx;
}

// Push one hop of new PCM into the history ring, filling `window[0..n_fft)`
// with [prev_hop | new_hop]. Leaves history updated to new_hop for next frame.
static void build_window(std::vector<float>& hist, const float* new_hop,
                         int hop, float* window) {
    std::memcpy(window,            hist.data(), hop * sizeof(float));
    std::memcpy(window + hop,      new_hop,     hop * sizeof(float));
    std::memcpy(hist.data(),       new_hop,     hop * sizeof(float));
}

// Stream one hop through the graph, producing one hop of OLA'd output.
// Returns hop samples to `out`. The OLA accumulator is emitted on every
// call including frame 0 — synthesis-window-tapered samples at the left
// edge of the first frame's reconstruction start near zero and ramp up,
// so callers can pre-pend silence (or simply concatenate) without a
// click at the boundary. v1's DCT codec previously zeroed the first hop
// and got away with it because its OLA divisor halved the next hop; the
// COLA=1 sqrt-Hann² codec lands the next hop's first sample at full
// synthesis-window peak, so emitting zeros for the first hop and
// peak-amplitude content for the second creates an audible step.
static void stream_one_frame(localvqe_ctx* ctx, const float* mic,
                             const float* ref, float* out) {
    auto& hp = ctx->graph_model.hparams;
    int n_fft = hp.n_fft;
    int hop   = hp.hop_length;

    // Front-end-only build: emit the adaptive filter's output `e` directly,
    // no mask graph / codec round-trip (and so no one-hop codec latency).
    if (ctx->daf_standalone) {
        daf_process(ctx->daf, mic, ref, hop,
                    ctx->daf_e.data(), ctx->daf_yhat.data());
        std::memcpy(out, ctx->daf_e.data(), hop * sizeof(float));
        if (ctx->noise_gate_enabled) {
            localvqe::apply_noise_gate(out, hop,
                                       ctx->noise_gate_threshold_dbfs);
        }
        return;
    }

    if (ctx->daf.loaded) {
        daf_process(ctx->daf, mic, ref, hop,
                    ctx->daf_e.data(), ctx->daf_yhat.data());
        mic = ctx->daf_e.data();
        ref = ctx->daf_yhat.data();
    }
    build_window(ctx->pcm_hist_mic, mic, hop, ctx->mic_window.data());
    build_window(ctx->pcm_hist_ref, ref, hop, ctx->ref_window.data());

    if (ctx->use_native) {
        ne_process_frame(ctx->ne, ctx->mic_window.data(),
                         ctx->ref_window.data(), ctx->enh_window.data());
    } else {
        process_frame_graph(ctx->stream_graph, ctx->graph_model,
                            ctx->mic_window.data(), ctx->ref_window.data(),
                            ctx->enh_window.data());
    }

    // OLA scale below: v1's DCT codec has no analysis window so the two
    // overlapping frame contributions must be halved; v1.1's sqrt-Hann²
    // STFT-256 is COLA=1 at hop=N/2, no divisor.
    for (int i = 0; i < n_fft; i++) ctx->ola[i] += ctx->enh_window[i];

    const float scale = (hp.version >= 2) ? 1.0f : 0.5f;
    for (int i = 0; i < hop; i++) out[i] = ctx->ola[i] * scale;
    if (ctx->noise_gate_enabled) {
        localvqe::apply_noise_gate(out, hop,
                                   ctx->noise_gate_threshold_dbfs);
    }

    // Slide accumulator left by hop, zero-fill the tail.
    std::memmove(ctx->ola.data(), ctx->ola.data() + hop,
                 (n_fft - hop) * sizeof(float));
    std::memset(ctx->ola.data() + (n_fft - hop), 0, hop * sizeof(float));
}

LOCALVQE_API int localvqe_process_f32(localvqe_ctx_t handle,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    ctx->last_error.clear();

    auto& hp = ctx->graph_model.hparams;
    int n_fft = hp.n_fft;
    int hop   = hp.hop_length;

    if (n_samples < n_fft) {
        ctx->last_error = "Input too short: need at least "
                          + std::to_string(n_fft) + " samples";
        return -2;
    }

    if (!ctx->daf_standalone)
        reset_stream_graph(ctx->stream_graph, ctx->graph_model);
    if (ctx->use_native) ne_reset(ctx->ne);
    if (ctx->daf.loaded) daf_reset(ctx->daf);
    // Standalone front-end: the whole clip is available here (callers pass the
    // full signal), so prime the bulk delay once and lock it — the filter is
    // aligned from frame 0 instead of acquiring over the first ~1-3 s. The
    // cascade keeps online acquisition (its mask hides the transient).
    if (ctx->daf_standalone && ctx->daf.loaded)
        daf_prime_delay(ctx->daf, mic, ref, n_samples);
    std::fill(ctx->pcm_hist_mic.begin(), ctx->pcm_hist_mic.end(), 0.0f);
    std::fill(ctx->pcm_hist_ref.begin(), ctx->pcm_hist_ref.end(), 0.0f);
    std::fill(ctx->ola.begin(), ctx->ola.end(), 0.0f);

    int n_frames = n_samples / hop;  // drop the trailing partial hop
    for (int t = 0; t < n_frames; t++) {
        stream_one_frame(ctx,
                         mic + t * hop,
                         ref + t * hop,
                         out + t * hop);
    }
    int tail = n_samples - n_frames * hop;
    if (tail > 0) std::memset(out + n_frames * hop, 0, tail * sizeof(float));
    return 0;
}

// ── s16/f32 conversion helpers ────────────────────────────────────────────

static void s16_to_f32(const int16_t* in, float* out, int n) {
    const float scale = 1.0f / 32768.0f;
    for (int i = 0; i < n; i++) out[i] = in[i] * scale;
}

static void f32_to_s16(const float* in, int16_t* out, int n) {
    for (int i = 0; i < n; i++) {
        float v = std::max(-32768.0f, std::min(32767.0f, in[i] * 32768.0f));
        out[i] = (int16_t)v;
    }
}

LOCALVQE_API int localvqe_process_s16(localvqe_ctx_t handle,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);

    ensure_size(ctx->batch_s16_a, n_samples);
    ensure_size(ctx->batch_s16_b, n_samples);
    ensure_size(ctx->batch_s16_out, n_samples);

    s16_to_f32(mic, ctx->batch_s16_a.data(), n_samples);
    s16_to_f32(ref, ctx->batch_s16_b.data(), n_samples);

    int ret = localvqe_process_f32(handle, ctx->batch_s16_a.data(),
                                   ctx->batch_s16_b.data(), n_samples,
                                   ctx->batch_s16_out.data());
    if (ret != 0) return ret;

    f32_to_s16(ctx->batch_s16_out.data(), out, n_samples);
    return 0;
}

LOCALVQE_API const char* localvqe_last_error(localvqe_ctx_t handle) {
    if (!handle) return "null context";
    return reinterpret_cast<localvqe_ctx*>(handle)->last_error.c_str();
}

LOCALVQE_API int localvqe_sample_rate(localvqe_ctx_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.sample_rate;
}

LOCALVQE_API int localvqe_hop_length(localvqe_ctx_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.hop_length;
}

LOCALVQE_API int localvqe_fft_size(localvqe_ctx_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.n_fft;
}

// ── Streaming C API ──────────────────────────────────────────────────────

LOCALVQE_API int localvqe_process_frame_f32(localvqe_ctx_t handle,
                                           const float* mic, const float* ref,
                                           int hop_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (hop_samples != ctx->graph_model.hparams.hop_length) return -2;
    stream_one_frame(ctx, mic, ref, out);
    return 0;
}

LOCALVQE_API int localvqe_process_frame_s16(localvqe_ctx_t handle,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);

    float* mic_f = ctx->s16_conv_buf.data();
    float* ref_f = mic_f + hop_samples;
    float* out_f = ref_f + hop_samples;
    s16_to_f32(mic, mic_f, hop_samples);
    s16_to_f32(ref, ref_f, hop_samples);

    int ret = localvqe_process_frame_f32(handle, mic_f, ref_f, hop_samples, out_f);
    if (ret != 0) return ret;

    f32_to_s16(out_f, out, hop_samples);
    return 0;
}

LOCALVQE_API void localvqe_reset(localvqe_ctx_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (!ctx->daf_standalone)
        reset_stream_graph(ctx->stream_graph, ctx->graph_model);
    if (ctx->use_native) ne_reset(ctx->ne);
    if (ctx->daf.loaded) daf_reset(ctx->daf);
    std::fill(ctx->pcm_hist_mic.begin(), ctx->pcm_hist_mic.end(), 0.0f);
    std::fill(ctx->pcm_hist_ref.begin(), ctx->pcm_hist_ref.end(), 0.0f);
    std::fill(ctx->ola.begin(), ctx->ola.end(), 0.0f);
    // Note: gate config is intentionally NOT reset — it's caller-set
    // policy, not stream state.
}

LOCALVQE_API int localvqe_set_noise_gate(localvqe_ctx_t handle,
                                        int enabled,
                                        float threshold_dbfs) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    ctx->noise_gate_enabled = (enabled != 0);
    ctx->noise_gate_threshold_dbfs = threshold_dbfs;
    return 0;
}

LOCALVQE_API int localvqe_get_noise_gate(localvqe_ctx_t handle,
                                        int* enabled_out,
                                        float* threshold_dbfs_out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (enabled_out) *enabled_out = ctx->noise_gate_enabled ? 1 : 0;
    if (threshold_dbfs_out) *threshold_dbfs_out = ctx->noise_gate_threshold_dbfs;
    return 0;
}

} // extern "C"
