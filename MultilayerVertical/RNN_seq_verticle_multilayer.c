#include "RNN_seq_verticle_multilayer.h"
#include <stdint.h>

static real rand_weight(void) {
    return ((real)rand() / (real)RAND_MAX) * 0.2f - 0.1f;
}

static real clip_grad(real x, real grad_clip) {
    if (x > grad_clip) return grad_clip;
    if (x < -grad_clip) return -grad_clip;
    return x;
}

void init_default_config(RNNConfig *cfg) {
    cfg->input_size = 0;
    cfg->hidden_size = 64;
    cfg->output_size = 1;
    cfg->num_layers = 1;
    cfg->seq_len = 8;
    cfg->batch_size = 16;
    cfg->learning_rate = 0.001f;
    cfg->grad_clip = 5.0f;
    cfg->epsilon = 1e-7f;
    cfg->num_epochs = 10;
    cfg->shuffle_seed = 42u;
}

void init_default_schema(DatasetSchema *schema) {
    memset(schema, 0, sizeof(DatasetSchema));
    schema->has_header = 1;
    schema->delimiter = ',';
    schema->label_col = -1;
    schema->allow_text_labels = 1;
    schema->infer_label_col = 1;
}

void free_schema(DatasetSchema *schema) {
    free(schema->feature_cols);
    free(schema->drop_cols);
    schema->feature_cols = NULL;
    schema->drop_cols = NULL;
    schema->feature_count = 0;
    schema->drop_count = 0;
}

int allocate_rnn(SequentialRNN *rnn, const RNNConfig *cfg) {
    rnn->Wxh = (real *)calloc(rnn_wxh_total(cfg), sizeof(real));
    rnn->Whh = (real *)calloc(rnn_whh_total(cfg), sizeof(real));
    rnn->Why = (real *)calloc(cfg->output_size * cfg->hidden_size, sizeof(real));
    rnn->bh = (real *)calloc(rnn_bh_total(cfg), sizeof(real));
    rnn->by = (real *)calloc(cfg->output_size, sizeof(real));

    return rnn->Wxh && rnn->Whh && rnn->Why && rnn->bh && rnn->by;
}

void free_rnn(SequentialRNN *rnn) {
    free(rnn->Wxh);
    free(rnn->Whh);
    free(rnn->Why);
    free(rnn->bh);
    free(rnn->by);
    rnn->Wxh = rnn->Whh = rnn->Why = rnn->bh = rnn->by = NULL;
}

/* Bounds are layer totals, not L=1 sizes. hidden_size*input_size would merge
 * layer 0 only and drop every gradient above it. */
int allocate_gradients(RNNGradients *grads, const RNNConfig *cfg) {
    grads->dWxh = (real *)calloc(rnn_wxh_total(cfg), sizeof(real));
    grads->dWhh = (real *)calloc(rnn_whh_total(cfg), sizeof(real));
    grads->dWhy = (real *)calloc(cfg->output_size * cfg->hidden_size, sizeof(real));
    grads->dbh = (real *)calloc(rnn_bh_total(cfg), sizeof(real));
    grads->dby = (real *)calloc(cfg->output_size, sizeof(real));

    return grads->dWxh && grads->dWhh && grads->dWhy && grads->dbh && grads->dby;
}

void free_gradients(RNNGradients *grads) {
    free(grads->dWxh);
    free(grads->dWhh);
    free(grads->dWhy);
    free(grads->dbh);
    free(grads->dby);
    grads->dWxh = grads->dWhh = grads->dWhy = grads->dbh = grads->dby = NULL;
}

int allocate_stats(NormalizationStats *stats, const RNNConfig *cfg) {
    stats->mean = (real *)calloc(cfg->input_size, sizeof(real));
    stats->std = (real *)calloc(cfg->input_size, sizeof(real));
    return stats->mean && stats->std;
}

void free_stats(NormalizationStats *stats) {
    free(stats->mean);
    free(stats->std);
    stats->mean = stats->std = NULL;
}

int allocate_vertical_scratch(VerticalScratch *scratch, const RNNConfig *cfg) {
    size_t cap = cfg->batch_size;
    scratch->capacity = cap;
    scratch->H_trace = (real *)calloc(cap * rnn_h_total(cfg), sizeof(real));
    scratch->dh_carry = (real *)calloc(cap * rnn_bh_total(cfg), sizeof(real));
    scratch->logit = (real *)calloc(cap, sizeof(real));
    scratch->score = (real *)calloc(cap, sizeof(real));
    scratch->loss = (double *)calloc(cap, sizeof(double));
    scratch->pred_label = (int *)calloc(cap, sizeof(int));
    return scratch->H_trace && scratch->dh_carry && scratch->logit &&
           scratch->score && scratch->loss && scratch->pred_label;
}

void free_vertical_scratch(VerticalScratch *scratch) {
    free(scratch->H_trace);
    free(scratch->dh_carry);
    free(scratch->logit);
    free(scratch->score);
    free(scratch->loss);
    free(scratch->pred_label);
    scratch->H_trace = scratch->dh_carry = scratch->logit = scratch->score = NULL;
    scratch->loss = NULL;
    scratch->pred_label = NULL;
    scratch->capacity = 0;
}

void compute_stage_ranges(StageRange *ranges, int num_threads, const RNNConfig *cfg, int *num_stages_out) {
    int seq_len = (int)cfg->seq_len;
    int num_stages = num_threads < seq_len ? num_threads : seq_len;
    if (num_stages < 1) num_stages = 1;
    // Contiguous, as-even-as-possible split of timesteps across stages: first
    // (seq_len % num_stages) stages get one extra timestep.
    int base = seq_len / num_stages;
    int rem = seq_len % num_stages;
    size_t lo = 0;
    for (int s = 0; s < num_stages; s++) {
        size_t span = (size_t)base + (s < rem ? 1u : 0u);
        ranges[s].ts_lo = lo;
        ranges[s].ts_hi = lo + span;
        lo += span;
    }
    *num_stages_out = num_stages;
}

// dst/src loop bounds are layer totals, see allocate_gradients comment.
void rnn_add_gradients(RNNGradients *dst, const RNNGradients *src, const RNNConfig *cfg) {
    for (size_t i = 0; i < rnn_wxh_total(cfg); i++) dst->dWxh[i] += src->dWxh[i];
    for (size_t i = 0; i < rnn_whh_total(cfg); i++) dst->dWhh[i] += src->dWhh[i];
    for (size_t i = 0; i < cfg->output_size * cfg->hidden_size; i++) dst->dWhy[i] += src->dWhy[i];
    for (size_t i = 0; i < rnn_bh_total(cfg); i++) dst->dbh[i] += src->dbh[i];
    for (size_t i = 0; i < cfg->output_size; i++) dst->dby[i] += src->dby[i];
}

void rnn_init(SequentialRNN *rnn, const RNNConfig *cfg, unsigned int seed) {
    srand(seed);

    for (size_t i = 0; i < rnn_wxh_total(cfg); i++) rnn->Wxh[i] = rand_weight();
    for (size_t i = 0; i < rnn_whh_total(cfg); i++) rnn->Whh[i] = rand_weight();
    for (size_t i = 0; i < cfg->output_size * cfg->hidden_size; i++) rnn->Why[i] = rand_weight();
    for (size_t i = 0; i < rnn_bh_total(cfg); i++) rnn->bh[i] = 0.0f;
    for (size_t i = 0; i < cfg->output_size; i++) rnn->by[i] = 0.0f;
}

real rnn_sigmoid(real x) {
    if (x >= 0.0f) {
        real z = RNN_EXP(-x);
        return 1.0f / (1.0f + z);
    }
    real z = RNN_EXP(x);
    return z / (1.0f + z);
}

void rnn_zero_gradients(RNNGradients *grads, const RNNConfig *cfg) {
    memset(grads->dWxh, 0, rnn_wxh_total(cfg) * sizeof(real));
    memset(grads->dWhh, 0, rnn_whh_total(cfg) * sizeof(real));
    memset(grads->dWhy, 0, cfg->output_size * cfg->hidden_size * sizeof(real));
    memset(grads->dbh, 0, rnn_bh_total(cfg) * sizeof(real));
    memset(grads->dby, 0, cfg->output_size * sizeof(real));
}

/*
 * One stage's forward work for one sequence, multi-layer: advance all L
 * layers through stage's owned timesteps, writing into that sequence's slice of
 * shared H_trace.
 *
 * Stage runs whole slice serially on one thread; parallelism comes from
 * different stages working different sequences at same tick.
 *
 * Cross-layer handoff within timestep is local to this call: layer l's h_new
 * becomes layer l+1's input at same timestep, stage, thread and tick, never
 * crosses stage boundary. Only recurrent input at lower boundary (one per
 * layer) came from stage below last tick; per-tick barrier in driver makes it
 * visible before this read.
 */
static void stage_forward(SequentialRNN *rnn, const real *x, real *h_seq,
                          size_t ts_lo, size_t ts_hi, const RNNConfig *cfg) {
    size_t H = cfg->hidden_size, I = cfg->input_size, L = cfg->num_layers;
    for (size_t t = ts_lo; t < ts_hi; t++) {
        const real *in = &x[t * I];
        for (size_t l = 0; l < L; l++) {
            size_t in_size = (l == 0) ? I : H;
            const real *Wxh_l = rnn->Wxh + rnn_wxh_off(cfg, l);
            const real *Whh_l = rnn->Whh + rnn_whh_off(cfg, l);
            const real *bh_l = rnn->bh + rnn_bh_off(cfg, l);
            const real *h_prev = h_seq + rnn_h_off(cfg, l, t);
            real *h_new = h_seq + rnn_h_off(cfg, l, t + 1);
            for (size_t j = 0; j < H; j++) {
                real z = bh_l[j];
                for (size_t k = 0; k < in_size; k++) z += Wxh_l[j * in_size + k] * in[k];
                for (size_t k = 0; k < H; k++) z += Whh_l[j * H + k] * h_prev[k];
                h_new[j] = RNN_TANH(z);
            }
            in = h_new;  // feeds layer above at same timestep
        }
    }
}

/*
 * One stage's backward (BPTT) work for one sequence, multi-layer: backprop
 * owned timesteps in reverse, layers top-down inside each timestep (as serial
 * multi-layer program), accumulating into stage's own buffer `g`, no
 * cross-stage race.
 *
 * dh_next[l] gradient into layer l from timestep t+1 (recurrent). At top of
 *            stage's range, seeded from Why (top layer, output stage only) or
 *            handed down from stage above via dh_in, one slot per layer.
 * dh_up[l]   gradient into layer l from layer l+1, same timestep (vertical).
 *            Zeroed per timestep, local to stage/thread, never crosses stage
 *            boundary.
 * dh_prev[l] gradient layer hands back to timestep t-1.
 *
 * After reverse loop, dh_next holds per-layer dh at lower boundary, handed
 * down via dh_out. Stage 0 skips it: lower boundary is zero init state.
 */
static void stage_backward(SequentialRNN *rnn, const SequenceSample *sample, real *h_seq,
                           size_t ts_lo, size_t ts_hi, const RNNConfig *cfg,
                           real dy_seed, int is_output_stage,
                           const real *dh_in, real *dh_out, RNNGradients *g) {
    size_t H = cfg->hidden_size, L = cfg->num_layers;
    real dh_next[cfg->num_layers * cfg->hidden_size];
    real dh_up[cfg->num_layers * cfg->hidden_size];
    real dh_prev[cfg->num_layers * cfg->hidden_size];

    if (is_output_stage) {
        memset(dh_next, 0, L * H * sizeof(real));
        g->dby[0] += dy_seed;
        const real *h_final = h_seq + rnn_h_off(cfg, L - 1, cfg->seq_len);
        for (size_t k = 0; k < H; k++) {
            g->dWhy[k] += dy_seed * h_final[k];
            dh_next[(L - 1) * H + k] = rnn->Why[k] * dy_seed;
        }
    } else {
        memcpy(dh_next, dh_in, L * H * sizeof(real));
    }

    for (size_t ti = ts_hi; ti-- > ts_lo; ) {
        memset(dh_prev, 0, L * H * sizeof(real));
        memset(dh_up, 0, L * H * sizeof(real));

        for (size_t li = L; li-- > 0; ) {
            const size_t in_size = (li == 0) ? cfg->input_size : H;
            const real *in = (li == 0) ? &sample->x[ti * cfg->input_size]
                                       : h_seq + rnn_h_off(cfg, li - 1, ti + 1);
            const real *Wxh_l = rnn->Wxh + rnn_wxh_off(cfg, li);
            const real *Whh_l = rnn->Whh + rnn_whh_off(cfg, li);
            real *dWxh_l = g->dWxh + rnn_wxh_off(cfg, li);
            real *dWhh_l = g->dWhh + rnn_whh_off(cfg, li);
            real *dbh_l = g->dbh + rnn_bh_off(cfg, li);
            const real *h_cur = h_seq + rnn_h_off(cfg, li, ti + 1);
            const real *h_old = h_seq + rnn_h_off(cfg, li, ti);
            // Where layer deposits vertical gradient for layer below. NULL at layer 0.
            real *dh_down = (li > 0) ? &dh_up[(li - 1) * H] : NULL;

            real dh[cfg->hidden_size];
            for (size_t k = 0; k < H; k++) {
                dh[k] = dh_next[li * H + k];
                if (li + 1 < L) dh[k] += dh_up[li * H + k];
            }

            for (size_t k = 0; k < H; k++) {
                real h_tp1 = h_cur[k];
                real dz = dh[k] * (1.0f - h_tp1 * h_tp1);
                dbh_l[k] += dz;

                if (dh_down) {
                    for (size_t i = 0; i < in_size; i++) {
                        dWxh_l[k * in_size + i] += dz * in[i];
                        dh_down[i] += Wxh_l[k * in_size + i] * dz;
                    }
                } else {
                    for (size_t i = 0; i < in_size; i++) {
                        dWxh_l[k * in_size + i] += dz * in[i];
                    }
                }

                for (size_t j = 0; j < H; j++) {
                    dWhh_l[k * H + j] += dz * h_old[j];
                    dh_prev[li * H + j] += Whh_l[k * H + j] * dz;
                }
            }
        }

        memcpy(dh_next, dh_prev, L * H * sizeof(real));
    }

    // dh_next holds per-layer dh at ts_lo boundary; hand it down. Stage 0's
    // lower boundary is zero init state, never read.
    if (ts_lo > 0) {
        memcpy(dh_out, dh_next, L * H * sizeof(real));
    }
}

/*
 * Forward wavefront over batch. At tick `tick`, stage `tid` works sequence
 * idx = tick - tid (diagonal), so stage 0 leads, each later stage trails by
 * one tick: stage 0 starts next sequence while stage 1 works current one.
 *
 * One barrier per tick publishes each stage's per-layer boundary hidden state
 * before next tick reads it. Runs batch_count + num_stages - 1 ticks: fill,
 * steady state, drain.
 */
void rnn_pipeline_forward_batch(SequentialRNN *rnn, const SequenceSample *batch, int batch_count,
                                const RNNConfig *cfg, real pos_weight, int weighted_loss,
                                const StageRange *ranges, int num_stages,
                                int tid, VerticalScratch *scratch) {
    size_t H = cfg->hidden_size, L = cfg->num_layers;
    size_t trace_stride = rnn_h_total(cfg);
    int total_ticks = batch_count + num_stages - 1;
    int is_stage = (tid < num_stages);
    size_t ts_lo = is_stage ? ranges[tid].ts_lo : 0;
    size_t ts_hi = is_stage ? ranges[tid].ts_hi : 0;
    int output_stage = num_stages - 1;

    for (int tick = 0; tick < total_ticks; tick++) {
        if (is_stage) {
            int idx = tick - tid;
            if (idx >= 0 && idx < batch_count) {
                real *h_seq = &scratch->H_trace[(size_t)idx * trace_stride];
                /* Zero every layer's t=0 boundary, start of recurrence for all L layers.
                 * Full-trace memset is simplest way to hit all L non-contiguous t=0 slots. */
                if (tid == 0) memset(h_seq, 0, trace_stride * sizeof(real));
                stage_forward(rnn, batch[idx].x, h_seq, ts_lo, ts_hi, cfg);
                if (tid == output_stage) {
                    const real *h_final = h_seq + rnn_h_off(cfg, L - 1, cfg->seq_len);
                    real acc = 0.0f;
                    for (size_t k = 0; k < H; k++) acc += rnn->Why[k] * h_final[k];
                    real logit = rnn->by[0] + acc;
                    real sc = rnn_sigmoid(logit);
                    real target = (real)batch[idx].label;
                    real clipped = RNN_MIN(RNN_MAX(sc, cfg->epsilon), 1.0f - cfg->epsilon);
                    real w = weighted_loss ? ((batch[idx].label == 1) ? pos_weight : 1.0f) : 1.0f;
                    scratch->logit[idx] = logit;
                    scratch->score[idx] = sc;
                    scratch->loss[idx] = -(double)w * (target * RNN_LOG(clipped) + (1.0f - target) * RNN_LOG(1.0f - clipped));
                    scratch->pred_label[idx] = (sc >= 0.5f) ? 1 : 0;
                }
            }
        }
        #pragma omp barrier
    }
}

/*
 * Backward (BPTT) wavefront over same batch, reversed: at tick `btick`, stage
 * `tid` works idx = btick - (output_stage - tid), so output stage leads (owns
 * final timestep, seeds recurrence) and stage 0 trails.
 *
 * Each stage backprops own timesteps, all layers, into own stage_grads buffer.
 * Barrier publishes per-layer handed-down dh at each stage boundary before
 * lower stage reads it next tick. Call after rnn_pipeline_forward_batch on
 * same batch.
 */
void rnn_pipeline_backward_batch(SequentialRNN *rnn, const SequenceSample *batch, int batch_count,
                                 const RNNConfig *cfg, real pos_weight,
                                 const StageRange *ranges, int num_stages,
                                 int tid, VerticalScratch *scratch, RNNGradients *stage_grads_tid) {
    size_t trace_stride = rnn_h_total(cfg);
    size_t carry_stride = rnn_bh_total(cfg);
    int total_ticks = batch_count + num_stages - 1;
    int is_stage = (tid < num_stages);
    size_t ts_lo = is_stage ? ranges[tid].ts_lo : 0;
    size_t ts_hi = is_stage ? ranges[tid].ts_hi : 0;
    int output_stage = num_stages - 1;

    for (int btick = 0; btick < total_ticks; btick++) {
        if (is_stage) {
            int idx = btick - (output_stage - tid);
            if (idx >= 0 && idx < batch_count) {
                real *h_seq = &scratch->H_trace[(size_t)idx * trace_stride];
                real *dh_slot = &scratch->dh_carry[(size_t)idx * carry_stride];
                int is_out = (tid == output_stage);
                real dy_seed = 0.0f;
                if (is_out) {
                    real sc = scratch->score[idx];
                    real target = (real)batch[idx].label;
                    real cw = (batch[idx].label == 1) ? pos_weight : 1.0f;
                    dy_seed = cw * (sc - target);
                }
                stage_backward(rnn, &batch[idx], h_seq, ts_lo, ts_hi, cfg,
                               dy_seed, is_out, dh_slot, dh_slot, stage_grads_tid);
            }
        }
        #pragma omp barrier
    }
}

void rnn_apply_gradients(SequentialRNN *rnn, const RNNGradients *grads, const RNNConfig *cfg, int batch_count) {
    real scale = 1.0f / (real)batch_count;

    for (size_t i = 0; i < rnn_wxh_total(cfg); i++) {
        rnn->Wxh[i] -= cfg->learning_rate * clip_grad(grads->dWxh[i] * scale, cfg->grad_clip);
    }
    for (size_t i = 0; i < rnn_whh_total(cfg); i++) {
        rnn->Whh[i] -= cfg->learning_rate * clip_grad(grads->dWhh[i] * scale, cfg->grad_clip);
    }
    for (size_t i = 0; i < cfg->output_size * cfg->hidden_size; i++) {
        rnn->Why[i] -= cfg->learning_rate * clip_grad(grads->dWhy[i] * scale, cfg->grad_clip);
    }
    for (size_t i = 0; i < rnn_bh_total(cfg); i++) {
        rnn->bh[i] -= cfg->learning_rate * clip_grad(grads->dbh[i] * scale, cfg->grad_clip);
    }
    for (size_t i = 0; i < cfg->output_size; i++) {
        rnn->by[i] -= cfg->learning_rate * clip_grad(grads->dby[i] * scale, cfg->grad_clip);
    }
}

void compute_normalization_stats(const SequenceSample *data, size_t n_samples, const RNNConfig *cfg, NormalizationStats *stats) {
    const double total_steps = (double)n_samples * (double)cfg->seq_len;

    for (size_t d = 0; d < cfg->input_size; d++) {
        double sum = 0.0;
        double sq_sum = 0.0;

        for (size_t i = 0; i < n_samples; i++) {
            for (size_t t = 0; t < cfg->seq_len; t++) {
                double v = data[i].x[t * cfg->input_size + d];
                sum += v;
                sq_sum += v * v;
            }
        }

        double mean = sum / total_steps;
        double var = (sq_sum / total_steps) - (mean * mean);
        if (var < 1e-12) var = 1e-12;

        stats->mean[d] = (real)mean;
        stats->std[d] = (real)sqrt(var);
    }
}

void apply_normalization(SequenceSample *data, size_t n_samples, const RNNConfig *cfg, const NormalizationStats *stats) {
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t t = 0; t < cfg->seq_len; t++) {
            for (size_t d = 0; d < cfg->input_size; d++) {
                real *v = &data[i].x[t * cfg->input_size + d];
                *v = (*v - stats->mean[d]) / (stats->std[d] + cfg->epsilon);
            }
        }
    }
}

/*
 * NumPy-compatible MT19937 for seeded sequence-level balancing and
 * training-sequence shuffle. Same seed gives same sequence as NumPy.
 */
typedef struct {
    uint32_t mt[624];
    int index;
} NumpyMT19937;

static void np_mt_seed(NumpyMT19937 *rng, uint32_t seed) {
    rng->mt[0] = seed;
    for (int i = 1; i < 624; i++) rng->mt[i] = 1812433253u * (rng->mt[i - 1] ^ (rng->mt[i - 1] >> 30)) + (uint32_t)i;
    rng->index = 624;
}

static void np_mt_twist(NumpyMT19937 *rng) {
    for (int i = 0; i < 624; i++) {
        uint32_t y = (rng->mt[i] & 0x80000000u) | (rng->mt[(i + 1) % 624] & 0x7fffffffu);
        rng->mt[i] = rng->mt[(i + 397) % 624] ^ (y >> 1);
        if (y & 1u) rng->mt[i] ^= 0x9908b0dfu;
    }
    rng->index = 0;
}

static uint32_t np_mt_uint32(NumpyMT19937 *rng) {
    if (rng->index >= 624) np_mt_twist(rng);
    uint32_t y = rng->mt[rng->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

static size_t np_mt_interval(NumpyMT19937 *rng, size_t bound) {
    if (bound <= 1) return 0;
    uint64_t threshold = ((uint64_t)0x100000000ULL - (uint64_t)bound) % (uint64_t)bound;
    for (;;) {
        uint32_t x = np_mt_uint32(rng);
        if ((uint64_t)x >= threshold) return (size_t)((uint64_t)x % (uint64_t)bound);
    }
}

void shuffle_raw_samples(RawSample *data, size_t n, unsigned int seed) {
    NumpyMT19937 rng;
    np_mt_seed(&rng, (uint32_t)seed);
    for (size_t i = n; i > 1; i--) {
        size_t j = np_mt_interval(&rng, i);
        RawSample tmp = data[i - 1];
        data[i - 1] = data[j];
        data[j] = tmp;
    }
}

void shuffle_sequences(SequenceSample *data, size_t n, unsigned int seed) {
    NumpyMT19937 rng;
    np_mt_seed(&rng, (uint32_t)seed);
    for (size_t i = n; i > 1; i--) {
        size_t j = np_mt_interval(&rng, i);
        SequenceSample tmp = data[i - 1];
        data[i - 1] = data[j];
        data[j] = tmp;
    }
}

size_t sample_sequences(SequenceSample *data, size_t n, double frac, unsigned int seed) {
    if (frac >= 1.0 || n == 0) return n;
    size_t keep = (size_t)(frac * (double)n + 0.5);
    if (keep < 1) keep = 1;
    if (keep > n) keep = n;
    shuffle_sequences(data, n, seed);   // random order, so first `keep` are uniform sample
    for (size_t k = keep; k < n; k++) {
        free(data[k].x);
        data[k].x = NULL;
    }
    return keep;
}

// Row-level analog of balance_binary_sequences. Unused: balancing is per sequence.
size_t balance_binary_rows(const RawSample *rows, size_t n_rows, RawSample **balanced_rows_out, real benign_to_attack_ratio, unsigned int seed) {
    size_t neg_count = 0, pos_count = 0;
    if (!rows || !balanced_rows_out || n_rows == 0) {
        *balanced_rows_out = NULL;
        return 0;
    }

    for (size_t i = 0; i < n_rows; i++) {
        if (rows[i].label == 1) pos_count++;
        else neg_count++;
    }

    if (pos_count == 0 || neg_count == 0) {
        RawSample *out = (RawSample *)malloc(n_rows * sizeof(RawSample));
        if (!out) return 0;
        memcpy(out, rows, n_rows * sizeof(RawSample));
        *balanced_rows_out = out;
        return n_rows;
    }

    size_t *neg_idx = (size_t *)malloc(neg_count * sizeof(size_t));
    size_t *pos_idx = (size_t *)malloc(pos_count * sizeof(size_t));
    if (!neg_idx || !pos_idx) {
        free(neg_idx);
        free(pos_idx);
        return 0;
    }

    size_t ni = 0, pi = 0;
    for (size_t i = 0; i < n_rows; i++) {
        if (rows[i].label == 1) pos_idx[pi++] = i;
        else neg_idx[ni++] = i;
    }

    size_t keep_neg = (size_t)(benign_to_attack_ratio * (real)pos_count);
    if (keep_neg > neg_count) keep_neg = neg_count;
    size_t keep_count = pos_count + keep_neg;

    NumpyMT19937 rng;
    np_mt_seed(&rng, (uint32_t)seed);
    for (size_t i = neg_count; i > 1; i--) {
        size_t j = np_mt_interval(&rng, i);
        size_t tmp = neg_idx[i - 1];
        neg_idx[i - 1] = neg_idx[j];
        neg_idx[j] = tmp;
    }

    size_t *keep_idx = (size_t *)malloc(keep_count * sizeof(size_t));
    if (!keep_idx) {
        free(neg_idx);
        free(pos_idx);
        return 0;
    }
    for (size_t i = 0; i < pos_count; i++) keep_idx[i] = pos_idx[i];
    for (size_t i = 0; i < keep_neg; i++) keep_idx[pos_count + i] = neg_idx[i];
    for (size_t i = keep_count; i > 1; i--) {
        size_t j = np_mt_interval(&rng, i);
        size_t tmp = keep_idx[i - 1];
        keep_idx[i - 1] = keep_idx[j];
        keep_idx[j] = tmp;
    }

    RawSample *out = (RawSample *)malloc(keep_count * sizeof(RawSample));
    if (!out) {
        free(neg_idx);
        free(pos_idx);
        free(keep_idx);
        return 0;
    }
    for (size_t i = 0; i < keep_count; i++) out[i] = rows[keep_idx[i]];

    free(neg_idx);
    free(pos_idx);
    free(keep_idx);
    *balanced_rows_out = out;
    return keep_count;
}

size_t balance_binary_sequences(SequenceSample *seqs, size_t n, SequenceSample **out_seqs, real benign_to_attack_ratio, unsigned int seed) {
    if (!seqs || !out_seqs) return 0;
    *out_seqs = NULL;

    size_t neg_count = 0, pos_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (seqs[i].label == 1) pos_count++;
        else neg_count++;
    }

    if (n == 0 || pos_count == 0 || neg_count == 0) {
        SequenceSample *out = (SequenceSample *)malloc((n > 0 ? n : 0) * sizeof(SequenceSample));
        if (n > 0 && !out) {
            free_sequences(seqs, n);
            return 0;
        }
        for (size_t i = 0; i < n; i++) out[i] = seqs[i];
        free(seqs);
        *out_seqs = out;
        return n;
    }

    size_t *neg_idx = (size_t *)malloc(neg_count * sizeof(size_t));
    size_t *pos_idx = (size_t *)malloc(pos_count * sizeof(size_t));
    if (!neg_idx || !pos_idx) {
        free(neg_idx);
        free(pos_idx);
        free_sequences(seqs, n);
        return 0;
    }

    size_t ni = 0, pi = 0;
    for (size_t i = 0; i < n; i++) {
        if (seqs[i].label == 1) pos_idx[pi++] = i;
        else neg_idx[ni++] = i;
    }

    size_t keep_neg = (size_t)(benign_to_attack_ratio * (real)pos_count);
    if (keep_neg > neg_count) keep_neg = neg_count;
    size_t keep_count = pos_count + keep_neg;

    NumpyMT19937 rng;
    np_mt_seed(&rng, (uint32_t)seed);
    for (size_t i = neg_count; i > 1; i--) {
        size_t j = np_mt_interval(&rng, i);
        size_t tmp = neg_idx[i - 1];
        neg_idx[i - 1] = neg_idx[j];
        neg_idx[j] = tmp;
    }

    size_t *keep_idx = (size_t *)malloc(keep_count * sizeof(size_t));
    if (!keep_idx) {
        free(neg_idx);
        free(pos_idx);
        free_sequences(seqs, n);
        return 0;
    }
    for (size_t i = 0; i < pos_count; i++) keep_idx[i] = pos_idx[i];
    for (size_t i = 0; i < keep_neg; i++) keep_idx[pos_count + i] = neg_idx[i];
    for (size_t i = keep_count; i > 1; i--) {
        size_t j = np_mt_interval(&rng, i);
        size_t tmp = keep_idx[i - 1];
        keep_idx[i - 1] = keep_idx[j];
        keep_idx[j] = tmp;
    }

    // Mark kept slots so dropped feature buffers freed without touching kept
    // ones. Ownership of seqs[i].x moves to output array, no copy.
    char *keep_mask = (char *)calloc(n, 1);
    if (!keep_mask) {
        free(neg_idx);
        free(pos_idx);
        free(keep_idx);
        free_sequences(seqs, n);
        return 0;
    }
    for (size_t i = 0; i < keep_count; i++) keep_mask[keep_idx[i]] = 1;

    SequenceSample *out = (SequenceSample *)malloc(keep_count * sizeof(SequenceSample));
    if (!out) {
        free(neg_idx);
        free(pos_idx);
        free(keep_idx);
        free(keep_mask);
        free_sequences(seqs, n);
        return 0;
    }
    for (size_t i = 0; i < keep_count; i++) out[i] = seqs[keep_idx[i]];

    for (size_t i = 0; i < n; i++) {
        if (!keep_mask[i]) free(seqs[i].x);
    }

    free(neg_idx);
    free(pos_idx);
    free(keep_idx);
    free(keep_mask);
    free(seqs);

    *out_seqs = out;
    return keep_count;
}

void free_row_array(RawSample *rows, size_t count) {
    if (!rows) return;
    for (size_t i = 0; i < count; i++) free(rows[i].features);
    free(rows);
}

void free_raw_samples(RawSample *rows, size_t count) {
    free_row_array(rows, count);
}

void free_sequences(SequenceSample *seqs, size_t count) {
    if (!seqs) return;
    for (size_t i = 0; i < count; i++) free(seqs[i].x);
    free(seqs);
}

size_t build_sequences(const RawSample *rows, size_t n_rows, SequenceSample *seqs, size_t max_seqs, const RNNConfig *cfg) {
    size_t count = 0;

    if (n_rows < cfg->seq_len) return 0;

    for (size_t start = 0; start + cfg->seq_len <= n_rows && count < max_seqs; start++) {
        real *x = (real *)malloc(cfg->seq_len * cfg->input_size * sizeof(real));
        if (!x) break;

        for (size_t t = 0; t < cfg->seq_len; t++) {
            memcpy(&x[t * cfg->input_size], rows[start + t].features, cfg->input_size * sizeof(real));
        }

        seqs[count].x = x;
        seqs[count].label = rows[start + cfg->seq_len - 1].label;
        count++;
    }

    return count;
}

void compute_class_counts(const SequenceSample *data, size_t n, int *neg_count, int *pos_count) {
    *neg_count = 0;
    *pos_count = 0;

    for (size_t i = 0; i < n; i++) {
        if (data[i].label == 1) (*pos_count)++;
        else (*neg_count)++;
    }
}
