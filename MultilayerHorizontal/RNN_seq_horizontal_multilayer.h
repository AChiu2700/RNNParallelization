#ifndef RNN_SEQ_HORIZONTAL_MULTILAYER_H
#define RNN_SEQ_HORIZONTAL_MULTILAYER_H

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef MAX_LINE
#define MAX_LINE 65536
#endif

#ifndef DEFAULT_MAX_ROWS
/* Load full datasets by default: CIC combined ~2.83M rows, KDD full ~4.9M.
 * 10M leaves headroom. Override with --max-rows. */
#define DEFAULT_MAX_ROWS 10000000UL
#endif

/* Model precision.
 *
 * Default float.
 *
 * -DRNN_USE_DOUBLE switches whole model to double, for finite-difference
 * gradcheck.c: analytic gradient vs (L(w+eps)-L(w-eps))/2eps. Subtraction
 * eats most of float's ~7 digits, so float check only good to few percent.
 * Double good to ~1e-9.
 *
 * Macros below expand to same float calls in default build. */
#ifdef RNN_USE_DOUBLE
typedef double real;
#define RNN_TANH tanh
#define RNN_LOG  log
#define RNN_EXP  exp
#define RNN_MIN  fmin
#define RNN_MAX  fmax
#else
typedef float real;
#define RNN_TANH tanhf
#define RNN_LOG  logf
#define RNN_EXP  expf
#define RNN_MIN  fminf
#define RNN_MAX  fmaxf
#endif

typedef struct {
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    size_t num_layers;  // L. 1 = single-layer model (default).
    size_t seq_len;
    size_t batch_size;
    real learning_rate;
    real grad_clip;
    real epsilon;
    int num_epochs;
    unsigned int shuffle_seed;
} RNNConfig;

typedef struct {
    int has_header;
    char delimiter;
    int label_col;
    int *feature_cols;
    size_t feature_count;
    int *drop_cols;
    size_t drop_count;
    int positive_label_numeric;
    double positive_label_value;
    char positive_label_text[128];
    char negative_label_text[128];
    int allow_text_labels;
    int infer_label_col;
} DatasetSchema;

typedef struct {
    real *features;
    int label;
} RawSample;

typedef struct {
    real *x;
    int label;
} SequenceSample;

typedef struct {
    real *Wxh;
    real *Whh;
    real *Why;
    real *bh;
    real *by;
} SequentialRNN;

typedef struct {
    real *mean;
    real *std;
} NormalizationStats;

typedef struct {
    real *dWxh;
    real *dWhh;
    real *dWhy;
    real *dbh;
    real *dby;
} RNNGradients;

typedef struct {
    int pred_label;
    real pred_score;
    double loss;
} RNNPrediction;

typedef struct {
    size_t total_lines;
    size_t valid_rows;
    size_t skipped_lines;
    size_t total_sequences;
} DatasetStats;

/*
 * Shared scratch for hidden-dim partitioning. Allocate once before enclosing
 * parallel region opens, pass by pointer into forward and gradient functions.
 *
 * Local array inside those functions would not be shared: each thread gets own
 * stack frame per call, whatever OpenMP data-sharing attribute, so buffers
 * whose slices different threads read and write must live outside thread stacks.
 *
 * One instance for whole run: samples processed one at a time, all threads
 * on same sample.
 */
typedef struct {
    real *h;           /* rnn_h_total(cfg) = L*(seq_len+1)*hidden_size, forward/backward trace */
    real *dh_cur;      /* L * hidden_size, backprop signal into each layer from timestep t+1 */
    real *dh_prev;     /* L * hidden_size, backprop signal accumulated for timestep t-1 */
    real *dh_up;       /* L * hidden_size, vertical gradient layer l+1 hands down to layer l at
                        * same timestep. Zeroed per timestep, written by layer above through
                        * array-section reduction (fan-in over its owned hidden units, like
                        * dh_prev), then read by layer below. Unused at L=1. */
    real *logit_accum; /* 1 element, shared reduction target for final-timestep logit. Needs shared
                         * slot for same reason as h/dh: plain local as reduction variable on orphaned
                         * omp for is separate stack variable per thread, nothing shared to combine into. */
} HorizontalScratch;

/* ---------------------------------------------------------------------------
 * Layer-indexed weight layout
 *
 * Weights are flat contiguous buffers plus per-layer offset table, not
 * `real **`: batch gradient merge walks buffers end to end, and hidden-dim
 * array-section reduction needs contiguous section.
 *
 *   Wxh      layer 0 is H x I (raw features); layers above are H x H
 *            (hidden state below)
 *   Whh      L blocks of H x H
 *   bh       L blocks of H
 *   Why, by  top layer only, applied to its final hidden state
 *
 * At L=1 totals collapse to single-layer sizes and rnn_init draws weights in
 * same order, so L=1 model matches single-layer program exactly.
 * --------------------------------------------------------------------------- */

// Input width of layer l: raw features for layer 0, hidden size above it.
static inline size_t rnn_layer_in_size(const RNNConfig *cfg, size_t l) {
    return (l == 0) ? cfg->input_size : cfg->hidden_size;
}
static inline size_t rnn_wxh_off(const RNNConfig *cfg, size_t l) {
    return (l == 0) ? 0
                    : cfg->hidden_size * cfg->input_size
                      + (l - 1) * cfg->hidden_size * cfg->hidden_size;
}
static inline size_t rnn_wxh_total(const RNNConfig *cfg) {
    return cfg->hidden_size * cfg->input_size
         + (cfg->num_layers - 1) * cfg->hidden_size * cfg->hidden_size;
}
static inline size_t rnn_whh_off(const RNNConfig *cfg, size_t l) {
    return l * cfg->hidden_size * cfg->hidden_size;
}
static inline size_t rnn_whh_total(const RNNConfig *cfg) {
    return cfg->num_layers * cfg->hidden_size * cfg->hidden_size;
}
static inline size_t rnn_bh_off(const RNNConfig *cfg, size_t l) {
    return l * cfg->hidden_size;
}
static inline size_t rnn_bh_total(const RNNConfig *cfg) {
    return cfg->num_layers * cfg->hidden_size;
}
// hidden trace, indexed [layer][timestep 0..seq_len][hidden]
static inline size_t rnn_h_off(const RNNConfig *cfg, size_t l, size_t t) {
    return (l * (cfg->seq_len + 1) + t) * cfg->hidden_size;
}
static inline size_t rnn_h_total(const RNNConfig *cfg) {
    return cfg->num_layers * (cfg->seq_len + 1) * cfg->hidden_size;
}

void init_default_config(RNNConfig *cfg);
void init_default_schema(DatasetSchema *schema);
void free_schema(DatasetSchema *schema);

int parse_int_list(const char *text, int **values_out, size_t *count_out);
int split_csv_line(char *line, char delimiter, char ***fields_out, int *count_out);
int detect_header_row(char *line, char delimiter);
int infer_label_column(char **fields, int count);
int infer_feature_columns(char **fields, int count, int label_col, const int *drop_cols, size_t drop_count, int **feature_cols_out, size_t *feature_count_out);
int encode_label(const char *text, const DatasetSchema *schema, int *label_out);
int parse_row_with_schema(char *line, const DatasetSchema *schema, real *features, int *label);

int allocate_rnn(SequentialRNN *rnn, const RNNConfig *cfg);
void free_rnn(SequentialRNN *rnn);
int allocate_gradients(RNNGradients *grads, const RNNConfig *cfg);
void free_gradients(RNNGradients *grads);
int allocate_stats(NormalizationStats *stats, const RNNConfig *cfg);
void free_stats(NormalizationStats *stats);
int allocate_horizontal_scratch(HorizontalScratch *scratch, const RNNConfig *cfg);
void free_horizontal_scratch(HorizontalScratch *scratch);
void rnn_init(SequentialRNN *rnn, const RNNConfig *cfg, unsigned int seed);
real rnn_sigmoid(real x);
void rnn_zero_gradients(RNNGradients *grads, const RNNConfig *cfg);

/*
 * Cooperative: called by all threads of enclosing omp parallel region
 * together (same call, same args); each thread computes own static slice of
 * hidden dim. Not safe from one thread while others idle.
 */
RNNPrediction rnn_forward_sequence(SequentialRNN *rnn, const SequenceSample *sample, const RNNConfig *cfg, HorizontalScratch *scratch);
RNNPrediction rnn_accumulate_sequence_gradients(SequentialRNN *rnn, const SequenceSample *sample, const RNNConfig *cfg, real pos_weight, RNNGradients *grads, HorizontalScratch *scratch);

void rnn_apply_gradients(SequentialRNN *rnn, const RNNGradients *grads, const RNNConfig *cfg, int batch_count);
void compute_normalization_stats(const SequenceSample *data, size_t n_samples, const RNNConfig *cfg, NormalizationStats *stats);
void apply_normalization(SequenceSample *data, size_t n_samples, const RNNConfig *cfg, const NormalizationStats *stats);

size_t balance_binary_rows(const RawSample *rows, size_t n_rows, RawSample **balanced_rows_out, real benign_to_attack_ratio, unsigned int seed);
/* Sequence-level 1:1 balancing: undersamples benign windows, not raw rows,
 * so kept windows stay consecutive in time. Consumes seqs: moves kept windows
 * to *out_seqs (caller owns), frees dropped windows + input array. */
size_t balance_binary_sequences(SequenceSample *seqs, size_t n, SequenceSample **out_seqs, real benign_to_attack_ratio, unsigned int seed);
void free_row_array(RawSample *rows, size_t count);
void free_raw_samples(RawSample *rows, size_t count);
void free_sequences(SequenceSample *seqs, size_t count);
size_t build_sequences(const RawSample *rows, size_t n_rows, SequenceSample *seqs, size_t max_seqs, const RNNConfig *cfg);
void shuffle_raw_samples(RawSample *data, size_t n, unsigned int seed);
void shuffle_sequences(SequenceSample *data, size_t n, unsigned int seed);
// Keeps seeded uniform random fraction of windows (0 < frac <= 1), frees rest,
// returns new count. frac >= 1 is no-op.
size_t sample_sequences(SequenceSample *data, size_t n, double frac, unsigned int seed);
void compute_class_counts(const SequenceSample *data, size_t n, int *neg_count, int *pos_count);
void log_both(FILE *out, const char *fmt, ...);

int load_numeric_dataset(const char *dataset_path,
                         RNNConfig *cfg,
                         DatasetSchema *schema,
                         size_t max_rows,
                         FILE *log,
                         RawSample **rows_out,
                         DatasetStats *stats_out);

#endif
