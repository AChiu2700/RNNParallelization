#ifndef DEFAULT_RNN_SEQ_VERTICLE_H
#define DEFAULT_RNN_SEQ_VERTICLE_H

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

typedef float real;

typedef struct {
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
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
 * Timestep-pipeline partitioning: each thread is stage owning contiguous
 * slice of timesteps. Sequences stream through GPipe-style, forward stage 0 ->
 * last, backward reversed, so in steady state every stage busy on different
 * sequence.
 *
 * Depth capped at seq_len (stage needs >=1 timestep), so
 * num_stages = min(num_threads, seq_len).
 */
typedef struct {
    size_t ts_lo;  // first owned timestep (inclusive)
    size_t ts_hi;  // one past last owned timestep (exclusive)
} StageRange;

/*
 * Pipeline scratch, allocated once in main() before persistent parallel
 * region opens, shared by all stage threads. Local array in cooperative
 * function would be per-thread stack copy.
 *
 * Sized for one batch of in-flight sequences: forward keeps every sequence's
 * full hidden trace since reverse wavefront reads it back, and dh_carry holds
 * backprop signal handed down toward stage 0.
 */
typedef struct {
    size_t capacity;  // == batch_size: max sequences in flight
    real *H_trace;     /* capacity * (seq_len+1) * hidden_size: per-sequence forward hidden trace */
    real *dh_carry;    /* capacity * hidden_size: per-sequence backprop signal handed between stages */
    real *logit;  // capacity: final-timestep logit per sequence
    real *score;  // capacity: sigmoid(logit) per sequence
    double *loss;  // capacity: per-sequence loss (weighted in training, plain in eval)
    int *pred_label;  // capacity: predicted label per sequence
} VerticalScratch;

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
int allocate_vertical_scratch(VerticalScratch *scratch, const RNNConfig *cfg);
void free_vertical_scratch(VerticalScratch *scratch);
/* Split seq_len timesteps contiguously across up to num_threads stages.
 * Writes ranges, sets *num_stages_out = min(num_threads, seq_len). */
void compute_stage_ranges(StageRange *ranges, int num_threads, const RNNConfig *cfg, int *num_stages_out);
void rnn_init(SequentialRNN *rnn, const RNNConfig *cfg, unsigned int seed);
real rnn_sigmoid(real x);
void rnn_zero_gradients(RNNGradients *grads, const RNNConfig *cfg);

/*
 * Cooperative pipeline drivers: all threads of enclosing parallel region call
 * these together, same args. Each thread is stage `tid` running per-tick
 * barrier wavefront. Threads with tid >= num_stages are idle but still enter,
 * since they take part in team-wide barriers. ranges/num_stages from
 * compute_stage_ranges().
 *
 * forward_batch runs forward wavefront over batch, fills H_trace and, on stage
 * owning final timestep, logit/score/loss/pred_label. If weighted_loss, loss
 * is pos_weight-scaled for training report, else plain loss for eval.
 *
 * backward_batch runs reverse wavefront, must follow forward_batch on same
 * batch. Accumulates into stage's own buffer, no cross-stage race. Caller
 * merges per-stage buffers in fixed stage order after region and applies once,
 * so result deterministic in thread count and bit-exact vs serial at
 * num_stages==1.
 */
void rnn_pipeline_forward_batch(SequentialRNN *rnn, const SequenceSample *batch, int batch_count,
                                const RNNConfig *cfg, real pos_weight, int weighted_loss,
                                const StageRange *ranges, int num_stages,
                                int tid, VerticalScratch *scratch);
void rnn_pipeline_backward_batch(SequentialRNN *rnn, const SequenceSample *batch, int batch_count,
                                 const RNNConfig *cfg, real pos_weight,
                                 const StageRange *ranges, int num_stages,
                                 int tid, VerticalScratch *scratch, RNNGradients *stage_grads_tid);
void rnn_add_gradients(RNNGradients *dst, const RNNGradients *src, const RNNConfig *cfg);

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
