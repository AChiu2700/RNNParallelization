#ifndef CIC_SEQUENTIAL_RNN_H
#define CIC_SEQUENTIAL_RNN_H

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * CIC raw dataset version.
 * Expected raw CIC-IDS style row:
 *   optional Unnamed/index column + 78 numeric feature columns + text Label
 * Examples:
 *   80 columns: Unnamed: 0, feature1 ... feature78, Label
 *   79 columns: feature1 ... feature78, Label
 */

typedef float real;

#define INPUT_SIZE 78
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 1

#define SEQ_LEN 8
#define BATCH_SIZE 16

#define LEARNING_RATE 0.001f
#define GRAD_CLIP 5.0f
#define EPSILON 1e-7f
#define NUM_EPOCHS 10
#define SHUFFLE_SEED 42u

typedef struct {
    real x[SEQ_LEN][INPUT_SIZE];
    int label;
} SequenceSample;

typedef struct {
    real Wxh[HIDDEN_SIZE][INPUT_SIZE];
    real Whh[HIDDEN_SIZE][HIDDEN_SIZE];
    real Why[OUTPUT_SIZE][HIDDEN_SIZE];
    real bh[HIDDEN_SIZE];
    real by[OUTPUT_SIZE];
} SequentialRNN;

typedef struct {
    real mean[INPUT_SIZE];
    real std[INPUT_SIZE];
} NormalizationStats;

typedef struct {
    real dWxh[HIDDEN_SIZE][INPUT_SIZE];
    real dWhh[HIDDEN_SIZE][HIDDEN_SIZE];
    real dWhy[OUTPUT_SIZE][HIDDEN_SIZE];
    real dbh[HIDDEN_SIZE];
    real dby[OUTPUT_SIZE];
} RNNGradients;

typedef struct {
    int pred_label;
    real pred_score;
    double loss;
} RNNPrediction;

void rnn_init(SequentialRNN *rnn, unsigned int seed);
real rnn_sigmoid(real x);

int rnn_parse_csv_line(char *line, real *features, int *label);

void compute_normalization_stats(
    const SequenceSample *data,
    int n_samples,
    NormalizationStats *stats
);

void apply_normalization(
    SequenceSample *data,
    int n_samples,
    const NormalizationStats *stats
);

void rnn_zero_gradients(RNNGradients *grads);

RNNPrediction rnn_forward_sequence(
    SequentialRNN *rnn,
    const SequenceSample *sample
);

RNNPrediction rnn_accumulate_sequence_gradients(
    SequentialRNN *rnn,
    const SequenceSample *sample,
    real pos_weight,
    RNNGradients *grads
);

void rnn_apply_gradients(
    SequentialRNN *rnn,
    const RNNGradients *grads,
    real learning_rate,
    int batch_count
);

#endif
