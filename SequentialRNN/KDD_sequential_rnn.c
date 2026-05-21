#include "KDD_sequential_rnn.h"

static real rand_weight(void) {
    return ((real)rand() / (real)RAND_MAX) * 0.2f - 0.1f;
}

static real clip_grad(real x) {
    if (x > GRAD_CLIP) return GRAD_CLIP;
    if (x < -GRAD_CLIP) return -GRAD_CLIP;
    return x;
}

static void trim_whitespace(char *s) {
    char *start = s;
    while (isspace((unsigned char)*start)) start++;

    if (*start == '\0') {
        *s = '\0';
        return;
    }

    if (start != s) {
        memmove(s, start, strlen(start) + 1);
    }

    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
}

real rnn_sigmoid(real x) {
    if (x >= 0.0f) {
        real z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    real z = expf(x);
    return z / (1.0f + z);
}

void rnn_init(SequentialRNN *rnn, unsigned int seed) {
    srand(seed);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            rnn->Wxh[i][j] = rand_weight();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            rnn->Whh[i][j] = rand_weight();
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            rnn->Why[i][j] = rand_weight();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        rnn->bh[i] = 0.0f;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        rnn->by[i] = 0.0f;
    }
}

int rnn_parse_csv_line(char *line, real *features, int *label) {
    char *tokens[INPUT_SIZE + 1];
    int count = 0;

    char *ptr = strtok(line, ",");
    while (ptr != NULL && count < INPUT_SIZE + 1) {
        tokens[count++] = ptr;
        ptr = strtok(NULL, ",");
    }

    if (count != INPUT_SIZE + 1) {
        return 0;
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        char *endptr;
        trim_whitespace(tokens[i]);
        features[i] = strtof(tokens[i], &endptr);
        if (endptr == tokens[i]) {
            return 0;
        }
    }

    trim_whitespace(tokens[INPUT_SIZE]);
    char *endptr;
    float label_value = strtof(tokens[INPUT_SIZE], &endptr);
    if (endptr == tokens[INPUT_SIZE]) {
        return 0;
    }

    *label = (label_value > 0.5f) ? 1 : 0;
    return 1;
}

void compute_normalization_stats(
    const SequenceSample *data,
    int n_samples,
    NormalizationStats *stats
) {
    const double total_steps = (double)n_samples * (double)SEQ_LEN;

    for (int d = 0; d < INPUT_SIZE; d++) {
        double sum = 0.0;
        double sq_sum = 0.0;

        for (int i = 0; i < n_samples; i++) {
            for (int t = 0; t < SEQ_LEN; t++) {
                double v = data[i].x[t][d];
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

void apply_normalization(
    SequenceSample *data,
    int n_samples,
    const NormalizationStats *stats
) {
    for (int i = 0; i < n_samples; i++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            for (int d = 0; d < INPUT_SIZE; d++) {
                data[i].x[t][d] =
                    (data[i].x[t][d] - stats->mean[d]) / (stats->std[d] + EPSILON);
            }
        }
    }
}

void rnn_zero_gradients(RNNGradients *grads) {
    memset(grads, 0, sizeof(RNNGradients));
}

static void rnn_step(
    SequentialRNN *rnn,
    const real *xt,
    const real *h_prev,
    real *h_new,
    real *logit
) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        real z = rnn->bh[j];

        for (int k = 0; k < INPUT_SIZE; k++) {
            z += rnn->Wxh[j][k] * xt[k];
        }

        for (int k = 0; k < HIDDEN_SIZE; k++) {
            z += rnn->Whh[j][k] * h_prev[k];
        }

        h_new[j] = tanhf(z);
    }

    logit[0] = rnn->by[0];
    for (int k = 0; k < HIDDEN_SIZE; k++) {
        logit[0] += rnn->Why[0][k] * h_new[k];
    }
}

RNNPrediction rnn_forward_sequence(
    SequentialRNN *rnn,
    const SequenceSample *sample
) {
    real h[SEQ_LEN + 1][HIDDEN_SIZE] = {{0}};
    real y[SEQ_LEN][OUTPUT_SIZE] = {{0}};

    for (int t = 0; t < SEQ_LEN; t++) {
        rnn_step(rnn, sample->x[t], h[t], h[t + 1], y[t]);
    }

    real score = rnn_sigmoid(y[SEQ_LEN - 1][0]);
    real target = (real)sample->label;
    real clipped = fminf(fmaxf(score, EPSILON), 1.0f - EPSILON);

    RNNPrediction pred;
    pred.pred_score = score;
    pred.loss = -(target * logf(clipped) + (1.0f - target) * logf(1.0f - clipped));
    pred.pred_label = (score >= 0.5f) ? 1 : 0;
    return pred;
}

RNNPrediction rnn_accumulate_sequence_gradients(
    SequentialRNN *rnn,
    const SequenceSample *sample,
    real pos_weight,
    RNNGradients *grads
) {
    real h[SEQ_LEN + 1][HIDDEN_SIZE] = {{0}};
    real y[SEQ_LEN][OUTPUT_SIZE] = {{0}};

    for (int t = 0; t < SEQ_LEN; t++) {
        rnn_step(rnn, sample->x[t], h[t], h[t + 1], y[t]);
    }

    real score = rnn_sigmoid(y[SEQ_LEN - 1][0]);
    real target = (real)sample->label;
    real clipped = fminf(fmaxf(score, EPSILON), 1.0f - EPSILON);
    real class_weight = (sample->label == 1) ? pos_weight : 1.0f;

    RNNPrediction pred;
    pred.pred_score = score;
    pred.loss = -class_weight * (target * logf(clipped) + (1.0f - target) * logf(1.0f - clipped));
    pred.pred_label = (score >= 0.5f) ? 1 : 0;

    real dh_next[HIDDEN_SIZE] = {0};
    real dy = class_weight * (score - target);
    grads->dby[0] += dy;

    for (int k = 0; k < HIDDEN_SIZE; k++) {
        grads->dWhy[0][k] += dy * h[SEQ_LEN][k];
        dh_next[k] = rnn->Why[0][k] * dy;
    }

    for (int t = SEQ_LEN - 1; t >= 0; t--) {
        real dh[HIDDEN_SIZE];
        real dh_prev[HIDDEN_SIZE] = {0};

        for (int k = 0; k < HIDDEN_SIZE; k++) {
            dh[k] = dh_next[k];
        }

        for (int k = 0; k < HIDDEN_SIZE; k++) {
            real dz = dh[k] * (1.0f - h[t + 1][k] * h[t + 1][k]);
            grads->dbh[k] += dz;

            for (int i = 0; i < INPUT_SIZE; i++) {
                grads->dWxh[k][i] += dz * sample->x[t][i];
            }

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                grads->dWhh[k][j] += dz * h[t][j];
                dh_prev[j] += rnn->Whh[k][j] * dz;
            }
        }

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            dh_next[j] = dh_prev[j];
        }
    }

    return pred;
}

void rnn_apply_gradients(
    SequentialRNN *rnn,
    const RNNGradients *grads,
    real learning_rate,
    int batch_count
) {
    real scale = 1.0f / (real)batch_count;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            rnn->Wxh[i][j] -= learning_rate * clip_grad(grads->dWxh[i][j] * scale);
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            rnn->Whh[i][j] -= learning_rate * clip_grad(grads->dWhh[i][j] * scale);
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        rnn->Why[0][j] -= learning_rate * clip_grad(grads->dWhy[0][j] * scale);
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        rnn->bh[i] -= learning_rate * clip_grad(grads->dbh[i] * scale);
    }

    rnn->by[0] -= learning_rate * clip_grad(grads->dby[0] * scale);
}
