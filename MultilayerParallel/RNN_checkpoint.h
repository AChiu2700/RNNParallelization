#ifndef RNN_CHECKPOINT_H
#define RNN_CHECKPOINT_H

/* Checkpoint / resume for multi-layer programs. Same file in each folder;
 * include after folder's RNN_seq_*.h (uses SequentialRNN, RNNConfig,
 * rnn_*_total helpers).
 *
 * Plain SGD, no optimizer state, training order fixed by seeded shuffle before
 * epoch loop, so checkpoint = weights + position in training. Resume rebuilds
 * data pipeline from same flags (same windows, same order) and continues.
 *
 * File layout: RNNCkptHeader, then Wxh, Whh, Why, bh, by as raw `real` arrays.
 * Write to PATH.tmp, fsync, rename; previous good file kept as PATH.prev, so
 * crash mid-write always leaves one valid checkpoint. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char magic[8];             /* "RNNCKPT1" */
    uint32_t real_size;        /* sizeof(real): float and double builds do not mix */
    uint32_t reserved;
    uint64_t input_size, hidden_size, output_size, num_layers, seq_len, batch_size;
    uint64_t train_size, test_size, shuffle_seed;
    double sample_frac;
    uint64_t epoch;            /* 0-based epoch in progress (== finished epochs when next_batch == 0) */
    uint64_t next_batch;       /* index of next training window in that epoch */
    uint64_t train_correct;    /* running counters for partly finished epoch */
    double train_loss;
    double elapsed;            /* training seconds spent before this save */
} RNNCkptHeader;

static size_t rnn_ckpt_why_total(const RNNConfig *cfg) { return cfg->output_size * cfg->hidden_size; }

static int rnn_ckpt_write_arrays(FILE *f, const SequentialRNN *rnn, const RNNConfig *cfg) {
    const size_t n[5] = { rnn_wxh_total(cfg), rnn_whh_total(cfg), rnn_ckpt_why_total(cfg), rnn_bh_total(cfg), cfg->output_size };
    const real *p[5] = { rnn->Wxh, rnn->Whh, rnn->Why, rnn->bh, rnn->by };
    for (int i = 0; i < 5; i++)
        if (fwrite(p[i], sizeof(real), n[i], f) != n[i]) return 0;
    return 1;
}

static int rnn_ckpt_read_arrays(FILE *f, SequentialRNN *rnn, const RNNConfig *cfg) {
    const size_t n[5] = { rnn_wxh_total(cfg), rnn_whh_total(cfg), rnn_ckpt_why_total(cfg), rnn_bh_total(cfg), cfg->output_size };
    real *p[5] = { rnn->Wxh, rnn->Whh, rnn->Why, rnn->bh, rnn->by };
    for (int i = 0; i < 5; i++)
        if (fread(p[i], sizeof(real), n[i], f) != n[i]) return 0;
    return 1;
}

/* Returns 1 on success. Failed save prints warning, returns 0; run continues,
 * one lost checkpoint should not kill multi-day job. */
static int rnn_ckpt_save(const char *path, const SequentialRNN *rnn, const RNNConfig *cfg,
                         size_t train_size, size_t test_size, double sample_frac,
                         int epoch, size_t next_batch, int train_correct, double train_loss, double elapsed) {
    char tmp[4096], prev[4096];
    if (snprintf(tmp, sizeof tmp, "%s.tmp", path) >= (int)sizeof tmp ||
        snprintf(prev, sizeof prev, "%s.prev", path) >= (int)sizeof prev) {
        fprintf(stderr, "checkpoint: path too long\n");
        return 0;
    }

    RNNCkptHeader h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, "RNNCKPT1", 8);
    h.real_size = (uint32_t)sizeof(real);
    h.input_size = cfg->input_size;   h.hidden_size = cfg->hidden_size;
    h.output_size = cfg->output_size; h.num_layers = cfg->num_layers;
    h.seq_len = cfg->seq_len;         h.batch_size = cfg->batch_size;
    h.train_size = train_size;        h.test_size = test_size;
    h.shuffle_seed = cfg->shuffle_seed;
    h.sample_frac = sample_frac;
    h.epoch = (uint64_t)epoch;        h.next_batch = next_batch;
    h.train_correct = (uint64_t)train_correct;
    h.train_loss = train_loss;        h.elapsed = elapsed;

    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "checkpoint: cannot write %s\n", tmp); return 0; }
    int ok = fwrite(&h, sizeof h, 1, f) == 1 && rnn_ckpt_write_arrays(f, rnn, cfg) && fflush(f) == 0 && fsync(fileno(f)) == 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok) { fprintf(stderr, "checkpoint: write failed for %s\n", tmp); remove(tmp); return 0; }

    (void)rename(path, prev);              /* keep last good one; fine if none exists */
    if (rename(tmp, path) != 0) { fprintf(stderr, "checkpoint: rename failed for %s\n", path); return 0; }
    return 1;
}

static int rnn_ckpt_try_load(const char *file, SequentialRNN *rnn, const RNNConfig *cfg,
                             size_t train_size, size_t test_size, double sample_frac, RNNCkptHeader *out) {
    FILE *f = fopen(file, "rb");
    if (!f) return 0;
    RNNCkptHeader h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, "RNNCKPT1", 8) != 0) {
        fprintf(stderr, "checkpoint: %s is not a valid checkpoint\n", file);
        fclose(f); return 0;
    }
    const char *bad = NULL;
    if (h.real_size != (uint32_t)sizeof(real)) bad = "float/double precision";
    else if (h.input_size != cfg->input_size) bad = "input size";
    else if (h.hidden_size != cfg->hidden_size) bad = "hidden size";
    else if (h.output_size != cfg->output_size) bad = "output size";
    else if (h.num_layers != cfg->num_layers) bad = "layer count";
    else if (h.seq_len != cfg->seq_len) bad = "sequence length";
    else if (h.batch_size != cfg->batch_size) bad = "batch size";
    else if (h.train_size != train_size) bad = "training window count (rows, sampling or seed differ)";
    else if (h.test_size != test_size) bad = "test window count (rows, sampling or seed differ)";
    else if (h.shuffle_seed != cfg->shuffle_seed) bad = "shuffle seed";
    else if (h.sample_frac != sample_frac) bad = "sample fraction";
    if (bad) {
        fprintf(stderr, "checkpoint: %s does not match this run (%s)\n", file, bad);
        fclose(f); return 0;
    }
    if (!rnn_ckpt_read_arrays(f, rnn, cfg)) {
        fprintf(stderr, "checkpoint: %s is truncated\n", file);
        fclose(f); return 0;
    }
    fclose(f);
    *out = h;
    return 1;
}

/* Tries PATH, then PATH.prev (crash between two renames in save leaves only
 * .prev). Returns 1 and fills `out` on success. */
static int rnn_ckpt_load(const char *path, SequentialRNN *rnn, const RNNConfig *cfg,
                         size_t train_size, size_t test_size, double sample_frac, RNNCkptHeader *out) {
    char prev[4096];
    if (rnn_ckpt_try_load(path, rnn, cfg, train_size, test_size, sample_frac, out)) return 1;
    if (snprintf(prev, sizeof prev, "%s.prev", path) < (int)sizeof prev &&
        rnn_ckpt_try_load(prev, rnn, cfg, train_size, test_size, sample_frac, out)) {
        fprintf(stderr, "checkpoint: used %s\n", prev);
        return 1;
    }
    return 0;
}

#endif
