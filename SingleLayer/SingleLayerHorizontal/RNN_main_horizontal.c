#include <omp.h>
#include "RNN_seq_horizontal.h"

static double safe_div(double num, double den) {
    return (den == 0.0) ? 0.0 : (num / den);
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s dataset.csv output.txt [options]\n", prog);
    fprintf(stderr,
        "Options:\n"
        "  --no-header\n"
        "  --delimiter ,|;|tab\n"
        "  --label-col N\n"
        "  --drop-cols 0,3,5\n"
        "  --feature-cols 1,2,4\n"
        "  --positive-label TEXT\n"
        "  --positive-value NUM\n"
        "  --seq-len N\n"
        "  --hidden-size N\n"
        "  --layers 1            (single-layer only; for L>1 use\n"
        "                        MultilayerHorizontal/rnn_horizontal_multilayer)\n"
        "  --batch-size N\n"
        "  --epochs N\n"
        "  --max-rows N\n"
        "  --shuffle 0|1\n");
}

// Strict strtol: whole arg must be consumed. Returns 0 on success.
static int strict_strtol(const char *s, long *out) {
    if (!s || *s == '\0') return -1;
    char *endptr;
    long v = strtol(s, &endptr, 10);
    if (endptr == s || *endptr != '\0') return -1;
    *out = v;
    return 0;
}

static int strict_strtoull(const char *s, unsigned long long *out) {
    if (!s || *s == '\0') return -1;
    char *endptr;
    unsigned long long v = strtoull(s, &endptr, 10);
    if (endptr == s || *endptr != '\0') return -1;
    *out = v;
    return 0;
}

static int strict_strtod(const char *s, double *out) {
    if (!s || *s == '\0') return -1;
    char *endptr;
    double v = strtod(s, &endptr);
    if (endptr == s || *endptr != '\0') return -1;
    *out = v;
    return 0;
}

int main(int argc, char **argv) {
    time_t wall_start = time(NULL);

    if (argc <= 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *dataset_path = argv[1];
    const char *output_path = argv[2];

    RNNConfig cfg;
    DatasetSchema schema;
    init_default_config(&cfg);
    init_default_schema(&schema);

    size_t max_rows = DEFAULT_MAX_ROWS;
    int shuffle_train = 1;

    for (int i = 3; i < argc; i++) {
        const char *flag = argv[i];

        if (strcmp(flag, "--no-header") == 0) {
            schema.has_header = 0;
            continue;
        }

        // Remaining flags all take a value.
        const char *val = (i + 1 < argc) ? argv[i + 1] : NULL;

        if (strcmp(flag, "--delimiter") == 0) {
            if (!val) { print_usage(argv[0]); return 1; }
            schema.delimiter = (strcmp(val, "tab") == 0) ? '\t' : val[0];
            i++;
        } else if (strcmp(flag, "--label-col") == 0) {
            long v;
            if (!val || strict_strtol(val, &v) != 0) { print_usage(argv[0]); return 1; }
            schema.label_col = (int)v;
            schema.infer_label_col = 0;
            i++;
        } else if (strcmp(flag, "--drop-cols") == 0) {
            if (!val || !parse_int_list(val, &schema.drop_cols, &schema.drop_count)) {
                print_usage(argv[0]);
                return 1;
            }
            i++;
        } else if (strcmp(flag, "--feature-cols") == 0) {
            if (!val || !parse_int_list(val, &schema.feature_cols, &schema.feature_count)) {
                print_usage(argv[0]);
                return 1;
            }
            i++;
        } else if (strcmp(flag, "--positive-label") == 0) {
            if (!val) { print_usage(argv[0]); return 1; }
            strncpy(schema.positive_label_text, val, sizeof(schema.positive_label_text) - 1);
            schema.positive_label_text[sizeof(schema.positive_label_text) - 1] = '\0';
            i++;
        } else if (strcmp(flag, "--positive-value") == 0) {
            double v;
            if (!val || strict_strtod(val, &v) != 0) { print_usage(argv[0]); return 1; }
            schema.positive_label_numeric = 1;
            schema.positive_label_value = v;
            i++;
        } else if (strcmp(flag, "--seq-len") == 0) {
            long v;
            if (!val || strict_strtol(val, &v) != 0 || v <= 0) { print_usage(argv[0]); return 1; }
            cfg.seq_len = (size_t)v;
            i++;
        } else if (strcmp(flag, "--hidden-size") == 0) {
            long v;
            if (!val || strict_strtol(val, &v) != 0 || v <= 0) { print_usage(argv[0]); return 1; }
            cfg.hidden_size = (size_t)v;
            i++;
        } else if (strcmp(flag, "--layers") == 0) {
            /* Single-layer program: accepts --layers 1 only; L>1 is error pointing at
             * multi-layer program. */
            long v;
            if (!val || strict_strtol(val, &v) != 0 || v <= 0) { print_usage(argv[0]); return 1; }
            if (v != 1) {
                fprintf(stderr,
                    "Error: %s is the single-layer hidden-dimension program and supports --layers 1 only "
                    "(got %ld).\n"
                    "       For L > 1 use MultilayerHorizontal/rnn_horizontal_multilayer.\n",
                    argv[0], v);
                return 1;
            }
            cfg.num_layers = (size_t)v;
            i++;
        } else if (strcmp(flag, "--batch-size") == 0) {
            long v;
            if (!val || strict_strtol(val, &v) != 0 || v <= 0) { print_usage(argv[0]); return 1; }
            cfg.batch_size = (size_t)v;
            i++;
        } else if (strcmp(flag, "--epochs") == 0) {
            long v;
            if (!val || strict_strtol(val, &v) != 0 || v <= 0) { print_usage(argv[0]); return 1; }
            cfg.num_epochs = (int)v;
            i++;
        } else if (strcmp(flag, "--max-rows") == 0) {
            unsigned long long v;
            if (!val || strict_strtoull(val, &v) != 0) { print_usage(argv[0]); return 1; }
            max_rows = (size_t)v;
            i++;
        } else if (strcmp(flag, "--shuffle") == 0) {
            long v;
            if (!val || strict_strtol(val, &v) != 0 || (v != 0 && v != 1)) { print_usage(argv[0]); return 1; }
            shuffle_train = (int)v;
            i++;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    FILE *out = fopen(output_path, "w");
    if (!out) {
        printf("Error: could not create output file: %s\n", output_path);
        return 1;
    }

    RawSample *rows = NULL;
    DatasetStats stats;
    if (!load_numeric_dataset(dataset_path, &cfg, &schema, max_rows, out, &rows, &stats)) {
        fclose(out);
        free_row_array(rows, stats.valid_rows);
        free_schema(&schema);
        return 1;
    }

    log_both(out, "Dataset file: %s\n", dataset_path);
    log_both(out, "Valid rows loaded: %zu\n", stats.valid_rows);
    log_both(out, "Skipped lines: %zu\n", stats.skipped_lines);
    log_both(out, "Input size: %zu\n", cfg.input_size);
    log_both(out, "Sequence length: %zu\n", cfg.seq_len);
    if (cfg.num_layers > 1) log_both(out, "Layers: %zu\n", cfg.num_layers);
    log_both(out, "Label column: %d\n", schema.label_col);
    log_both(out, "OpenMP max threads: %d\n", omp_get_max_threads());
    log_both(out, "Partition mode: horizontal (hidden-dimension, AllGather/AllReduce via OpenMP barriers/reduction)\n");

    size_t raw_train_size = (size_t)(0.8 * (double)stats.valid_rows);
    size_t raw_test_size = stats.valid_rows - raw_train_size;
    if (raw_train_size < cfg.seq_len || raw_test_size < cfg.seq_len) {
        log_both(out, "Error: not enough rows in a split to build sequences of length %zu.\n", cfg.seq_len);
        fclose(out);
        free_row_array(rows, stats.valid_rows);
        free_schema(&schema);
        return 1;
    }

    size_t n_train_all = raw_train_size - cfg.seq_len + 1;
    size_t n_test_seq = raw_test_size - cfg.seq_len + 1;

    SequenceSample *train_all = (SequenceSample *)malloc(n_train_all * sizeof(SequenceSample));
    SequenceSample *test = (SequenceSample *)malloc(n_test_seq * sizeof(SequenceSample));
    if (!train_all || !test) {
        log_both(out, "Error: train/test sequence allocation failed.\n");
        fclose(out);
        free(train_all);
        free(test);
        free_row_array(rows, stats.valid_rows);
        free_schema(&schema);
        return 1;
    }

    size_t train_all_size = build_sequences(rows, raw_train_size, train_all, n_train_all, &cfg);
    size_t test_size = build_sequences(rows + raw_train_size, raw_test_size, test, n_test_seq, &cfg);
    free_row_array(rows, stats.valid_rows);
    rows = NULL;

    if (train_all_size == 0 || test_size == 0) {
        log_both(out, "Error: sequence building produced an empty split.\n");
        fclose(out);
        free_sequences(train_all, train_all_size);
        free_sequences(test, test_size);
        free_schema(&schema);
        return 1;
    }

    SequenceSample *train = NULL;
    size_t train_size = balance_binary_sequences(train_all, train_all_size, &train, 1.0f, cfg.shuffle_seed);
    if (!train || train_size == 0) {
        log_both(out, "Error: training-sequence balancing failed.\n");
        fclose(out);
        free_sequences(test, test_size);
        free_schema(&schema);
        return 1;
    }

    if (shuffle_train) {
        shuffle_sequences(train, train_size, cfg.shuffle_seed);
    }

    NormalizationStats norm_stats;
    if (!allocate_stats(&norm_stats, &cfg)) {
        log_both(out, "Error: failed to allocate model objects.\n");
        fclose(out);
        free_sequences(train, train_size);
        free_sequences(test, test_size);
        free_schema(&schema);
        return 1;
    }
    compute_normalization_stats(train, train_size, &cfg, &norm_stats);
    apply_normalization(train, train_size, &cfg, &norm_stats);
    apply_normalization(test, test_size, &cfg, &norm_stats);

    int train_neg = 0, train_pos = 0;
    compute_class_counts(train, train_size, &train_neg, &train_pos);
    real pos_weight = (train_pos > 0) ? (real)train_neg / (real)train_pos : 1.0f;

    log_both(out, "Raw train rows: %zu\n", raw_train_size);
    log_both(out, "Raw test rows: %zu\n", raw_test_size);
    log_both(out, "Train sequences (pre-balance): %zu\n", train_all_size);
    log_both(out, "Train sequences: %zu\n", train_size);
    log_both(out, "Test sequences: %zu\n\n", test_size);

    SequentialRNN rnn;
    RNNGradients grads;
    HorizontalScratch scratch;
    if (!allocate_rnn(&rnn, &cfg) || !allocate_gradients(&grads, &cfg) || !allocate_horizontal_scratch(&scratch, &cfg)) {
        log_both(out, "Error: failed to allocate model objects.\n");
        fclose(out);
        free_rnn(&rnn);
        free_gradients(&grads);
        free_horizontal_scratch(&scratch);
        free_stats(&norm_stats);
        free_sequences(train, train_size);
        free_sequences(test, test_size);
        free_schema(&schema);
        return 1;
    }
    rnn_init(&rnn, &cfg, cfg.shuffle_seed);

    omp_set_dynamic(0);

    /* Shared accumulators declared before parallel region, so shared, not
     * per-thread stack copies. Every thread computes same epoch/batch/sample
     * indices from read-only values, so all reach each omp single and omp for in
     * step. Only accumulator updates need guarding. */
    int train_correct = 0;
    double train_loss = 0.0;
    int tp = 0, tn = 0, fp_count = 0, fn = 0, test_correct = 0;
    double test_loss = 0.0;
    double train_acc = 0.0, avg_train_loss = 0.0;
    double test_acc = 0.0, avg_test_loss = 0.0, precision = 0.0, recall = 0.0, f1 = 0.0;

    #pragma omp parallel
    {
        for (int epoch = 0; epoch < cfg.num_epochs; epoch++) {
            #pragma omp single
            {
                train_correct = 0;
                train_loss = 0.0;
            }

            for (size_t batch_start = 0; batch_start < train_size; batch_start += cfg.batch_size) {
                size_t batch_end = batch_start + cfg.batch_size;
                if (batch_end > train_size) batch_end = train_size;
                int batch_count = (int)(batch_end - batch_start);

                #pragma omp single
                rnn_zero_gradients(&grads, &cfg);

                for (size_t i = batch_start; i < batch_end; i++) {
                    RNNPrediction pred = rnn_accumulate_sequence_gradients(&rnn, &train[i], &cfg, pos_weight, &grads, &scratch);
                    #pragma omp single
                    {
                        train_loss += pred.loss;
                        if (pred.pred_label == train[i].label) train_correct++;
                    }
                }

                #pragma omp single
                rnn_apply_gradients(&rnn, &grads, &cfg, batch_count);
            }

            #pragma omp single
            {
                train_acc = 100.0 * safe_div((double)train_correct, (double)train_size);
                avg_train_loss = safe_div(train_loss, (double)train_size);

                tp = 0; tn = 0; fp_count = 0; fn = 0; test_correct = 0;
                test_loss = 0.0;
            }

            for (size_t i = 0; i < test_size; i++) {
                const SequenceSample *sample = &test[i];
                RNNPrediction pred = rnn_forward_sequence(&rnn, sample, &cfg, &scratch);

                #pragma omp single
                {
                    test_loss += pred.loss;
                    if (pred.pred_label == sample->label) test_correct++;

                    if (sample->label == 1 && pred.pred_label == 1) tp++;
                    else if (sample->label == 0 && pred.pred_label == 0) tn++;
                    else if (sample->label == 0 && pred.pred_label == 1) fp_count++;
                    else if (sample->label == 1 && pred.pred_label == 0) fn++;
                }
            }

            #pragma omp single
            {
                test_acc = 100.0 * safe_div((double)test_correct, (double)test_size);
                avg_test_loss = safe_div(test_loss, (double)test_size);
                precision = safe_div((double)tp, (double)(tp + fp_count));
                recall = safe_div((double)tp, (double)(tp + fn));
                f1 = safe_div(2.0 * precision * recall, precision + recall);
                double elapsed_wall = difftime(time(NULL), wall_start);

                log_both(out, "Epoch %d TRAIN | Loss %.6f | Acc %.2f%%\n", epoch + 1, avg_train_loss, train_acc);
                log_both(out, "Epoch %d TEST  | Loss %.6f | Acc %.2f%% | Precision %.4f | Recall %.4f | F1 %.4f | Elapsed %.2f sec\n",
                         epoch + 1, avg_test_loss, test_acc, precision, recall, f1, elapsed_wall);
                log_both(out, "Confusion Matrix: TN=%d FP=%d FN=%d TP=%d\n\n", tn, fp_count, fn, tp);
            }
        }
    }

    log_both(out, "Program finished.\n");

    fclose(out);
    free_gradients(&grads);
    free_rnn(&rnn);
    free_horizontal_scratch(&scratch);
    free_stats(&norm_stats);
    free_sequences(train, train_size);
    free_sequences(test, test_size);
    free_schema(&schema);
    return 0;
}
