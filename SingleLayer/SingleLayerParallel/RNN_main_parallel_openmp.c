#include <omp.h>
#include "RNN_seq_parallel_openmp.h"

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
        "  --layers N            (L, number of stacked RNN layers; default 1)\n"
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
                    "Error: %s is the single-layer batch data-parallel program and supports --layers 1 only "
                    "(got %ld).\n"
                    "       For L > 1 use MultilayerParallel/rnn_parallel_multilayer.\n",
                    argv[0], v);
                return 1;
            }
            cfg.num_layers = 1;
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
    if (!allocate_rnn(&rnn, &cfg) || !allocate_gradients(&grads, &cfg)) {
        log_both(out, "Error: failed to allocate model objects.\n");
        fclose(out);
        free_rnn(&rnn);
        free_gradients(&grads);
        free_stats(&norm_stats);
        free_sequences(train, train_size);
        free_sequences(test, test_size);
        free_schema(&schema);
        return 1;
    }
    rnn_init(&rnn, &cfg, cfg.shuffle_seed);

    /* Deterministic data-parallel training. Fixed team size keeps merge order
     * stable; each thread has own gradient/loss/count buffers, so no shared writes
     * in parallel region, no race, no critical section. Buffers merged in
     * ascending thread order after region. Result independent of thread count. */
    omp_set_dynamic(0);
    int max_threads = omp_get_max_threads();

    RNNGradients *thread_grads = (RNNGradients *)malloc((size_t)max_threads * sizeof(RNNGradients));
    double *thr_loss = (double *)malloc((size_t)max_threads * sizeof(double));
    int *thr_correct = (int *)malloc((size_t)max_threads * sizeof(int));
    int *thr_tp = (int *)malloc((size_t)max_threads * sizeof(int));
    int *thr_tn = (int *)malloc((size_t)max_threads * sizeof(int));
    int *thr_fp = (int *)malloc((size_t)max_threads * sizeof(int));
    int *thr_fn = (int *)malloc((size_t)max_threads * sizeof(int));
    if (!thread_grads || !thr_loss || !thr_correct || !thr_tp || !thr_tn || !thr_fp || !thr_fn) {
        log_both(out, "Error: per-thread buffer allocation failed.\n");
        fclose(out);
        free_rnn(&rnn);
        free_gradients(&grads);
        free_stats(&norm_stats);
        free_sequences(train, train_size);
        free_sequences(test, test_size);
        free_schema(&schema);
        free(thread_grads); free(thr_loss); free(thr_correct);
        free(thr_tp); free(thr_tn); free(thr_fp); free(thr_fn);
        return 1;
    }
    for (int t = 0; t < max_threads; t++) {
        if (!allocate_gradients(&thread_grads[t], &cfg)) {
            log_both(out, "Error: per-thread buffer allocation failed.\n");
            fclose(out);
            free_rnn(&rnn);
            free_gradients(&grads);
            free_stats(&norm_stats);
            free_sequences(train, train_size);
            free_sequences(test, test_size);
            free_schema(&schema);
            for (int u = 0; u < t; u++) free_gradients(&thread_grads[u]);
            free(thread_grads); free(thr_loss); free(thr_correct);
            free(thr_tp); free(thr_tn); free(thr_fp); free(thr_fn);
            return 1;
        }
    }

    for (int epoch = 0; epoch < cfg.num_epochs; epoch++) {
        int train_correct = 0;
        double train_loss = 0.0;

        for (size_t batch_start = 0; batch_start < train_size; batch_start += cfg.batch_size) {
            size_t batch_end = batch_start + cfg.batch_size;
            if (batch_end > train_size) batch_end = train_size;
            int batch_count = (int)(batch_end - batch_start);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                RNNGradients *lg = &thread_grads[tid];
                rnn_zero_gradients(lg, &cfg);
                double lloss = 0.0;
                int lcorrect = 0;

                #pragma omp for schedule(static)
                for (size_t i = batch_start; i < batch_end; i++) {
                    RNNPrediction pred = rnn_accumulate_sequence_gradients(&rnn, &train[i], &cfg, pos_weight, lg);
                    lloss += pred.loss;
                    if (pred.pred_label == train[i].label) {
                        lcorrect++;
                    }
                }

                thr_loss[tid] = lloss;
                thr_correct[tid] = lcorrect;
            }

            // Deterministic merge, fixed thread order.
            rnn_zero_gradients(&grads, &cfg);
            for (int t = 0; t < max_threads; t++) {
                rnn_add_gradients(&grads, &thread_grads[t], &cfg);
                train_loss += thr_loss[t];
                train_correct += thr_correct[t];
            }

            rnn_apply_gradients(&rnn, &grads, &cfg, batch_count);
        }

        double train_acc = 100.0 * safe_div((double)train_correct, (double)train_size);
        double avg_train_loss = safe_div(train_loss, (double)train_size);

        int tp = 0, tn = 0, fp_count = 0, fn = 0, test_correct = 0;
        double test_loss = 0.0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int ltp = 0, ltn = 0, lfp = 0, lfn = 0, lcorrect = 0;
            double lloss = 0.0;

            #pragma omp for schedule(static)
            for (size_t i = 0; i < test_size; i++) {
                const SequenceSample *sample = &test[i];
                RNNPrediction pred = rnn_forward_sequence(&rnn, sample, &cfg);

                lloss += pred.loss;
                if (pred.pred_label == sample->label) {
                    lcorrect++;
                }

                if (sample->label == 1 && pred.pred_label == 1) ltp++;
                else if (sample->label == 0 && pred.pred_label == 0) ltn++;
                else if (sample->label == 0 && pred.pred_label == 1) lfp++;
                else if (sample->label == 1 && pred.pred_label == 0) lfn++;
            }

            thr_tp[tid] = ltp;
            thr_tn[tid] = ltn;
            thr_fp[tid] = lfp;
            thr_fn[tid] = lfn;
            thr_correct[tid] = lcorrect;
            thr_loss[tid] = lloss;
        }

        // Deterministic fold, fixed thread order.
        for (int t = 0; t < max_threads; t++) {
            tp += thr_tp[t];
            tn += thr_tn[t];
            fp_count += thr_fp[t];
            fn += thr_fn[t];
            test_correct += thr_correct[t];
            test_loss += thr_loss[t];
        }

        double test_acc = 100.0 * safe_div((double)test_correct, (double)test_size);
        double avg_test_loss = safe_div(test_loss, (double)test_size);
        double precision = safe_div((double)tp, (double)(tp + fp_count));
        double recall = safe_div((double)tp, (double)(tp + fn));
        double f1 = safe_div(2.0 * precision * recall, precision + recall);
        double elapsed_wall = difftime(time(NULL), wall_start);

        log_both(out, "Epoch %d TRAIN | Loss %.6f | Acc %.2f%%\n", epoch + 1, avg_train_loss, train_acc);
        log_both(out, "Epoch %d TEST  | Loss %.6f | Acc %.2f%% | Precision %.4f | Recall %.4f | F1 %.4f | Elapsed %.2f sec\n",
                 epoch + 1, avg_test_loss, test_acc, precision, recall, f1, elapsed_wall);
        log_both(out, "Confusion Matrix: TN=%d FP=%d FN=%d TP=%d\n\n", tn, fp_count, fn, tp);
    }

    log_both(out, "Program finished.\n");

    fclose(out);
    for (int t = 0; t < max_threads; t++) free_gradients(&thread_grads[t]);
    free(thread_grads);
    free(thr_loss);
    free(thr_correct);
    free(thr_tp);
    free(thr_tn);
    free(thr_fp);
    free(thr_fn);
    free_gradients(&grads);
    free_rnn(&rnn);
    free_stats(&norm_stats);
    free_sequences(train, train_size);
    free_sequences(test, test_size);
    free_schema(&schema);
    return 0;
}
