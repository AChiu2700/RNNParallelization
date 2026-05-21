#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#include "KDD_sequential_rnn.h"

#define MAX_LINE 4096
#define MAX_ROWS 300000
#define MAX_SEQUENCES MAX_ROWS

#ifndef DATASET_PATH

#define DATASET_PATH "/mnt/c/Users/achiu/Documents/Thesis/KDD_Dataset/KDD_processed_dataset.csv"
#endif

#ifndef OUTPUT_PATH
#define OUTPUT_PATH "KDD_rnn_output.txt"
#endif

typedef struct {
    real features[INPUT_SIZE];
    int label;
} RawSample;

static double safe_div(double num, double den) {
    return (den == 0.0) ? 0.0 : (num / den);
}

static void log_both(FILE *out, const char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fflush(out);
}

static void shuffle_sequences(SequenceSample *data, int n, unsigned int seed) {
    srand(seed);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        SequenceSample tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

static int build_sequences(
    const RawSample *rows,
    int n_rows,
    SequenceSample *seqs,
    int max_seqs
) {
    int count = 0;

    for (int start = 0; start + SEQ_LEN <= n_rows && count < max_seqs; start++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            for (int d = 0; d < INPUT_SIZE; d++) {
                seqs[count].x[t][d] = rows[start + t].features[d];
            }
        }
        seqs[count].label = rows[start + SEQ_LEN - 1].label;
        count++;
    }

    return count;
}

static void compute_class_counts(
    const SequenceSample *data,
    int n,
    int *neg_count,
    int *pos_count
) {
    *neg_count = 0;
    *pos_count = 0;

    for (int i = 0; i < n; i++) {
        if (data[i].label == 1) {
            (*pos_count)++;
        } else {
            (*neg_count)++;
        }
    }
}

int main(int argc, char **argv) {
    time_t wall_start = time(NULL);
    const char *dataset_path = (argc > 1) ? argv[1] : DATASET_PATH;
    const char *output_path = (argc > 2) ? argv[2] : OUTPUT_PATH;

    FILE *out = fopen(output_path, "w");
    if (!out) {
        printf("Error: could not create output file: %s\n", output_path);
        return 1;
    }

    FILE *fp = fopen(dataset_path, "r");
    if (!fp) {
        log_both(out, "Error: could not open dataset file: %s\n", dataset_path);
        log_both(out, "Usage: %s [dataset.csv] [output.txt]\n", argv[0]);
        fclose(out);
        return 1;
    }

    RawSample *rows = (RawSample *)malloc(MAX_ROWS * sizeof(RawSample));
    SequenceSample *data = (SequenceSample *)malloc(MAX_SEQUENCES * sizeof(SequenceSample));
    if (!rows || !data) {
        log_both(out, "Error: memory allocation failed.\n");
        fclose(fp);
        fclose(out);
        free(rows);
        free(data);
        return 1;
    }

    char line[MAX_LINE];
    int total_rows = 0;
    int total_lines = 0;
    int skipped_lines = 0;

    while (fgets(line, sizeof(line), fp) != NULL && total_rows < MAX_ROWS) {
        total_lines++;

        char copy[MAX_LINE];
        strncpy(copy, line, MAX_LINE - 1);
        copy[MAX_LINE - 1] = '\0';

        if (!rnn_parse_csv_line(copy, rows[total_rows].features, &rows[total_rows].label)) {
            skipped_lines++;
            if (skipped_lines <= 5) {
                log_both(out, "Skipped line %d: %s", total_lines, line);
            }
            continue;
        }

        total_rows++;
    }
    fclose(fp);

    int total_sequences = build_sequences(rows, total_rows, data, MAX_SEQUENCES);

    log_both(out, "Total raw lines read: %d\n", total_lines);
    log_both(out, "Valid rows loaded: %d\n", total_rows);
    log_both(out, "Built sequences: %d\n", total_sequences);
    log_both(out, "Skipped lines: %d\n", skipped_lines);
    log_both(out, "Output log file: %s\n", output_path);

    if (total_sequences <= 1) {
        log_both(out, "Error: not enough rows to build sequences of length %d.\n", SEQ_LEN);
        fclose(out);
        free(rows);
        free(data);
        return 1;
    }

    //shuffle_sequences(data, total_sequences, SHUFFLE_SEED);

    int train_size = (int)(0.8 * total_sequences);
    int test_size = total_sequences - train_size;
    if (train_size <= 0 || test_size <= 0) {
        log_both(out, "Error: invalid dataset split. train_size=%d test_size=%d\n", train_size, test_size);
        fclose(out);
        free(rows);
        free(data);
        return 1;
    }

    NormalizationStats stats;
    compute_normalization_stats(data, train_size, &stats);
    apply_normalization(data, train_size, &stats);
    apply_normalization(data + train_size, test_size, &stats);

    int train_neg = 0;
    int train_pos = 0;
    compute_class_counts(data, train_size, &train_neg, &train_pos);

    real pos_weight = 1.0f;
    if (train_pos > 0) {
        pos_weight = (real)train_neg / (real)train_pos;
    }

    log_both(out, "Train sequences: %d\n", train_size);
    log_both(out, "Test sequences: %d\n", test_size);
    log_both(out, "Sequence length: %d\n", SEQ_LEN);
    log_both(out, "Batch size: %d\n", BATCH_SIZE);
    log_both(out, "Input size: %d\n", INPUT_SIZE);
    log_both(out, "Train class counts -> neg: %d pos: %d | pos_weight: %.4f\n\n",
             train_neg, train_pos, pos_weight);

    SequentialRNN rnn;
    rnn_init(&rnn, SHUFFLE_SEED);

    clock_t global_start = clock();

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        int train_correct = 0;
        double train_loss = 0.0;

        for (int batch_start = 0; batch_start < train_size; batch_start += BATCH_SIZE) {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > train_size) {
                batch_end = train_size;
            }
            int batch_count = batch_end - batch_start;

            RNNGradients grads;
            rnn_zero_gradients(&grads);

            for (int i = batch_start; i < batch_end; i++) {
                RNNPrediction pred = rnn_accumulate_sequence_gradients(&rnn, &data[i], pos_weight, &grads);
                train_loss += pred.loss;
                if (pred.pred_label == data[i].label) {
                    train_correct++;
                }
            }

            rnn_apply_gradients(&rnn, &grads, LEARNING_RATE, batch_count);
        }

        double train_acc = 100.0 * safe_div((double)train_correct, (double)train_size);
        double avg_train_loss = safe_div(train_loss, (double)train_size);

        int tp = 0;
        int tn = 0;
        int fp_count = 0;
        int fn = 0;
        int test_correct = 0;
        double test_loss = 0.0;

        for (int i = 0; i < test_size; i++) {
            const SequenceSample *sample = &data[train_size + i];
            RNNPrediction pred = rnn_forward_sequence(&rnn, sample);

            test_loss += pred.loss;
            if (pred.pred_label == sample->label) {
                test_correct++;
            }

            if (sample->label == 1 && pred.pred_label == 1) {
                tp++;
            } else if (sample->label == 0 && pred.pred_label == 0) {
                tn++;
            } else if (sample->label == 0 && pred.pred_label == 1) {
                fp_count++;
            } else if (sample->label == 1 && pred.pred_label == 0) {
                fn++;
            }
        }

        double test_acc = 100.0 * safe_div((double)test_correct, (double)test_size);
        double avg_test_loss = safe_div(test_loss, (double)test_size);
        double precision = safe_div((double)tp, (double)(tp + fp_count));
        double recall = safe_div((double)tp, (double)(tp + fn));
        double f1 = safe_div(2.0 * precision * recall, precision + recall);
        double elapsed_wall = difftime(time(NULL), wall_start);

        log_both(out,
                 "Epoch %d TRAIN complete | Samples %d | Avg Loss %.6f | Accuracy %.2f%%\n",
                 epoch + 1, train_size, avg_train_loss, train_acc);

        log_both(out,
                 "Epoch %d TEST complete | Samples %d | Avg Loss %.6f | Accuracy %.2f%% | Precision %.4f | Recall %.4f | F1 %.4f | Elapsed %.2f sec\n",
                 epoch + 1, test_size, avg_test_loss, test_acc, precision, recall, f1, elapsed_wall);

        log_both(out, "Confusion Matrix (TEST):\n");
        log_both(out, "         Pred 0   Pred 1\n");
        log_both(out, "True 0 %8d %8d\n", tn, fp_count);
        log_both(out, "True 1 %8d %8d\n\n", fn, tp);
    }

    double total_cpu_runtime = (double)(clock() - global_start) / CLOCKS_PER_SEC;
    double total_elapsed_wall = difftime(time(NULL), wall_start);

    log_both(out, "=====================================\n");
    log_both(out, "Program finished.\n");
    log_both(out, "Total CPU runtime: %.6f seconds\n", total_cpu_runtime);
    log_both(out, "Total elapsed wall time: %.2f seconds (%.2f minutes)\n",
             total_elapsed_wall, total_elapsed_wall / 60.0);
    log_both(out, "=====================================\n");

    fclose(out);
    free(rows);
    free(data);
    return 0;
}