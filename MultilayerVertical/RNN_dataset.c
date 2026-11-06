#include "RNN_seq_verticle_multilayer.h"

static char *dup_string(const char *s) {
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, s, len + 1);
    return copy;
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

static int string_ieq(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

/*
 * Strict numeric parse: trimmed strtod, fails on trailing garbage, rejects
 * inf/-inf/NaN via |v| > DBL_MAX (NaN compares unordered, so caught too).
 * Skips CIC rows with literal "inf" features.
 */
static int parse_double_strict(const char *text, double *out) {
    if (!text) return 0;
    const char *p = text;
    while (isspace((unsigned char)*p)) p++;
    if (*p == '\0') return 0;

    char *endptr;
    double v = strtod(p, &endptr);
    if (endptr == p) return 0;

    while (isspace((unsigned char)*endptr)) endptr++;
    if (*endptr != '\0') return 0;

    if (fabs(v) > DBL_MAX) return 0;

    *out = v;
    return 1;
}

int parse_int_list(const char *text, int **values_out, size_t *count_out) {
    if (!text || *text == '\0') {
        *values_out = NULL;
        *count_out = 0;
        return 1;
    }

    char *copy = dup_string(text);
    if (!copy) return 0;

    int *arr = NULL;
    size_t n = 0, cap = 0;

    char *tok = strtok(copy, ",");
    while (tok) {
        trim_whitespace(tok);
        if (*tok != '\0') {
            char *endptr;
            long v = strtol(tok, &endptr, 10);
            if (endptr == tok || *endptr != '\0' || v < 0 || v >= 0x80000000L) {
                free(copy);
                free(arr);
                return 0;
            }
            if (n == cap) {
                cap = cap ? cap * 2 : 8;
                int *tmp = (int *)realloc(arr, cap * sizeof(int));
                if (!tmp) {
                    free(copy);
                    free(arr);
                    return 0;
                }
                arr = tmp;
            }
            arr[n++] = (int)v;
        }
        tok = strtok(NULL, ",");
    }

    free(copy);
    *values_out = arr;
    *count_out = n;
    return 1;
}

int split_csv_line(char *line, char delimiter, char ***fields_out, int *count_out) {
    size_t cap = 16, n = 0;
    char **fields = (char **)malloc(cap * sizeof(char *));
    if (!fields) return 0;

    char *field_start = line;
    int in_quotes = 0;
    char *p = line;

    for (;;) {
        char c = *p;

        if (c == '"') {
            in_quotes = !in_quotes;
            p++;
            continue;
        }

        int is_end = (c == '\0') || (!in_quotes && (c == '\r' || c == '\n'));
        int is_delim = (!in_quotes && c == delimiter);

        if (is_delim || is_end) {
            *p = '\0';
            trim_whitespace(field_start);

            if (n == cap) {
                cap *= 2;
                char **tmp = (char **)realloc(fields, cap * sizeof(char *));
                if (!tmp) {
                    free(fields);
                    return 0;
                }
                fields = tmp;
            }
            fields[n++] = field_start;

            if (is_end) {
                break;
            }

            p++;
            field_start = p;
            continue;
        }

        p++;
    }

    char **shrunk = (char **)realloc(fields, (n > 0 ? n : 1) * sizeof(char *));
    if (shrunk) fields = shrunk;

    *fields_out = fields;
    *count_out = (int)n;
    return 1;
}

int detect_header_row(char *line, char delimiter) {
    char *copy = dup_string(line);
    if (!copy) return 0;

    char **fields = NULL;
    int count = 0;
    if (!split_csv_line(copy, delimiter, &fields, &count)) {
        free(copy);
        return 0;
    }

    int non_numeric = 0;
    for (int i = 0; i < count; i++) {
        double tmp;
        if (!parse_double_strict(fields[i], &tmp)) non_numeric++;
    }

    int result = (non_numeric > 1) ? 1 : 0;
    free(fields);
    free(copy);
    return result;
}

int infer_label_column(char **fields, int count) {
    static const char *candidates[] = {"label", "class", "target", "y"};
    for (int i = 0; i < count; i++) {
        for (size_t c = 0; c < 4; c++) {
            if (string_ieq(fields[i], candidates[c])) return i;
        }
    }
    return count - 1;
}

int infer_feature_columns(
    char **fields,
    int count,
    int label_col,
    const int *drop_cols,
    size_t drop_count,
    int **feature_cols_out,
    size_t *feature_count_out
) {
    (void)fields;

    int *out = (int *)malloc((size_t)count * sizeof(int));
    if (!out) return 0;

    size_t n = 0;
    for (int i = 0; i < count; i++) {
        if (i == label_col) continue;

        int dropped = 0;
        for (size_t d = 0; d < drop_count; d++) {
            if (drop_cols[d] == i) {
                dropped = 1;
                break;
            }
        }
        if (dropped) continue;

        out[n++] = i;
    }

    if (n == 0) {
        free(out);
        return 0;
    }

    *feature_cols_out = out;
    *feature_count_out = n;
    return 1;
}

/*
 * Unknown text label defaults to positive (attack) and succeeds, so row is
 * kept, not skipped. Unexpected or misspelled labels fall to attack class.
 */
int encode_label(const char *text, const DatasetSchema *schema, int *label_out) {
    double value;
    if (parse_double_strict(text, &value)) {
        if (schema->positive_label_numeric) {
            *label_out = (fabs(value - schema->positive_label_value) < 1e-12) ? 1 : 0;
        } else {
            *label_out = (value > 0.5) ? 1 : 0;
        }
        return 1;
    }

    if (!schema->allow_text_labels) return 0;

    if (schema->positive_label_text[0] != '\0') {
        *label_out = string_ieq(text, schema->positive_label_text) ? 1 : 0;
        return 1;
    }

    static const char *positive_candidates[] = {"1", "true", "attack", "anomaly", "malicious"};
    static const char *negative_candidates[] = {"0", "false", "benign", "normal"};

    for (size_t i = 0; i < 5; i++) {
        if (string_ieq(text, positive_candidates[i])) {
            *label_out = 1;
            return 1;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        if (string_ieq(text, negative_candidates[i])) {
            *label_out = 0;
            return 1;
        }
    }

    *label_out = 1;
    return 1;
}

int parse_row_with_schema(char *line, const DatasetSchema *schema, real *features, int *label) {
    char **fields = NULL;
    int count = 0;
    if (!split_csv_line(line, schema->delimiter, &fields, &count)) return 0;

    int ok = 0;
    if (schema->label_col >= 0 && schema->label_col < count) {
        size_t k = 0;
        for (; k < schema->feature_count; k++) {
            int col = schema->feature_cols[k];
            if (col < 0 || col >= count) break;
            double v;
            if (!parse_double_strict(fields[col], &v)) break;
            features[k] = (real)v;
        }
        if (k == schema->feature_count) {
            ok = encode_label(fields[schema->label_col], schema, label);
        }
    }

    free(fields);
    return ok;
}

void log_both(FILE *out, const char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fflush(out);
}

int load_numeric_dataset(
    const char *dataset_path,
    RNNConfig *cfg,
    DatasetSchema *schema,
    size_t max_rows,
    FILE *log,
    RawSample **rows_out,
    DatasetStats *stats_out
) {
    if (!dataset_path || !cfg || !schema || !rows_out || !stats_out) return 0;

    FILE *fp = fopen(dataset_path, "r");
    if (!fp) {
        if (log) log_both(log, "Error: could not open dataset file: %s\n", dataset_path);
        return 0;
    }

    char line[MAX_LINE];
    if (!fgets(line, sizeof(line), fp)) {
        if (log) log_both(log, "Error: not enough raw rows to build sequences.\n");
        fclose(fp);
        return 0;
    }

    if (schema->has_header < 0) {
        char *copy = dup_string(line);
        schema->has_header = detect_header_row(copy, schema->delimiter);
        free(copy);
    }

    int have_pending_line = 0;
    char pending[MAX_LINE];

    {
        char *copy = dup_string(line);
        char **fields = NULL;
        int count = 0;
        split_csv_line(copy, schema->delimiter, &fields, &count);

        if (schema->has_header) {
            if (schema->infer_label_col) schema->label_col = infer_label_column(fields, count);
        } else {
            if (schema->infer_label_col) schema->label_col = count - 1;
        }

        if (!schema->feature_cols) {
            if (!infer_feature_columns(fields, count, schema->label_col,
                                        schema->drop_cols, schema->drop_count,
                                        &schema->feature_cols, &schema->feature_count)) {
                free(fields);
                free(copy);
                fclose(fp);
                if (log) log_both(log, "Error: failed to infer schema from dataset.\n");
                return 0;
            }
        }

        free(fields);
        free(copy);

        if (!schema->has_header) {
            strncpy(pending, line, MAX_LINE - 1);
            pending[MAX_LINE - 1] = '\0';
            have_pending_line = 1;
        }
    }

    cfg->input_size = schema->feature_count;
    if (cfg->input_size == 0) {
        fclose(fp);
        return 0;
    }

    RawSample *rows = (RawSample *)calloc(max_rows, sizeof(RawSample));
    if (!rows) {
        if (log) log_both(log, "Error: memory allocation failed for raw rows.\n");
        fclose(fp);
        return 0;
    }

    size_t row_count = 0;
    size_t total_lines = schema->has_header ? 1 : 0;
    size_t skipped = 0;

    while (row_count < max_rows) {
        char *cur;
        if (have_pending_line) {
            cur = pending;
            have_pending_line = 0;
        } else {
            if (!fgets(line, sizeof(line), fp)) break;
            cur = line;
        }
        total_lines++;

        real *feat = (real *)malloc(cfg->input_size * sizeof(real));
        if (!feat) {
            if (log) log_both(log, "Error: feature allocation failed at row %zu.\n", row_count);
            break;
        }

        int label;
        if (!parse_row_with_schema(cur, schema, feat, &label)) {
            free(feat);
            skipped++;
            if (skipped <= 5 && log) log_both(log, "Skipped line %zu: %s", total_lines, cur);
            continue;
        }

        rows[row_count].features = feat;
        rows[row_count].label = label;
        row_count++;
    }

    fclose(fp);

    *rows_out = rows;
    stats_out->total_lines = total_lines;
    stats_out->valid_rows = row_count;
    stats_out->skipped_lines = skipped;
    stats_out->total_sequences = 0;

    return row_count != 0;
}
