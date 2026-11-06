/* gradcheck.c -- finite-difference check of layer-indexed BPTT.
 *
 * Backward pass uses two-source dh in rnn_accumulate_sequence_gradients():
 *
 *     dh[l][t] = dh_next[l]   (recurrent, from timestep t+1, same layer)
 *              + dh_up[l]     (vertical,  from layer l+1, same timestep)
 *
 * For every weight w computes central difference
 *
 *     dL/dw ~= ( L(w + eps) - L(w - eps) ) / 2 eps
 *
 * and compares to analytic gradient from backward pass. L(.) is same weighted
 * loss backward pass differentiates: rnn_forward_sequence() returns unweighted
 * loss, class_weight is constant scale, so class_weight * loss is exact.
 *
 * Build in double (-DRNN_USE_DOUBLE): central difference cancels most digits,
 * float leaves ~2 usable.
 *
 *   gcc -O2 -std=gnu99 -DRNN_USE_DOUBLE gradcheck.c RNN_seq_multilayer.c RNN_dataset.c \
 *       -o gradcheck -lm
 *   ./gradcheck            # exits 0 on pass, 1 on any failure
 *
 * Results per weight block, Wxh split per layer: Wxh[l>0] is fed by dh_down,
 * so it isolates vertical term.
 */

#include "RNN_seq_multilayer.h"

/* Accept if |analytic - numeric| <= FD_ATOL + FD_RTOL * |analytic|,
 * not pure relative test.
 *
 * Central difference has absolute noise floor: roundoff in two loss
 * evaluations ~machine_eps * |L|, divided by 2*eps, so in double with
 * eps=1e-5 numeric derivative good to ~1e-10 absolute. Deep in stack true
 * gradients fall to 1e-8..1e-9 (normal decay in vanilla RNN), where pure
 * relative test compares noise with noise and fails on correct code.
 *
 * FD_EPS 1e-5 near optimum for double: truncation error grows as eps^2,
 * roundoff as 1/eps, crossing near (machine_eps)^(1/3) ~ 6e-6. */
#ifdef RNN_USE_DOUBLE
#define FD_EPS  1e-5
#define FD_RTOL 1e-5
#define FD_ATOL 1e-9
#define PRECISION_NAME "double"
#else
// float32 fallback: check runs but only good to few percent.
// Pass here means no gross error only.
#define FD_EPS  1e-2
#define FD_RTOL 5e-2
#define FD_ATOL 1e-4
#define PRECISION_NAME "float (LOOSE -- rebuild with -DRNN_USE_DOUBLE)"
#endif

#define MAX_BLOCKS 64

typedef struct {
    char name[32];
    real *w;  // weights
    const real *g;  // analytic gradients, same indexing
    size_t n;
} ParamBlock;

// uniform in [-1, 1]; real inputs are normalized features
static double frand(void) {
    return 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
}

static double weighted_loss(SequentialRNN *rnn, const SequenceSample *sample,
                            const RNNConfig *cfg, double class_weight) {
    RNNPrediction pred = rnn_forward_sequence(rnn, sample, cfg);
    return class_weight * pred.loss;
}

/* How far outside tolerance pair is: <= 1.0 passes, > 1.0 fails. Single ratio
 * keeps tiny-gradient entries comparable with large ones. */
static double tol_ratio(double a, double n) {
    return fabs(a - n) / (FD_ATOL + FD_RTOL * fabs(a));
}

static int check_config(size_t I, size_t H, size_t L, size_t S,
                        int label, double pos_weight, unsigned int seed,
                        int verbose) {
    RNNConfig cfg;
    init_default_config(&cfg);
    cfg.input_size = I;
    cfg.hidden_size = H;
    cfg.output_size = 1;
    cfg.num_layers = L;
    cfg.seq_len = S;

    SequentialRNN rnn;
    RNNGradients grads;
    // both return TRUE on success, not 0
    if (!allocate_rnn(&rnn, &cfg) || !allocate_gradients(&grads, &cfg)) {
        fprintf(stderr, "allocation failed\n");
        return -1;
    }
    rnn_init(&rnn, &cfg, seed);

    /* Move off training init.
     *
     * rnn_init draws weights uniform in [-0.1, 0.1], biases 0. With weights that
     * small hidden activations stay near zero, signal decays through timestep and
     * layer axes, and two layers down true gradients are ~1e-9, at finite
     * difference noise floor: check would compare noise with noise.
     *
     * Larger weights and nonzero biases put tanh in responsive band, gradients
     * O(1e-2..1e-4), so wrong term shows as wrong number, not noise. Derivative
     * correctness does not depend on weight values. */
    for (size_t i = 0; i < rnn_wxh_total(&cfg); i++) rnn.Wxh[i] = (real)(0.8 * frand());
    for (size_t i = 0; i < rnn_whh_total(&cfg); i++) rnn.Whh[i] = (real)(0.8 * frand());
    for (size_t i = 0; i < cfg.output_size * H; i++)  rnn.Why[i] = (real)(0.8 * frand());
    for (size_t i = 0; i < rnn_bh_total(&cfg); i++)   rnn.bh[i]  = (real)(0.2 * frand());
    for (size_t i = 0; i < cfg.output_size; i++)      rnn.by[i]  = (real)(0.2 * frand());

    SequenceSample sample;
    sample.label = label;
    sample.x = (real *)malloc(S * I * sizeof(real));
    if (!sample.x) { fprintf(stderr, "allocation failed\n"); return -1; }
    for (size_t i = 0; i < S * I; i++) sample.x[i] = (real)frand();

    double class_weight = (label == 1) ? pos_weight : 1.0;

    /* Analytic gradients. rnn_accumulate_sequence_gradients does not clip
     * (clipping in rnn_apply_gradients), so this is raw gradient, comparable to
     * finite difference. */
    rnn_zero_gradients(&grads, &cfg);
    RNNPrediction pred =
        rnn_accumulate_sequence_gradients(&rnn, &sample, &cfg, (real)pos_weight, &grads);

    /* Loss clips score into [epsilon, 1-epsilon], backward treats clip as
     * straight-through. If clip active, two disagree by construction, so refuse
     * instead of reporting false failure. */
    if (pred.pred_score <= cfg.epsilon || pred.pred_score >= 1.0 - cfg.epsilon) {
        fprintf(stderr, "  score %.9g is in the clipped region -- check invalid\n",
                (double)pred.pred_score);
        free(sample.x); free_rnn(&rnn); free_gradients(&grads);
        return -1;
    }

    // Blocks, Wxh/Whh/bh split per layer.
    ParamBlock blocks[MAX_BLOCKS];
    size_t nb = 0;
    for (size_t l = 0; l < L; l++) {
        size_t in_size = rnn_layer_in_size(&cfg, l);
        snprintf(blocks[nb].name, sizeof(blocks[nb].name), "Wxh[%zu] (%zux%zu)", l, H, in_size);
        blocks[nb].w = rnn.Wxh + rnn_wxh_off(&cfg, l);
        blocks[nb].g = grads.dWxh + rnn_wxh_off(&cfg, l);
        blocks[nb].n = H * in_size;
        nb++;
    }
    for (size_t l = 0; l < L; l++) {
        snprintf(blocks[nb].name, sizeof(blocks[nb].name), "Whh[%zu]", l);
        blocks[nb].w = rnn.Whh + rnn_whh_off(&cfg, l);
        blocks[nb].g = grads.dWhh + rnn_whh_off(&cfg, l);
        blocks[nb].n = H * H;
        nb++;
    }
    for (size_t l = 0; l < L; l++) {
        snprintf(blocks[nb].name, sizeof(blocks[nb].name), "bh[%zu]", l);
        blocks[nb].w = rnn.bh + rnn_bh_off(&cfg, l);
        blocks[nb].g = grads.dbh + rnn_bh_off(&cfg, l);
        blocks[nb].n = H;
        nb++;
    }
    snprintf(blocks[nb].name, sizeof(blocks[nb].name), "Why");
    blocks[nb].w = rnn.Why; blocks[nb].g = grads.dWhy;
    blocks[nb].n = cfg.output_size * H; nb++;
    snprintf(blocks[nb].name, sizeof(blocks[nb].name), "by");
    blocks[nb].w = rnn.by; blocks[nb].g = grads.dby;
    blocks[nb].n = cfg.output_size; nb++;

    double worst_overall = 0.0;
    int failures = 0;

    for (size_t b = 0; b < nb; b++) {
        double worst = 0.0;
        size_t worst_i = 0;
        double worst_a = 0.0, worst_n = 0.0;

        for (size_t i = 0; i < blocks[b].n; i++) {
            real saved = blocks[b].w[i];

            blocks[b].w[i] = (real)((double)saved + FD_EPS);
            double lp = weighted_loss(&rnn, &sample, &cfg, class_weight);
            blocks[b].w[i] = (real)((double)saved - FD_EPS);
            double lm = weighted_loss(&rnn, &sample, &cfg, class_weight);
            blocks[b].w[i] = saved;

            double numeric = (lp - lm) / (2.0 * FD_EPS);
            double analytic = (double)blocks[b].g[i];
            double e = tol_ratio(analytic, numeric);

            if (e > worst) { worst = e; worst_i = i; worst_a = analytic; worst_n = numeric; }
        }

        if (worst > worst_overall) worst_overall = worst;
        int bad = (worst > 1.0);
        if (bad) failures++;

        if (verbose || bad) {
            printf("    %-18s n=%-5zu worst %.3f x tol   analytic %+.9g  numeric %+.9g  [i=%zu]  %s\n",
                   blocks[b].name, blocks[b].n, worst, worst_a, worst_n, worst_i,
                   bad ? "FAIL" : "ok");
        }
    }

    printf("  I=%zu H=%zu L=%zu S=%zu label=%d pos_weight=%.2f seed=%u  "
           "score=%.6f  worst %.3f x tol  %s\n",
           I, H, L, S, label, pos_weight, seed,
           (double)pred.pred_score, worst_overall,
           failures ? "*** FAIL ***" : "PASS");

    free(sample.x);
    free_rnn(&rnn);
    free_gradients(&grads);
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) verbose = 1;
        else {
            fprintf(stderr, "usage: %s [--verbose]\n", argv[0]);
            return 2;
        }
    }

    printf("Finite-difference gradient check -- layer-indexed BPTT\n");
    printf("precision: %s   sizeof(real)=%zu   fd eps=%g\n", PRECISION_NAME, sizeof(real), (double)FD_EPS);
    printf("accept:    |analytic - numeric| <= %g + %g * |analytic|\n\n",
           (double)FD_ATOL, (double)FD_RTOL);

    int fails = 0, errors = 0, total = 0;

    /* Cases cover L=1..4 and both labels (class-weighted loss path). S=1
     * removes recurrent term, so any error there is purely vertical. */
    struct { size_t I, H, L, S; int label; double pw; unsigned seed; } cases[] = {
        { 4, 5, 1, 5, 0, 1.0,  1 },
        { 4, 5, 1, 5, 1, 3.7,  2 },
        { 4, 5, 2, 5, 0, 1.0,  1 },
        { 4, 5, 2, 5, 1, 3.7,  2 },
        { 4, 5, 3, 5, 0, 1.0,  3 },
        { 4, 5, 3, 5, 1, 3.7,  4 },
        { 4, 5, 4, 5, 1, 2.0,  5 },
        { 7, 3, 3, 8, 0, 1.0,  6 },  // wider input, deeper sequence
        { 4, 5, 3, 1, 1, 3.7,  7 },  // S=1: vertical path in isolation
        { 4, 1, 3, 4, 0, 1.0,  8 },  // H=1: degenerate width
        { 3, 6, 2, 6, 1, 1.0,  9 },
    };
    size_t ncases = sizeof(cases) / sizeof(cases[0]);

    for (size_t c = 0; c < ncases; c++) {
        total++;
        int rc = check_config(cases[c].I, cases[c].H, cases[c].L, cases[c].S,
                              cases[c].label, cases[c].pw, cases[c].seed, verbose);
        if (rc < 0) errors++;
        else if (rc > 0) fails++;
    }

    printf("\n%d configurations: %d passed, %d failed, %d could not be evaluated\n",
           total, total - fails - errors, fails, errors);
    return (fails || errors) ? 1 : 0;
}
