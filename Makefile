# Build all eight RNN programs.
#   make check      # first on new host: compiler + OpenMP level
#   make            # build all
#   make seq        # serial baseline only
#   make omp31      # only what compiles under OpenMP 3.1 (skips horizontal)
#   make clean
#
# Toolchain is overridable:
#   make CC=$CC                       # conda gcc_linux-64 sets $CC
#   make CC=icx OMPFLAG=-qopenmp      # Intel oneAPI
#
# -std=gnu99 required: gcc 4.8 defaults to gnu90, which warns on
# declaration-after-statement.

CC      ?= gcc
STD     ?= -std=gnu99
OPT     ?= -O2
OMPFLAG ?= -fopenmp
WARN    ?= -Wall -Wextra

# ARCH: ISA to generate for.
#   broadwell = Xeon E5-2620 v4: AVX2 + FMA
#   native    = compiling host; avoid if login and compute nodes differ (SIGILL risk)
ARCH    ?= -march=broadwell

# CFLAGS uses ':=' so environment CFLAGS (e.g. from `conda activate`, which sets
# -march=nocona and drops -std/-Wall) cannot override it. Command-line
# `make CFLAGS=...` still wins. Environment value kept for `make check`.
CFLAGS_FROM_ENV := $(if $(filter environment,$(origin CFLAGS)),$(CFLAGS),<none>)
CFLAGS  := $(STD) $(OPT) $(WARN) $(ARCH) $(EXTRA_CFLAGS)
LDLIBS  ?= -lm

SEQ_DIR   = SingleLayer/SingleLayerRNN
MULTI_DIR = MultilayerRNN
PAR_DIR   = SingleLayer/SingleLayerParallel
HORIZ_DIR = SingleLayer/SingleLayerHorizontal
MHORIZ_DIR = MultilayerHorizontal
VERT_DIR  = SingleLayer/SingleLayerVerticle
MVERT_DIR = MultilayerVertical
MPAR_DIR  = MultilayerParallel

.PHONY: all omp31 multilayer multilayer31 multilayer-double seq par horiz multihoriz vert multivert multipar multi clean check gradcheck gradcheck-single gradcheck-multi

all: seq par horiz multihoriz vert multivert multipar multi

# Everything that builds under OpenMP 3.1 (both horizontal programs excluded).
omp31: seq par vert multivert multipar multi

# Four multi-layer programs only. --layers 1 is bit-exact with single-layer counterpart.
multilayer: multi multipar multivert multihoriz

# Same, minus hidden-dimension program, for OpenMP 3.1 hosts.
multilayer31: multi multipar multivert

# Double-precision builds of multi-layer programs, separate binary names (*_dbl).
multilayer-double:
	cd "$(MULTI_DIR)"  && $(CC) $(CFLAGS) -DRNN_USE_DOUBLE RNN_main_multilayer.c RNN_seq_multilayer.c RNN_dataset.c -o rnn_multilayer_dbl $(LDLIBS)
	cd "$(MPAR_DIR)"   && $(CC) $(CFLAGS) $(OMPFLAG) -DRNN_USE_DOUBLE RNN_main_parallel_multilayer.c RNN_seq_parallel_multilayer.c RNN_dataset.c -o rnn_parallel_multilayer_dbl $(LDLIBS)
	cd "$(MHORIZ_DIR)" && $(CC) $(CFLAGS) $(OMPFLAG) -DRNN_USE_DOUBLE RNN_main_horizontal_multilayer.c RNN_seq_horizontal_multilayer.c RNN_dataset.c -o rnn_horizontal_multilayer_dbl $(LDLIBS)
	cd "$(MVERT_DIR)"  && $(CC) $(CFLAGS) $(OMPFLAG) -DRNN_USE_DOUBLE RNN_main_verticle_multilayer.c RNN_seq_verticle_multilayer.c RNN_dataset.c -o rnn_verticle_multilayer_dbl $(LDLIBS)

seq:
	@echo "==> SingleLayerRNN (serial baseline)"
	cd "$(SEQ_DIR)" && $(CC) $(CFLAGS) RNN_main.c RNN_seq.c RNN_dataset.c -o rnn $(LDLIBS)

par:
	@echo "==> SingleLayerParallel (OpenMP, batch data-parallel)"
	cd "$(PAR_DIR)" && $(CC) $(CFLAGS) $(OMPFLAG) RNN_main_parallel_openmp.c RNN_seq_parallel_openmp.c RNN_dataset.c -o rnn_parallel_openmp $(LDLIBS)

# Requires OpenMP 4.5 (_OPENMP >= 201511) for array-section reductions.
horiz:
	@echo "==> SingleLayerHorizontal (OpenMP 4.5 required)"
	cd "$(HORIZ_DIR)" && $(CC) $(CFLAGS) $(OMPFLAG) RNN_main_horizontal.c RNN_seq_horizontal.c RNN_dataset.c -o rnn_horizontal $(LDLIBS)

# Multi-layer hidden-dimension partitioning. OpenMP 4.5: same array-section
# reductions as horiz, plus second one per (timestep, layer) for vertical gradient
# dh_down. --layers 1 is bit-exact with rnn_horizontal.
multihoriz:
	@echo "==> MultilayerHorizontal (OpenMP 4.5 required, L > 1)"
	cd "$(MHORIZ_DIR)" && $(CC) $(CFLAGS) $(OMPFLAG) RNN_main_horizontal_multilayer.c RNN_seq_horizontal_multilayer.c RNN_dataset.c -o rnn_horizontal_multilayer $(LDLIBS)

vert:
	@echo "==> SingleLayerVerticle (OpenMP, timestep pipeline)"
	cd "$(VERT_DIR)" && $(CC) $(CFLAGS) $(OMPFLAG) RNN_main_verticle.c RNN_seq_verticle.c RNN_dataset.c -o rnn_verticle $(LDLIBS)

# Multi-layer timestep pipeline. Each stage owns contiguous timestep range and
# runs all L layers per timestep; stage-boundary handoff (dh_carry) holds one
# hidden_size slot per layer. OpenMP 3.1 suffices (barriers only, no
# array-section reductions). --layers 1 is bit-exact with rnn_verticle.
multivert:
	@echo "==> MultilayerVertical (OpenMP, timestep pipeline, L > 1)"
	cd "$(MVERT_DIR)" && $(CC) $(CFLAGS) $(OMPFLAG) RNN_main_verticle_multilayer.c RNN_seq_verticle_multilayer.c RNN_dataset.c -o rnn_verticle_multilayer $(LDLIBS)

# Multi-layer batch data-parallel program. Same core as SingleLayerParallel.
multipar:
	@echo "==> MultilayerParallel (OpenMP, batch data-parallel, L > 1 -- same core as SingleLayerParallel, which is L=1 only)"
	cd "$(MPAR_DIR)" && $(CC) $(CFLAGS) $(OMPFLAG) RNN_main_parallel_multilayer.c RNN_seq_parallel_multilayer.c RNN_dataset.c -o rnn_parallel_multilayer $(LDLIBS)

# Multi-layer serial program (no OpenMP). --layers 1 is bit-exact with SingleLayerRNN.
multi:
	@echo "==> MultilayerRNN (serial, L > 1)"
	cd "$(MULTI_DIR)" && $(CC) $(CFLAGS) RNN_main_multilayer.c RNN_seq_multilayer.c RNN_dataset.c -o rnn_multilayer $(LDLIBS)

# Host diagnostic: compiler, CFLAGS (incl. any from environment), vector ISA,
# OpenMP level (horizontal programs need >= 201511), core/NUMA layout.
check:
	@echo "compiler : $$($(CC) --version | head -1)"
	@echo "CFLAGS   : $(CFLAGS)"
	@echo "env CFLAGS wanted: $(CFLAGS_FROM_ENV)"
	@printf 'vector   : '; echo | $(CC) -dM -E -march=native - 2>/dev/null | grep -oE '__(AVX2|AVX|FMA|SSE4_2)__' | tr '\n' ' '; echo
	@printf 'openmp   : '; echo | $(CC) $(OMPFLAG) -dM -E - 2>/dev/null | grep '_OPENMP' || echo "NO OpenMP"
	@echo "           (need _OPENMP >= 201511 for both horizontal programs;"
	@echo "            201107 = OpenMP 3.1 -> use \"make omp31\")"
	@printf 'cores    : '; nproc --all | tr '\n' ' '; echo "logical"
	@lscpu 2>/dev/null | grep -E "Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)" || true

# Finite-difference check of every analytic gradient in both BPTT copies
# (SingleLayerRNN, MultilayerRNN). Built in double: float central differences
# too coarse. Exits nonzero on failure.
gradcheck: gradcheck-single gradcheck-multi

gradcheck-single:
	@echo "==> gradcheck: SingleLayerRNN (double precision)"
	cd "$(SEQ_DIR)" && $(CC) $(CFLAGS) -DRNN_USE_DOUBLE gradcheck.c RNN_seq.c RNN_dataset.c -o gradcheck $(LDLIBS) && ./gradcheck

# Multi-layer copy only. Covers L = 1..4.
gradcheck-multi:
	@echo "==> gradcheck: MultilayerRNN (double precision)"
	cd "$(MULTI_DIR)" && $(CC) $(CFLAGS) -DRNN_USE_DOUBLE gradcheck.c RNN_seq_multilayer.c RNN_dataset.c -o gradcheck $(LDLIBS) && ./gradcheck

clean:
	rm -f "$(SEQ_DIR)/rnn" "$(SEQ_DIR)/gradcheck" \
	      "$(MULTI_DIR)/rnn_multilayer" "$(MULTI_DIR)/gradcheck" \
	      "$(PAR_DIR)/rnn_parallel_openmp" \
	      "$(HORIZ_DIR)/rnn_horizontal" \
	      "$(MHORIZ_DIR)/rnn_horizontal_multilayer" \
	      "$(VERT_DIR)/rnn_verticle" \
	      "$(MVERT_DIR)/rnn_verticle_multilayer" \
	      "$(MPAR_DIR)/rnn_parallel_multilayer" \
	      "$(MULTI_DIR)/rnn_multilayer_dbl" "$(MPAR_DIR)/rnn_parallel_multilayer_dbl" \
	      "$(MHORIZ_DIR)/rnn_horizontal_multilayer_dbl" "$(MVERT_DIR)/rnn_verticle_multilayer_dbl"
