#!/bin/bash

# =============================================================================
# OPRO Ablation Experiment Runner
# =============================================================================
# This script runs ablation experiments comparing original subset selection
# methods with their "opposite" counterparts to demonstrate that the original
# methods make a meaningful difference.
#
# Ablation pairs:
#   - least_confident (hard) vs most_confident (easy)
#   - representative (diverse) vs least_representative (redundant)
#   - confidence_weighted_representative (hard+diverse) vs 
#     confidence_weighted_least_representative (easy+redundant)
#
# Architecture (8 GPUs):
#   - GPU 0: Local optimizer (gpt-oss-120b via vLLM)
#   - GPUs 1-7: Scorer workers (7 total)
#   - 7 candidates generated per step
# =============================================================================

set -e  # Exit on error

# Source uv environment (needed for uv command)
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
fi

# =============================================================================
# EXPERIMENT LOG
# =============================================================================
# Log file to track completed experiments
EXPERIMENT_LOG="outputs/ablation_experiment_log.csv"

# Create log file with headers if it doesn't exist
if [ ! -f "$EXPERIMENT_LOG" ]; then
    mkdir -p "$(dirname "$EXPERIMENT_LOG")"
    echo "timestamp,repetition,dataset,method,portion,scorer,steps,alpha,conf_weight,status,output_dir" > "$EXPERIMENT_LOG"
fi

# =============================================================================
# ABLATION EXPERIMENT CONFIGURATION
# =============================================================================

# Number of times to repeat the entire experiment run
NUM_REPETITIONS=3

# Subset selection methods for ablation study
# Pairs: original method vs opposite method
SUBSET_SELECT_METHODS=(
    # Original methods
    # "least_confident"                       # Select hardest examples
    # "representative"                        # Select most diverse examples
    # "confidence_weighted_representative"    # Select hard + diverse examples
    # Opposite/ablation methods
    "most_confident"                        # Select EASIEST examples
    "least_representative"                  # Select LEAST diverse examples
    "confidence_weighted_least_representative"  # Select EASY + LEAST DIVERSE
)

# =============================================================================
# RANDOM SEED EXPERIMENTS
# =============================================================================
# These 10 seeds are chosen to be mathematically diverse and produce very
# different random subsets. Why they're different:
#
# 1. Seeds are widely spaced (not consecutive) - different PRNG states
# 2. Mix of primes (7, 13, 97, 1009, 7919) and composites - different factorizations
# 3. Include small (7), medium (123, 456), and large (999999) values
# 4. Include powers of 2 adjacent (2024) and far from powers - different bit patterns
# 5. Each seed initializes numpy's Mersenne Twister PRNG to a completely different
#    state, producing independent random sequences with period 2^19937-1
#
# Mathematical guarantee: With these seeds, for a dataset of N items selecting k,
# the probability that any two seeds select the exact same subset is ~(k/N)^k,
# which for k=100, N=7000 is essentially 0.
#
# To verify seeds produce different subsets, you can run:
#   python -c "import numpy as np; [print(f'seed {s}:', sorted(np.random.default_rng(s).choice(1000, 35, replace=False))[:5]) for s in [7,42,123,456,789,1009,2024,7919,31415,999999]]"
# =============================================================================
RANDOM_SEEDS=(
    7        # Small prime
    123      # Simple memorable seed
    1009     # Prime number in 4 digits
    7919     # 1000th prime number
    999999   # Large seed, near boundary
)

# Parameters for representative methods
# Alpha: weight for dense (embedding) vs lexical (TF-IDF) similarity
ALPHA=0.7

# Confidence weight for weighted methods
CONFIDENCE_WEIGHT=0.5

# Embedding model for representative subset selection methods
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"

# Subset portions per dataset type
# Using same portions as main experiments for fair comparison
PORTIONS_GSM8K=(
    1.0
    # 3.5
)

PORTIONS_MATH=(
    1.0
    # 3.5
)

PORTIONS_GPQA=(
    10.0
    # 20.0
)

# Scorer model - using Qwen as requested
SCORER_MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

# Number of search steps
NUM_SEARCH_STEPS=(
    100
)

# Datasets to test - all three datasets
DATASETS=(
    "gsm8k"
    "math"
    "gpqa_diamond"
)

# =============================================================================
# FIXED PARAMETERS (less commonly changed)
# =============================================================================

# Default task (overridden for GPQA datasets)
DEFAULT_TASK="train"

# Optimizer configuration
USE_LOCAL_OPTIMIZER=true
OPTIMIZER_MODEL="openai/gpt-oss-120b"
OPTIMIZER_GPU_ID=0
OPTIMIZER_QUANTIZATION="none"

# Worker configuration
NUM_WORKERS=7
NUM_CANDIDATES_PER_STEP=7

OPTIMIZER_TEMPERATURE=1.0
SCORER_MAX_TOKENS=1024
GPU_MEMORY_UTILIZATION=0.90
INSTRUCTION_POS="Q_begin"
FEW_SHOT_SELECTION_CRITERIA="random"
NUM_FEW_SHOT_EXAMPLES=3
VERBOSE_EVAL_LOGGING=false
SEED=42

# =============================================================================
# ERROR HANDLING
# =============================================================================

# Variables to track current experiment (for error reporting)
CURRENT_REPETITION=0
CURRENT_EXPERIMENT_NUM=0
CURRENT_DATASET=""
CURRENT_METHOD=""
CURRENT_PORTION=""
CURRENT_SCORER=""
CURRENT_STEPS=""

# Error handler function
error_handler() {
    local exit_code=$?
    echo ""
    echo "=============================================="
    echo "ERROR: ABLATION EXPERIMENT FAILED!"
    echo "=============================================="
    echo "  Repetition: $CURRENT_REPETITION"
    echo "  Experiment: $CURRENT_EXPERIMENT_NUM"
    echo "  Dataset: $CURRENT_DATASET"
    echo "  Subset Method: $CURRENT_METHOD"
    echo "  Subset Portion: $CURRENT_PORTION%"
    echo "  Scorer Model: $CURRENT_SCORER"
    echo "  Search Steps: $CURRENT_STEPS"
    echo "  Exit Code: $exit_code"
    echo "=============================================="
    echo ""
    
    # Log failed experiment
    TIMESTAMP=$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S')
    LOG_ALPHA=""
    LOG_CONF_WEIGHT=""
    if [ "$CURRENT_METHOD" == "representative" ] || \
       [ "$CURRENT_METHOD" == "confidence_weighted_representative" ] || \
       [ "$CURRENT_METHOD" == "least_representative" ] || \
       [ "$CURRENT_METHOD" == "confidence_weighted_least_representative" ]; then
        LOG_ALPHA="$ALPHA"
    fi
    if [ "$CURRENT_METHOD" == "confidence_weighted_representative" ] || \
       [ "$CURRENT_METHOD" == "confidence_weighted_least_representative" ]; then
        LOG_CONF_WEIGHT="$CONFIDENCE_WEIGHT"
    fi
    echo "\"$TIMESTAMP\",\"$CURRENT_REPETITION\",\"$CURRENT_DATASET\",\"$CURRENT_METHOD\",\"$CURRENT_PORTION\",\"$CURRENT_SCORER\",\"$CURRENT_STEPS\",\"$LOG_ALPHA\",\"$LOG_CONF_WEIGHT\",\"failed (exit $exit_code)\",\"\"" >> "$EXPERIMENT_LOG"
    
    exit $exit_code
}

# Set up trap to catch errors
trap error_handler ERR

# =============================================================================
# EXPERIMENT LOOP
# =============================================================================

# Helper function to get portions array for a dataset
get_portions_for_dataset() {
    local dataset="$1"
    if [[ "$dataset" == "gsm8k" ]]; then
        echo "${PORTIONS_GSM8K[@]}"
    elif [[ "$dataset" == "math" ]]; then
        echo "${PORTIONS_MATH[@]}"
    elif [[ "$dataset" == gpqa_* ]]; then
        echo "${PORTIONS_GPQA[@]}"
    else
        echo "3.5"  # Default fallback
    fi
}

# Build full list of methods including random seeds
ALL_METHODS=()
for method in "${SUBSET_SELECT_METHODS[@]}"; do
    ALL_METHODS+=("$method")
done
# Add random seed experiments
for seed in "${RANDOM_SEEDS[@]}"; do
    ALL_METHODS+=("random_${seed}")
done

# Count total experiments
total_experiments=0
for dataset in "${DATASETS[@]}"; do
    portions=($(get_portions_for_dataset "$dataset"))
    for portion in "${portions[@]}"; do
        for method in "${ALL_METHODS[@]}"; do
            for scorer in "${SCORER_MODELS[@]}"; do
                for steps in "${NUM_SEARCH_STEPS[@]}"; do
                    ((total_experiments++)) || true
                done
            done
        done
    done
done

echo "=============================================="
echo "OPRO ABLATION Experiment Runner"
echo "=============================================="
echo ""
echo "PURPOSE: Demonstrate that subset selection methods matter"
echo "by comparing original methods with their 'opposite' versions,"
echo "and testing variance across different random baselines."
echo ""
echo "Ablation pairs being tested:"
echo "  - least_confident (hard) vs most_confident (easy)"
echo "  - representative (diverse) vs least_representative (redundant)"
echo "  - confidence_weighted_representative vs confidence_weighted_least_representative"
echo ""
echo "Random seed baselines (10 different seeds):"
echo "  Seeds: ${RANDOM_SEEDS[*]}"
echo "  Purpose: Show variance of random selection and that"
echo "           targeted methods outperform random consistently"
echo ""
echo "=============================================="
echo "Ablation methods: ${SUBSET_SELECT_METHODS[*]}"
echo "Random seeds: ${#RANDOM_SEEDS[@]} different seeds"
echo "Total methods per config: ${#ALL_METHODS[@]}"
echo "Experiments per repetition: $total_experiments"
echo "Number of repetitions: $NUM_REPETITIONS"
echo "Total experiments: $((total_experiments * NUM_REPETITIONS))"
if [ "$USE_LOCAL_OPTIMIZER" == "true" ]; then
    echo "Optimizer: LOCAL vLLM on GPU $OPTIMIZER_GPU_ID ($OPTIMIZER_MODEL)"
    echo "Workers: $NUM_WORKERS (GPUs $((OPTIMIZER_GPU_ID + 1))-$((OPTIMIZER_GPU_ID + NUM_WORKERS)))"
else
    echo "Optimizer: OpenAI API ($OPTIMIZER_MODEL)"
    echo "Workers: $NUM_WORKERS (GPUs 0-$((NUM_WORKERS - 1)))"
fi
echo "Candidates per step: $NUM_CANDIDATES_PER_STEP"
echo "Scorer model: ${SCORER_MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "=============================================="
echo ""

# Run experiments with repetitions
global_experiment_num=0
for rep in $(seq 1 $NUM_REPETITIONS); do
    echo ""
    echo "##############################################"
    echo "# ABLATION REPETITION $rep of $NUM_REPETITIONS"
    echo "##############################################"
    echo ""
    
    experiment_num=0
    for dataset in "${DATASETS[@]}"; do
    # Get portions for this dataset
    portions=($(get_portions_for_dataset "$dataset"))
    for portion in "${portions[@]}"; do
        for method in "${ALL_METHODS[@]}"; do
            for scorer in "${SCORER_MODELS[@]}"; do
                for steps in "${NUM_SEARCH_STEPS[@]}"; do
                    ((experiment_num++)) || true
                    ((global_experiment_num++)) || true
                    
                    # Update tracking variables for error handler
                    CURRENT_REPETITION=$rep
                    CURRENT_EXPERIMENT_NUM=$experiment_num
                    CURRENT_DATASET=$dataset
                    CURRENT_METHOD=$method
                    CURRENT_PORTION=$portion
                    CURRENT_SCORER=$scorer
                    CURRENT_STEPS=$steps
                    
                    # Parse dataset name and task
                    if [[ "$dataset" == gpqa_* ]]; then
                        ACTUAL_DATASET="gpqa"
                        TASK="${dataset#gpqa_}"
                    else
                        ACTUAL_DATASET="$dataset"
                        TASK="$DEFAULT_TASK"
                    fi
                    
                    # Determine method type for display
                    if [[ "$method" == "most_confident" ]] || \
                       [[ "$method" == "least_representative" ]] || \
                       [[ "$method" == "confidence_weighted_least_representative" ]]; then
                        METHOD_TYPE="[ABLATION/OPPOSITE]"
                    elif [[ "$method" == random_* ]]; then
                        METHOD_TYPE="[RANDOM SEED BASELINE]"
                    else
                        METHOD_TYPE="[ORIGINAL]"
                    fi
                    
                    echo ""
                    echo "=============================================="
                    echo "Repetition $rep/$NUM_REPETITIONS - Experiment $experiment_num/$total_experiments (Global: $global_experiment_num/$((total_experiments * NUM_REPETITIONS)))"
                    echo "=============================================="
                    echo "  $METHOD_TYPE"
                    echo "  Dataset: $ACTUAL_DATASET (task: $TASK)"
                    echo "  Subset Method: $method"
                    echo "  Subset Portion: $portion%"
                    echo "  Scorer Model: $scorer"
                    echo "  Optimizer: $OPTIMIZER_MODEL ($([ "$USE_LOCAL_OPTIMIZER" == "true" ] && echo "local" || echo "API"))"
                    echo "  Search Steps: $steps"
                    echo "  Workers: $NUM_WORKERS, Candidates/step: $NUM_CANDIDATES_PER_STEP"
                    echo "=============================================="
                    echo ""
                    
                    # Build command
                    CMD="uv run python run_opro_parallel.py \
                        --dataset=\"$ACTUAL_DATASET\" \
                        --task=\"$TASK\" \
                        --num_workers=$NUM_WORKERS \
                        --scorer_model=\"$scorer\" \
                        --optimizer_model=\"$OPTIMIZER_MODEL\" \
                        --num_search_steps=$steps \
                        --num_candidates_per_step=$NUM_CANDIDATES_PER_STEP \
                        --optimizer_temperature=$OPTIMIZER_TEMPERATURE \
                        --scorer_max_tokens=$SCORER_MAX_TOKENS \
                        --gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
                        --subset_portion=$portion \
                        --subset_select_method=\"$method\" \
                        --instruction_pos=\"$INSTRUCTION_POS\" \
                        --few_shot_selection_criteria=\"$FEW_SHOT_SELECTION_CRITERIA\" \
                        --num_few_shot_examples=$NUM_FEW_SHOT_EXAMPLES \
                        --verbose_eval_logging=$VERBOSE_EVAL_LOGGING \
                        --seed=$SEED"
                    
                    # Add local optimizer flags
                    if [ "$USE_LOCAL_OPTIMIZER" == "true" ]; then
                        CMD="$CMD --use_local_optimizer \
                            --optimizer_gpu_id=$OPTIMIZER_GPU_ID \
                            --optimizer_quantization=\"$OPTIMIZER_QUANTIZATION\""
                    else
                        CMD="$CMD --nouse_local_optimizer"
                    fi
                    
                    # Add embedding_model and alpha for representative methods (both original and ablation)
                    if [ "$method" == "representative" ] || \
                       [ "$method" == "confidence_weighted_representative" ] || \
                       [ "$method" == "least_representative" ] || \
                       [ "$method" == "confidence_weighted_least_representative" ]; then
                        CMD="$CMD --embedding_model=\"$EMBEDDING_MODEL\" --alpha=$ALPHA"
                    fi
                    
                    # Add confidence_weight for confidence-weighted methods (both original and ablation)
                    if [ "$method" == "confidence_weighted_representative" ] || \
                       [ "$method" == "confidence_weighted_least_representative" ]; then
                        CMD="$CMD --confidence_weight=$CONFIDENCE_WEIGHT"
                    fi
                    
                    eval $CMD
                    
                    # Log completed experiment
                    TIMESTAMP=$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S')
                    
                    # Get alpha and confidence_weight values
                    LOG_ALPHA=""
                    LOG_CONF_WEIGHT=""
                    if [ "$method" == "representative" ] || \
                       [ "$method" == "confidence_weighted_representative" ] || \
                       [ "$method" == "least_representative" ] || \
                       [ "$method" == "confidence_weighted_least_representative" ]; then
                        LOG_ALPHA="$ALPHA"
                    fi
                    if [ "$method" == "confidence_weighted_representative" ] || \
                       [ "$method" == "confidence_weighted_least_representative" ]; then
                        LOG_CONF_WEIGHT="$CONFIDENCE_WEIGHT"
                    fi
                    
                    # Append to log
                    echo "\"$TIMESTAMP\",\"$rep\",\"$dataset\",\"$method\",\"$portion\",\"$scorer\",\"$steps\",\"$LOG_ALPHA\",\"$LOG_CONF_WEIGHT\",\"completed\",\"\"" >> "$EXPERIMENT_LOG"
                    
                    echo ""
                    echo "Ablation experiment $experiment_num completed! (logged to $EXPERIMENT_LOG)"
                    echo ""
                    
                done
            done
        done
    done
done
    
    echo ""
    echo "##############################################"
    echo "# ABLATION REPETITION $rep of $NUM_REPETITIONS COMPLETE"
    echo "##############################################"
done

echo ""
echo "=============================================="
echo "All $NUM_REPETITIONS ablation repetitions completed!"
echo "Total experiments: $((total_experiments * NUM_REPETITIONS))"
echo ""
echo "Results summary:"
echo "  - Log file: $EXPERIMENT_LOG"
echo "  - Results directory: outputs/optimization-results/"
echo ""
echo "Ablation analysis:"
echo "  Compare performance between original vs opposite methods:"
echo "  - least_confident vs most_confident"
echo "  - representative vs least_representative"
echo "  - confidence_weighted_representative vs confidence_weighted_least_representative"
echo "=============================================="
