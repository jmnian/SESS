#!/bin/bash

# =============================================================================
# OPRO Experiment Runner
# =============================================================================
# This script runs multiple experiments with different configurations.
# Modify the arrays below to control which experiments to run.
#
# Architecture (8 GPUs):
#   - GPU 0: Local optimizer (gpt-oss-120b via vLLM)
#   - GPUs 1-7: Scorer workers (7 total)
#   - 7 candidates generated per step
#
# To use OpenAI API instead (all 8 GPUs for workers):
#   Set USE_LOCAL_OPTIMIZER=false, NUM_WORKERS=8, NUM_CANDIDATES_PER_STEP=8
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
EXPERIMENT_LOG="outputs/experiment_log.csv"

# Create log file with headers if it doesn't exist
if [ ! -f "$EXPERIMENT_LOG" ]; then
    mkdir -p "$(dirname "$EXPERIMENT_LOG")"
    echo "timestamp,repetition,dataset,method,portion,scorer,steps,alpha,conf_weight,status,output_dir" > "$EXPERIMENT_LOG"
fi

# =============================================================================
# CONFIGURABLE PARAMETERS (modify these arrays to run different experiments)
# =============================================================================

# Number of times to repeat the entire experiment run
# Useful for running multiple trials with different random seeds
NUM_REPETITIONS=3

# Subset selection methods to test
SUBSET_SELECT_METHODS=(
    "confidence_weighted_representative"
    "least_confident"
    "verbal_least_confident"
    "random"
    "representative"
    "IPOMP"  # Model Performance-Guided Evaluation Data Selection (Wu et al.)
    "anchor_points"  # Anchor Points: Benchmarking with Fewer Examples (Vivek et al.)
)

# Confidence weight for confidence_weighted_representative method
# (0 = pure diversity, 1 = heavily weight hard examples)
CONFIDENCE_WEIGHT=0.5

# Alpha: weight for dense (embedding) vs lexical (TF-IDF) similarity
# Used by representative and confidence_weighted_representative methods
# (1.0 = pure embedding, 0.0 = pure TF-IDF)
ALPHA=0.7

# Embedding model for representative subset selection methods
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"

# IPOMP (Model Performance-Guided) settings
# Correlation threshold for identifying redundant samples (default: 0.9)
IPOMP_CORRELATION_THRESHOLD=0.9
# Fraction of redundant samples to replace each iteration (beta in the paper, default: 0.1)
IPOMP_REPLACEMENT_RATIO=0.7

# Anchor Points settings
# Number of diverse small LMs to use as source models (scorer model is excluded)
# Set to empty string to use all available source models
# Source models are processed in parallel batches using NUM_WORKERS GPUs
# Available source models include: Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B/3B, Gemma-2B, Phi-3.5-mini, etc.
ANCHOR_NUM_SOURCE_MODELS=""  # Use all available source models (15+ models)

# Subset portions per dataset type (arrays - will run all combinations)
# Each dataset type can have multiple portion values
# All other configs (method, scorer, steps) will run for each portion
PORTIONS_GSM8K=(
    1.0
    3.5
)

PORTIONS_MATH=(
    1.0
    3.5
)

PORTIONS_GPQA=(
    10.0
    20.0
)

# Scorer models to test
SCORER_MODELS=(
    # "Qwen/Qwen2.5-7B-Instruct"
    # "openai/gpt-oss-20b"
    # "Qwen/Qwen2.5-14B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Number of search steps
NUM_SEARCH_STEPS=(
    100
    # 50
    # 200
)

# Datasets to test
# For GPQA, use "gpqa_main", "gpqa_extended", or "gpqa_diamond" to specify the subset
DATASETS=(
    "gsm8k"
    "math"
    # "bbh"
    # "gpqa_main"
    # "gpqa_extended"
    "gpqa_diamond"
)

# =============================================================================
# FIXED PARAMETERS (less commonly changed)
# =============================================================================

# Default task (overridden for GPQA datasets)
DEFAULT_TASK="train"

# Optimizer configuration
# Set USE_LOCAL_OPTIMIZER=true to use local vLLM optimizer on GPU 0
# Set USE_LOCAL_OPTIMIZER=false to use OpenAI API (all 8 GPUs for workers)
USE_LOCAL_OPTIMIZER=true
OPTIMIZER_MODEL="openai/gpt-oss-120b"  # Local: HuggingFace path, OpenAI: model name
OPTIMIZER_GPU_ID=0
OPTIMIZER_QUANTIZATION="none"  # gpt-oss uses built-in MXFP4 quantization (no extra quant needed)

# Worker configuration (when using local optimizer: 7 workers on GPUs 1-7)
NUM_WORKERS=7
NUM_CANDIDATES_PER_STEP=7

OPTIMIZER_TEMPERATURE=1.0  # Lower temp for coherent outputs from gpt-oss reasoning model
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
    echo "ERROR: EXPERIMENT FAILED!"
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
    if [ "$CURRENT_METHOD" == "representative" ] || [ "$CURRENT_METHOD" == "confidence_weighted_representative" ]; then
        LOG_ALPHA="$ALPHA"
    fi
    if [ "$CURRENT_METHOD" == "confidence_weighted_representative" ]; then
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

# Count total experiments
total_experiments=0
for dataset in "${DATASETS[@]}"; do
    # Get portions for this dataset
    portions=($(get_portions_for_dataset "$dataset"))
    for portion in "${portions[@]}"; do
        for method in "${SUBSET_SELECT_METHODS[@]}"; do
            for scorer in "${SCORER_MODELS[@]}"; do
                for steps in "${NUM_SEARCH_STEPS[@]}"; do
                    ((total_experiments++)) || true
                done
            done
        done
    done
done

echo "=============================================="
echo "OPRO Experiment Runner"
echo "=============================================="
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
echo "=============================================="
echo ""

# Run experiments with repetitions
global_experiment_num=0
for rep in $(seq 1 $NUM_REPETITIONS); do
    echo ""
    echo "##############################################"
    echo "# REPETITION $rep of $NUM_REPETITIONS"
    echo "##############################################"
    echo ""
    
    experiment_num=0
    for dataset in "${DATASETS[@]}"; do
    # Get portions for this dataset
    portions=($(get_portions_for_dataset "$dataset"))
    for portion in "${portions[@]}"; do
        for method in "${SUBSET_SELECT_METHODS[@]}"; do
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
                    # For GPQA, format is "gpqa_<subset>" (e.g., gpqa_main, gpqa_diamond)
                    if [[ "$dataset" == gpqa_* ]]; then
                        ACTUAL_DATASET="gpqa"
                        TASK="${dataset#gpqa_}"  # Extract subset name (main, extended, diamond)
                    else
                        ACTUAL_DATASET="$dataset"
                        TASK="$DEFAULT_TASK"
                    fi
                    
                    echo ""
                    echo "=============================================="
                    echo "Repetition $rep/$NUM_REPETITIONS - Experiment $experiment_num/$total_experiments (Global: $global_experiment_num/$((total_experiments * NUM_REPETITIONS)))"
                    echo "=============================================="
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
                    
                    # Add embedding_model and alpha for representative methods
                    if [ "$method" == "representative" ] || [ "$method" == "confidence_weighted_representative" ]; then
                        CMD="$CMD --embedding_model=\"$EMBEDDING_MODEL\" --alpha=$ALPHA"
                    fi
                    
                    # Add confidence_weight for confidence_weighted_representative method
                    if [ "$method" == "confidence_weighted_representative" ]; then
                        CMD="$CMD --confidence_weight=$CONFIDENCE_WEIGHT"
                    fi
                    
                    # Add IPOMP-specific parameters
                    if [ "$method" == "IPOMP" ]; then
                        CMD="$CMD --embedding_model=\"$EMBEDDING_MODEL\""
                        CMD="$CMD --ipomp_correlation_threshold=$IPOMP_CORRELATION_THRESHOLD"
                        CMD="$CMD --ipomp_replacement_ratio=$IPOMP_REPLACEMENT_RATIO"
                    fi
                    
                    # Add anchor_points-specific parameters
                    if [ "$method" == "anchor_points" ]; then
                        # Only add anchor_num_source_models if it's set (non-empty)
                        if [ -n "$ANCHOR_NUM_SOURCE_MODELS" ]; then
                            CMD="$CMD --anchor_num_source_models=$ANCHOR_NUM_SOURCE_MODELS"
                        fi
                    fi
                    
                    eval $CMD
                    
                    # Log completed experiment (only reached on success due to set -e)
                    TIMESTAMP=$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S')
                    
                    # Get alpha and confidence_weight values (may be empty for some methods)
                    LOG_ALPHA=""
                    LOG_CONF_WEIGHT=""
                    if [ "$method" == "representative" ] || [ "$method" == "confidence_weighted_representative" ]; then
                        LOG_ALPHA="$ALPHA"
                    fi
                    if [ "$method" == "confidence_weighted_representative" ]; then
                        LOG_CONF_WEIGHT="$CONFIDENCE_WEIGHT"
                    fi
                    
                    # Append to log
                    echo "\"$TIMESTAMP\",\"$rep\",\"$dataset\",\"$method\",\"$portion\",\"$scorer\",\"$steps\",\"$LOG_ALPHA\",\"$LOG_CONF_WEIGHT\",\"completed\",\"\"" >> "$EXPERIMENT_LOG"
                    
                    echo ""
                    echo "Experiment $experiment_num completed! (logged to $EXPERIMENT_LOG)"
                    echo ""
                    
                done
            done
        done
    done
done
    
    echo ""
    echo "##############################################"
    echo "# REPETITION $rep of $NUM_REPETITIONS COMPLETE"
    echo "##############################################"
done

echo ""
echo "=============================================="
echo "All $NUM_REPETITIONS repetitions completed!"
echo "Total experiments: $((total_experiments * NUM_REPETITIONS))"
echo "=============================================="
