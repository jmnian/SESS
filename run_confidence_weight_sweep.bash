#!/bin/bash

# =============================================================================
# OPRO Confidence Weight Hyperparameter Sweep
# =============================================================================
# This script runs hyperparameter tuning on confidence_weight for:
#   1. confidence_weighted_representative (logit-based confidence)
#   2. verbal_confidence_weighted_representative (verbal confidence)
#
# Sweep: confidence_weight from 0.1 to 1.0 in 0.1 increments
#
# Formula: w_i = (1 - confidence_weight) + confidence_weight * (1 - normalized_confidence_i)
#
# At confidence_weight = 0: all weights = 1 (pure diversity)
# At confidence_weight = 1: weights range from 0 (most confident) to 1 (least confident)
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
EXPERIMENT_LOG="outputs/confidence_weight_sweep_log.csv"

# Create log file with headers if it doesn't exist
if [ ! -f "$EXPERIMENT_LOG" ]; then
    mkdir -p "$(dirname "$EXPERIMENT_LOG")"
    echo "timestamp,repetition,dataset,method,portion,scorer,steps,alpha,conf_weight,status,output_dir" > "$EXPERIMENT_LOG"
fi

# =============================================================================
# HYPERPARAMETER SWEEP CONFIGURATION
# =============================================================================

# Number of times to repeat the entire experiment run for stability
NUM_REPETITIONS=3

# Subset selection methods to sweep
SUBSET_SELECT_METHODS=(
    "confidence_weighted_representative"          # Logit-based confidence
    "verbal_confidence_weighted_representative"   # Verbal confidence
)

# Confidence weight values to sweep (0.1 to 1.0 in 0.1 increments)
# Formula: w_i = (1 - cw) + cw * (1 - conf_normalized)
# At cw=0.1: weights range from 0.9 to 1.0 (mild emphasis on hard examples)
# At cw=1.0: weights range from 0.0 to 1.0 (strong emphasis on hard examples)
CONFIDENCE_WEIGHTS=(
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
    1.0
)

# Alpha: weight for dense (embedding) vs lexical (TF-IDF) similarity
ALPHA=0.7

# Embedding model for representative subset selection methods
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"

# Datasets
DATASETS=(
    "gsm8k"
    "math"
    "gpqa_diamond"
)

# Subset portions per dataset type
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

# Scorer model - using Qwen as specified
SCORER_MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

# Number of search steps
NUM_SEARCH_STEPS=(
    100
)

# =============================================================================
# FIXED PARAMETERS
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

CURRENT_REPETITION=0
CURRENT_EXPERIMENT_NUM=0
CURRENT_DATASET=""
CURRENT_METHOD=""
CURRENT_PORTION=""
CURRENT_SCORER=""
CURRENT_STEPS=""
CURRENT_CONF_WEIGHT=""

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
    echo "  Confidence Weight: $CURRENT_CONF_WEIGHT"
    echo "  Scorer Model: $CURRENT_SCORER"
    echo "  Search Steps: $CURRENT_STEPS"
    echo "  Exit Code: $exit_code"
    echo "=============================================="
    echo ""
    
    # Log failed experiment
    TIMESTAMP=$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S')
    echo "\"$TIMESTAMP\",\"$CURRENT_REPETITION\",\"$CURRENT_DATASET\",\"$CURRENT_METHOD\",\"$CURRENT_PORTION\",\"$CURRENT_SCORER\",\"$CURRENT_STEPS\",\"$ALPHA\",\"$CURRENT_CONF_WEIGHT\",\"failed (exit $exit_code)\",\"\"" >> "$EXPERIMENT_LOG"
    
    exit $exit_code
}

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
    portions=($(get_portions_for_dataset "$dataset"))
    for portion in "${portions[@]}"; do
        for method in "${SUBSET_SELECT_METHODS[@]}"; do
            for conf_weight in "${CONFIDENCE_WEIGHTS[@]}"; do
                for scorer in "${SCORER_MODELS[@]}"; do
                    for steps in "${NUM_SEARCH_STEPS[@]}"; do
                        ((total_experiments++)) || true
                    done
                done
            done
        done
    done
done

echo "=============================================="
echo "OPRO Confidence Weight Hyperparameter Sweep"
echo "=============================================="
echo ""
echo "PURPOSE: Find optimal confidence_weight for:"
echo "  - confidence_weighted_representative (logit-based)"
echo "  - verbal_confidence_weighted_representative (verbal)"
echo ""
echo "Formula: w_i = (1 - cw) + cw * (1 - normalized_confidence)"
echo "  cw=0.1: mild emphasis on hard examples (weights 0.9-1.0)"
echo "  cw=1.0: strong emphasis on hard examples (weights 0.0-1.0)"
echo ""
echo "=============================================="
echo "Methods: ${SUBSET_SELECT_METHODS[*]}"
echo "Confidence weights: ${CONFIDENCE_WEIGHTS[*]}"
echo "Datasets: ${DATASETS[*]}"
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
                for conf_weight in "${CONFIDENCE_WEIGHTS[@]}"; do
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
                            CURRENT_CONF_WEIGHT=$conf_weight
                            
                            # Parse dataset name and task
                            if [[ "$dataset" == gpqa_* ]]; then
                                ACTUAL_DATASET="gpqa"
                                TASK="${dataset#gpqa_}"
                            else
                                ACTUAL_DATASET="$dataset"
                                TASK="$DEFAULT_TASK"
                            fi
                            
                            # Determine confidence type for display
                            if [[ "$method" == "verbal_confidence_weighted_representative" ]]; then
                                CONF_TYPE="[VERBAL CONFIDENCE]"
                            else
                                CONF_TYPE="[LOGIT-BASED CONFIDENCE]"
                            fi
                            
                            echo ""
                            echo "=============================================="
                            echo "Repetition $rep/$NUM_REPETITIONS - Experiment $experiment_num/$total_experiments (Global: $global_experiment_num/$((total_experiments * NUM_REPETITIONS)))"
                            echo "=============================================="
                            echo "  $CONF_TYPE"
                            echo "  Dataset: $ACTUAL_DATASET (task: $TASK)"
                            echo "  Subset Method: $method"
                            echo "  Subset Portion: $portion%"
                            echo "  Confidence Weight: $conf_weight"
                            echo "  Alpha: $ALPHA"
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
                                --seed=$SEED \
                                --embedding_model=\"$EMBEDDING_MODEL\" \
                                --alpha=$ALPHA \
                                --confidence_weight=$conf_weight"
                            
                            # Add local optimizer flags
                            if [ "$USE_LOCAL_OPTIMIZER" == "true" ]; then
                                CMD="$CMD --use_local_optimizer \
                                    --optimizer_gpu_id=$OPTIMIZER_GPU_ID \
                                    --optimizer_quantization=\"$OPTIMIZER_QUANTIZATION\""
                            else
                                CMD="$CMD --nouse_local_optimizer"
                            fi
                            
                            eval $CMD
                            
                            # Log completed experiment
                            TIMESTAMP=$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S')
                            echo "\"$TIMESTAMP\",\"$rep\",\"$dataset\",\"$method\",\"$portion\",\"$scorer\",\"$steps\",\"$ALPHA\",\"$conf_weight\",\"completed\",\"\"" >> "$EXPERIMENT_LOG"
                            
                            echo ""
                            echo "Experiment $experiment_num completed! (logged to $EXPERIMENT_LOG)"
                            echo ""
                            
                        done
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
echo ""
echo "Results summary:"
echo "  - Log file: $EXPERIMENT_LOG"
echo "  - Results directory: outputs/optimization-results/"
echo ""
echo "Analysis:"
echo "  Compare performance across confidence_weight values for:"
echo "  - confidence_weighted_representative (logit-based)"
echo "  - verbal_confidence_weighted_representative (verbal)"
echo "=============================================="
