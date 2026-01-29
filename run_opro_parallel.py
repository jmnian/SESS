#!/usr/bin/env python3
r"""Multi-GPU OPRO optimization using vLLM workers and local LLM optimizer.

This script runs OPRO with:
- 1 GPU (GPU 0) for the optimizer LLM (e.g., gpt-oss-120b 4-bit)
- 7 vLLM worker processes (GPUs 1-7) for scoring
- 7 candidates generated per step (one per worker)
- Parallel evaluation across all scorer GPUs

Usage:
```bash
# Run with default settings (7 workers on GPUs 1-7, local optimizer on GPU 0)
python run_opro_parallel.py \
    --dataset="gsm8k" \
    --task="train" \
    --num_search_steps=100

# Use OpenAI API instead of local optimizer (all 8 GPUs for workers)
python run_opro_parallel.py \
    --dataset="gsm8k" \
    --task="train" \
    --nouse_local_optimizer \
    --num_workers=8 \
    --optimizer_model="gpt-4o"

# Use a different local optimizer model
python run_opro_parallel.py \
    --dataset="gsm8k" \
    --task="train" \
    --optimizer_model="Qwen/Qwen2.5-72B-Instruct" \
    --optimizer_quantization="awq"

# Resume from checkpoint
python run_opro_parallel.py \
    --dataset="gsm8k" \
    --task="train" \
    --resume_from="outputs/optimization-results/.../results_dict.pkl"
```

Architecture:
- GPU 0: Local optimizer (gpt-oss-120b or similar via vLLM)
- GPUs 1-7: Scorer workers (vLLM with scorer model)
- Controller process: manages OPRO loop, coordinates generation and evaluation
- Communication via multiprocessing queues (no HTTP, no model reload)
"""

import datetime
import os
import sys
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Add OPRO root to path
OPRO_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, OPRO_ROOT_PATH)

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from opro.optimization import opt_utils, subset_selection
from opro.evaluation import eval_utils
from opro.parallel.controller import OPROController

ROOT_DATA_FOLDER_PATH = os.path.join(OPRO_ROOT_PATH, "data")

# ============== Flags ==============

_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 7,
    "Number of GPU workers for scoring (default 7 for GPUs 1-7)."
)

_SCORER_MODEL = flags.DEFINE_string(
    "scorer_model", "Qwen/Qwen2.5-7B-Instruct",
    "Model for scoring (vLLM workers)."
)

_USE_LOCAL_OPTIMIZER = flags.DEFINE_boolean(
    "use_local_optimizer", True,
    "Use local vLLM optimizer (GPU 0) instead of OpenAI API. Default: True."
)

_OPTIMIZER_MODEL = flags.DEFINE_string(
    "optimizer_model", "openai/gpt-oss-120b",
    "Model for optimization. For local: HuggingFace path (e.g., openai/gpt-oss-120b). "
    "For OpenAI: model name (e.g., gpt-4o)."
)

_OPTIMIZER_GPU_ID = flags.DEFINE_integer(
    "optimizer_gpu_id", 0,
    "GPU ID for local optimizer (default 0)."
)

_OPTIMIZER_QUANTIZATION = flags.DEFINE_string(
    "optimizer_quantization", "none",
    "Quantization for local optimizer: fp8, awq, gptq, or none. "
    "For gpt-oss models, use 'none' as they have built-in MXFP4 quantization."
)

_DATASET = flags.DEFINE_string(
    "dataset", "gsm8k",
    "Dataset name: gsm8k, math, bbh, mmlu, or gpqa."
)

_TASK = flags.DEFINE_string(
    "task", "train",
    "Task within dataset."
)

_INSTRUCTION_POS = flags.DEFINE_string(
    "instruction_pos", "A_begin",
    "Position of instruction: before_Q, Q_begin, Q_end, or A_begin."
)

_SUBSET_SELECT_METHOD = flags.DEFINE_string(
    "subset_select_method", "random",
    "Method for selecting training subset: random, representative, least_confident, "
    "verbal_least_confident, confidence_weighted_representative, "
    "verbal_confidence_weighted_representative, IPOMP, or anchor_points."
)

_ANCHOR_NUM_SOURCE_MODELS = flags.DEFINE_integer(
    "anchor_num_source_models", None,
    "Anchor Points: Maximum number of diverse source models to use. "
    "If None, uses all available source models (excluding the scorer). "
    "Source models are actual small LMs (e.g., Qwen2.5-0.5B, Llama-3.2-1B, etc.). "
    "The scorer model is automatically excluded from source models."
)

_IPOMP_CORRELATION_THRESHOLD = flags.DEFINE_float(
    "ipomp_correlation_threshold", 0.9,
    "IPOMP: Correlation threshold for identifying redundant samples. "
    "Samples with performance correlation above this threshold are considered redundant. Default: 0.9."
)

_IPOMP_REPLACEMENT_RATIO = flags.DEFINE_float(
    "ipomp_replacement_ratio", 0.1,
    "IPOMP: Fraction of redundant samples to replace each iteration (beta in the paper). Default: 0.1."
)

_CONFIDENCE_WEIGHT = flags.DEFINE_float(
    "confidence_weight", 0.5,
    "Weight for confidence in confidence_weighted_representative method. "
    "0 = pure diversity, 1 = heavily weight hard examples. Default 0.5."
)

_ALPHA = flags.DEFINE_float(
    "alpha", 0.9,
    "Weight for dense (embedding) similarity vs lexical (TF-IDF) similarity. "
    "Used by representative and confidence_weighted_representative methods. "
    "alpha=1.0 means pure embedding, alpha=0.0 means pure TF-IDF. Default 0.9."
)

_EMBEDDING_MODEL = flags.DEFINE_string(
    "embedding_model", "Qwen/Qwen3-Embedding-8B",
    "Model for generating embeddings in representative subset selection. "
    "Used by representative and confidence_weighted_representative methods. "
    "Qwen3-Embedding-8B ranks #1 on MTEB multilingual leaderboard."
)

_SUBSET_PORTION = flags.DEFINE_float(
    "subset_portion", None,
    "Percentage of data for training (e.g., 3.5 for 3.5%)."
)

_NUM_SEARCH_STEPS = flags.DEFINE_integer(
    "num_search_steps", 100,
    "Number of OPRO optimization steps."
)

_NUM_CANDIDATES_PER_STEP = flags.DEFINE_integer(
    "num_candidates_per_step", 7,
    "Number of candidate prompts to generate per step (default 7, one per worker)."
)

_OPTIMIZER_TEMPERATURE = flags.DEFINE_float(
    "optimizer_temperature", 1.0,
    "Temperature for candidate generation."
)

_SCORER_MAX_TOKENS = flags.DEFINE_integer(
    "scorer_max_tokens", 1024,
    "Max tokens for scorer generation."
)

_GPU_MEMORY_UTILIZATION = flags.DEFINE_float(
    "gpu_memory_utilization", 0.90,
    "Fraction of GPU memory for vLLM."
)

_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint_interval", 5,
    "Steps between checkpoints."
)

_RESUME_FROM = flags.DEFINE_string(
    "resume_from", None,
    "Path to checkpoint file to resume from."
)

_SEED = flags.DEFINE_integer(
    "seed", 42,
    "Random seed for reproducibility."
)

_FEW_SHOT_SELECTION_CRITERIA = flags.DEFINE_string(
    "few_shot_selection_criteria", "random",
    "How to select few-shot examples for meta prompt: random, constant, or current_most_frequent."
)

_NUM_FEW_SHOT_EXAMPLES = flags.DEFINE_integer(
    "num_few_shot_examples", 3,
    "Number of few-shot examples to include in meta prompt."
)

_VERBOSE_EVAL_LOGGING = flags.DEFINE_boolean(
    "verbose_eval_logging", False,
    "Log detailed evaluation info for every prompt (saves to verbose_eval_log.json)."
)


def main(_):
    # Get flag values
    num_workers = _NUM_WORKERS.value
    scorer_model = _SCORER_MODEL.value
    use_local_optimizer = _USE_LOCAL_OPTIMIZER.value
    optimizer_model = _OPTIMIZER_MODEL.value
    optimizer_gpu_id = _OPTIMIZER_GPU_ID.value
    optimizer_quantization = _OPTIMIZER_QUANTIZATION.value
    if optimizer_quantization and optimizer_quantization.lower() == "none":
        optimizer_quantization = None
    dataset_name = _DATASET.value.lower()
    task_name = _TASK.value
    instruction_pos = _INSTRUCTION_POS.value
    subset_select_method = _SUBSET_SELECT_METHOD.value
    subset_portion = _SUBSET_PORTION.value
    confidence_weight = _CONFIDENCE_WEIGHT.value
    alpha = _ALPHA.value
    embedding_model = _EMBEDDING_MODEL.value
    num_search_steps = _NUM_SEARCH_STEPS.value
    num_candidates_per_step = _NUM_CANDIDATES_PER_STEP.value
    optimizer_temperature = _OPTIMIZER_TEMPERATURE.value
    scorer_max_tokens = _SCORER_MAX_TOKENS.value
    gpu_memory_utilization = _GPU_MEMORY_UTILIZATION.value
    checkpoint_interval = _CHECKPOINT_INTERVAL.value
    resume_from = _RESUME_FROM.value
    seed = _SEED.value
    few_shot_selection_criteria = _FEW_SHOT_SELECTION_CRITERIA.value
    num_few_shot_examples = _NUM_FEW_SHOT_EXAMPLES.value
    verbose_eval_logging = _VERBOSE_EVAL_LOGGING.value
    
    # API key validation only needed for OpenAI mode
    # For local optimizer, no API key needed
    
    # Set random seed
    np.random.seed(seed)
    
    # IPOMP-specific parameters
    ipomp_correlation_threshold = _IPOMP_CORRELATION_THRESHOLD.value
    ipomp_replacement_ratio = _IPOMP_REPLACEMENT_RATIO.value
    
    # Anchor Points specific parameters
    anchor_num_source_models = _ANCHOR_NUM_SOURCE_MODELS.value
    
    # Validate inputs
    assert dataset_name in {"gsm8k", "bbh", "mmlu", "math", "gpqa"}, f"Invalid dataset: {dataset_name}"
    assert instruction_pos in {"before_Q", "Q_begin", "Q_end", "A_begin"}
    # Validate subset selection method
    valid_methods = {
        "random", "representative", "least_confident", 
        "verbal_least_confident", "confidence_weighted_representative", 
        "verbal_confidence_weighted_representative", "IPOMP", "anchor_points",
        # Ablation methods (opposite/worst-case)
        "most_confident", "least_representative", "confidence_weighted_least_representative"
    }
    # Also allow random_{seed} format for ablation experiments
    is_random_seed_method = subset_select_method.startswith("random_") and subset_select_method.split("_")[1].isdigit()
    assert subset_select_method in valid_methods or is_random_seed_method, \
        f"Invalid subset_select_method: {subset_select_method}. Valid methods: {valid_methods} or random_{{seed}}"
    assert few_shot_selection_criteria in {"random", "constant", "current_most_frequent"}
    
    print(f"=" * 60)
    print(f"Multi-GPU OPRO Optimization")
    print(f"=" * 60)
    if use_local_optimizer:
        print(f"Optimizer: LOCAL vLLM on GPU {optimizer_gpu_id}")
        print(f"  Model: {optimizer_model}")
        print(f"  Quantization: {optimizer_quantization or 'None'}")
        print(f"Scorer Workers: {num_workers} (GPUs {optimizer_gpu_id + 1}-{optimizer_gpu_id + num_workers})")
    else:
        print(f"Optimizer: OpenAI API ({optimizer_model})")
        print(f"Scorer Workers: {num_workers} (GPUs 0-{num_workers - 1})")
    print(f"Scorer Model: {scorer_model}")
    print(f"Dataset: {dataset_name}/{task_name}")
    print(f"Steps: {num_search_steps}")
    print(f"Candidates per step: {num_candidates_per_step}")
    print(f"=" * 60)
    
    # ============== Load Data ==============
    
    if dataset_name == "mmlu":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "MMLU-data")
    elif dataset_name == "bbh":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "BIG-Bench-Hard-data/")
    elif dataset_name == "math":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "MATH-data")
    elif dataset_name == "gpqa":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "GPQA-data")
    else:
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "gsm_data")
    
    # Load raw data for training
    if dataset_name == "gsm8k":
        # For GSM8K, train and test are in separate files
        f_gsm_train = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
        raw_data = pd.read_csv(f_gsm_train, sep="\t", header=None)
        num_examples = raw_data.shape[0]
        
        # Load test data from gsm_test.tsv (following evaluate_instructions.py pattern)
        f_gsm_test = os.path.join(root_data_folder_path, "gsm_test.tsv")
        test_raw_data = pd.read_csv(f_gsm_test, sep="\t", header=None)
        num_test_examples = test_raw_data.shape[0]
        
        prediction_treat_as_number = True
        prediction_treat_as_bool = False
        is_multiple_choice = False
        
    elif dataset_name == "bbh":
        # For BBH, there's only one file per task
        # Following evaluate_instructions.py: train/test split on the same data
        raw_data = eval_utils.load_bbh_task_data(task_name, base_dir=root_data_folder_path)
        num_examples = len(raw_data)
        # Test data is the same, but we'll use different indices
        test_raw_data = raw_data
        num_test_examples = num_examples
        
        numerical_output_tasks = {"object_counting", "multistep_arithmetic_two"}
        boolean_tasks = {
            "boolean_expressions", "causal_judgement", "formal_fallacies",
            "navigate", "sports_understanding", "web_of_lies",
        }
        multiple_choice_tasks = {
            "date_understanding", "disambiguation_qa", "geometric_shapes",
            "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects",
            "logical_deduction_three_objects", "movie_recommendation", "penguins_in_a_table",
            "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
            "snarks", "temporal_sequences", "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
        }
        
        prediction_treat_as_number = task_name in numerical_output_tasks
        prediction_treat_as_bool = task_name in boolean_tasks
        is_multiple_choice = task_name in multiple_choice_tasks
        
    elif dataset_name == "math":
        # For MATH, train and test are in separate files (like GSM8K)
        raw_data = eval_utils.load_math_task_data("train", base_dir=root_data_folder_path)
        num_examples = len(raw_data)
        
        # Load test data from separate file
        test_raw_data = eval_utils.load_math_task_data("test", base_dir=root_data_folder_path)
        num_test_examples = len(test_raw_data)
        
        prediction_treat_as_number = True
        prediction_treat_as_bool = False
        is_multiple_choice = False
    
    elif dataset_name == "gpqa":
        # For GPQA, task_name is the subset: main, extended, or diamond
        # Default to diamond (the most challenging subset) if not specified
        valid_gpqa_subsets = {"main", "extended", "diamond"}
        if task_name not in valid_gpqa_subsets:
            print(f"Warning: task_name '{task_name}' not valid for GPQA. Using 'diamond' (hardest subset).")
            task_name = "diamond"
        
        # GPQA uses the same data for train and test (we sample 20% for training subset)
        raw_data = eval_utils.load_gpqa_task_data(task_name, base_dir=root_data_folder_path)
        num_examples = len(raw_data)
        
        # For GPQA, test is the entire dataset (same as raw_data)
        test_raw_data = raw_data
        num_test_examples = num_examples
        
        prediction_treat_as_number = False
        prediction_treat_as_bool = False
        is_multiple_choice = True
        
    else:  # mmlu
        # Load all MMLU tasks for the given category
        # This is simplified - see full implementation in optimize_instructions.py
        raise NotImplementedError("MMLU support in parallel mode coming soon")
    
    print(f"Loaded {num_examples} training examples from {dataset_name}/{task_name}")
    print(f"Loaded {num_test_examples} test examples")
    
    # ============== Split Data ==============
    
    if subset_portion is not None:
        train_ratio = subset_portion / 100.0
    else:
        if dataset_name == "gsm8k":
            train_ratio = 0.035  # 3.5% default
        elif dataset_name == "math":
            train_ratio = 0.035  # 3.5% default (~170 examples from 4866)
        elif dataset_name == "bbh":
            train_ratio = 0.2  # 20% default
        elif dataset_name == "gpqa":
            train_ratio = 0.2  # 20% default for GPQA
        else:
            train_ratio = 0.8  # 80% default for MMLU
    
    eval_ratio = 0  # No separate eval set for simplicity
    
    num_train = int(num_examples * train_ratio)
    print(f"Train ratio: {train_ratio} ({num_train} examples from training data)")
    
    # Select training indices from the training data
    # IMPORTANT: Use tensor_parallel_size=1 (single GPU) for subset selection
    # to avoid GPU state corruption that affects OPRO workers on GPUs 1-7.
    # This is slower but ensures clean GPU state for the main optimization.
    subset_tp_size = 1  # Use GPU 0 only for subset selection
    
    if subset_select_method == "random":
        train_index, eval_index = subset_selection.random_subset(
            num_examples, train_ratio, eval_ratio, seed=seed
        )
    elif subset_select_method.startswith("random_"):
        # Support random_{seed} format for ablation experiments
        # e.g., "random_42", "random_123", "random_2024"
        random_seed = int(subset_select_method.split("_")[1])
        print(f"Using random subset with explicit seed: {random_seed}")
        train_index, eval_index = subset_selection.random_subset(
            num_examples, train_ratio, eval_ratio, seed=random_seed
        )
    elif subset_select_method == "representative":
        train_index, eval_index = subset_selection.representative_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, 
            alpha=alpha,
            embedding_model=embedding_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    elif subset_select_method == "least_confident":
        train_index, eval_index = subset_selection.least_confident_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, scorer_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    elif subset_select_method == "verbal_least_confident":
        train_index, eval_index = subset_selection.verbal_least_confident_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, scorer_model, 
            k=4,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    elif subset_select_method == "confidence_weighted_representative":
        train_index, eval_index = subset_selection.confidence_weighted_representative_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, scorer_model,
            alpha=alpha, confidence_weight=confidence_weight,
            embedding_model=embedding_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    elif subset_select_method == "verbal_confidence_weighted_representative":
        # Uses VERBAL confidence (from "Just Ask for Calibration") instead of logit-based
        train_index, eval_index = subset_selection.verbal_confidence_weighted_representative_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, scorer_model,
            alpha=alpha, confidence_weight=confidence_weight,
            k=4,  # number of guesses for verbal confidence
            embedding_model=embedding_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    elif subset_select_method == "IPOMP":
        # IPOMP returns embeddings as well, needed for dynamic updates
        train_index, eval_index, ipomp_embeddings = subset_selection.ipomp_initial_subset(
            dataset_name, raw_data, train_ratio, eval_ratio,
            seed=seed,
            embedding_model=embedding_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            n_clusters=10,
        )
    elif subset_select_method == "anchor_points":
        # Anchor Points returns cluster sizes for APW scoring
        # Uses diverse small LMs as source models (scorer model is automatically excluded)
        # Models are processed in parallel batches using num_workers GPUs
        train_index, eval_index, anchor_cluster_sizes = subset_selection.anchor_points_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, scorer_model,
            seed=seed,
            num_source_models=anchor_num_source_models,
            num_workers=num_workers,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    # ============== ABLATION METHODS (opposite/worst-case selection) ==============
    elif subset_select_method == "most_confident":
        # ABLATION: Select EASIEST examples (opposite of least_confident)
        train_index, eval_index = subset_selection.most_confident_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, scorer_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    elif subset_select_method == "least_representative":
        # ABLATION: Select LEAST diverse examples (opposite of representative)
        train_index, eval_index = subset_selection.least_representative_subset(
            dataset_name, raw_data, train_ratio, eval_ratio,
            alpha=alpha,
            embedding_model=embedding_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    elif subset_select_method == "confidence_weighted_least_representative":
        # ABLATION: Select EASY + LEAST DIVERSE examples (opposite of confidence_weighted_representative)
        train_index, eval_index = subset_selection.confidence_weighted_least_representative_subset(
            dataset_name, raw_data, train_ratio, eval_ratio, scorer_model,
            alpha=alpha, confidence_weight=confidence_weight,
            embedding_model=embedding_model,
            tensor_parallel_size=subset_tp_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    else:
        raise ValueError(f"Unknown subset selection method: {subset_select_method}")
    
    # For GSM8K/MATH: test_index is ALL indices from test_raw_data (separate file)
    # For BBH: test_index is all indices not in train_index (same file)
    # For GPQA: test_index is ALL indices (entire dataset used for testing)
    if dataset_name in {"gsm8k", "math"}:
        # GSM8K and MATH use separate test file - use all indices
        test_index = np.arange(num_test_examples)
    elif dataset_name == "gpqa":
        # GPQA uses entire dataset for testing (20% subset for training signal)
        test_index = np.arange(num_test_examples)
    else:
        # BBH uses same file - test is complement of train
        all_indices = set(range(num_examples))
        test_index = np.array(sorted(all_indices - set(train_index)))
    
    print(f"Test set: {len(test_index)} examples")
    
    # ============== Create Save Directory ==============
    
    pacific_tz = ZoneInfo("America/Los_Angeles")
    datetime_str = datetime.datetime.now(pacific_tz).strftime("%Y-%m-%d-%H-%M")
    
    # Format percentage for folder name (e.g., "3.5" or "20")
    train_percentage_str = f"{train_ratio * 100:.10g}"
    
    # Format model names for folders (handle HuggingFace paths like "org/model-name")
    optimizer_short_name = optimizer_model.split("/")[-1]
    if use_local_optimizer:
        optimizer_short_name = f"local-{optimizer_short_name}"
    scorer_short_name = scorer_model.split("/")[-1]
    
    # Dataset folder name (include task if different from default)
    if dataset_name == "gpqa":
        dataset_folder = f"GPQA-{task_name}"  # e.g., GPQA-main, GPQA-diamond
    elif dataset_name == "bbh":
        dataset_folder = f"BBH-{task_name}"  # BBH has many tasks
    else:
        dataset_folder = dataset_name.upper()  # GSM8K, MATH, MMLU
    
    # New hierarchical structure:
    # parallel_{method}/{DATASET}/{percentage}/{optimizer}_optimizer/{scorer}_scorer/{timestamp}/
    save_folder = os.path.join(
        OPRO_ROOT_PATH,
        "outputs",
        "optimization-results",
        f"parallel_{subset_select_method}",
        dataset_folder,
        train_percentage_str,
        f"{optimizer_short_name}_optimizer",
        f"{scorer_short_name}_scorer",
        datetime_str,
    )
    os.makedirs(os.path.join(save_folder, "result_by_instruction"), exist_ok=True)
    print(f"Results will be saved to: {save_folder}")
    
    # Save config - include all command line arguments
    config = {
        # Worker/model settings
        "num_workers": num_workers,
        "scorer_model": scorer_model,
        "use_local_optimizer": use_local_optimizer,
        "optimizer_model": optimizer_model,
        "optimizer_gpu_id": optimizer_gpu_id,
        "optimizer_quantization": optimizer_quantization,
        "gpu_memory_utilization": gpu_memory_utilization,
        
        # Dataset settings
        "dataset_name": dataset_name,
        "task_name": task_name,
        
        # Subset selection settings
        "subset_select_method": subset_select_method,
        "subset_portion": subset_portion,
        "alpha": alpha,
        "confidence_weight": confidence_weight,
        "embedding_model": embedding_model,
        "ipomp_correlation_threshold": ipomp_correlation_threshold if subset_select_method == "IPOMP" else None,
        "ipomp_replacement_ratio": ipomp_replacement_ratio if subset_select_method == "IPOMP" else None,
        "anchor_num_source_models": anchor_num_source_models if subset_select_method == "anchor_points" else None,
        
        # Computed values
        "train_ratio": train_ratio,
        "num_train_examples": len(train_index),
        "num_test_examples": len(test_index),
        
        # Optimization settings
        "num_search_steps": num_search_steps,
        "num_candidates_per_step": num_candidates_per_step,
        "optimizer_temperature": optimizer_temperature,
        "scorer_max_tokens": scorer_max_tokens,
        "instruction_pos": instruction_pos,
        
        # Few-shot settings
        "few_shot_selection_criteria": few_shot_selection_criteria,
        "num_few_shot_examples": num_few_shot_examples,
        
        # Other settings
        "checkpoint_interval": checkpoint_interval,
        "resume_from": resume_from,
        "seed": seed,
    }
    
    import json
    with open(os.path.join(save_folder, "configs_dict.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # ============== Create and Run Controller ==============
    
    # Force garbage collection before starting OPRO
    # NOTE: We cannot call torch.cuda functions here because it would initialize
    # CUDA in the main process, preventing forked worker processes from using CUDA.
    # The workers will initialize CUDA themselves with CUDA_VISIBLE_DEVICES set.
    import gc
    gc.collect()
    print("[Main] Memory cleanup complete, starting OPRO")
    
    # Initial instructions to evaluate
    initial_instructions = [
        "Let's solve the problem.",
    ]
    
    controller = OPROController(
        num_workers=num_workers,
        scorer_model=scorer_model,
        scorer_max_tokens=scorer_max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        use_local_optimizer=use_local_optimizer,
        optimizer_model=optimizer_model,
        optimizer_temperature=optimizer_temperature,
        optimizer_gpu_id=optimizer_gpu_id,
        optimizer_quantization=optimizer_quantization,
        num_candidates_per_step=num_candidates_per_step,
        num_search_steps=num_search_steps,
        instruction_pos=instruction_pos,
        include_qa=True,  # Must be True when instruction_pos=A_begin (default)
        checkpoint_interval=checkpoint_interval,
        save_folder=save_folder,
        few_shot_selection_criteria=few_shot_selection_criteria,
        num_few_shot_examples=num_few_shot_examples,
        verbose_eval_logging=verbose_eval_logging,
    )
    
    # Load training dataset into controller
    controller.load_dataset(
        dataset_name=dataset_name,
        task_name=task_name,
        raw_data=raw_data,
        train_index=train_index,
        eval_index=eval_index,
        prediction_treat_as_number=prediction_treat_as_number,
        prediction_treat_as_bool=prediction_treat_as_bool,
        is_multiple_choice=is_multiple_choice,
    )
    
    # Load test dataset (separate file for GSM8K, same file with different indices for BBH)
    controller.load_test_dataset(
        test_raw_data=test_raw_data,
        test_index=test_index,
    )
    
    # Set up IPOMP manager if using IPOMP method
    if subset_select_method == "IPOMP":
        # Extract questions and answers for the full training data
        # Need these for IPOMP manager to handle dynamic subset updates
        from opro.optimization.subset_selection import IPOMPManager, _extract_questions_from_raw_data, _extract_short_answers_from_raw_data
        
        all_questions = _extract_questions_from_raw_data(dataset_name, raw_data)
        all_answers = _extract_short_answers_from_raw_data(dataset_name, raw_data)
        
        # Full training index = all available training examples
        full_train_index = np.arange(num_examples)
        
        ipomp_manager = IPOMPManager(
            embeddings=ipomp_embeddings,
            full_train_index=full_train_index,
            initial_train_index=train_index,
            questions=all_questions,
            answers=all_answers,
            correlation_threshold=ipomp_correlation_threshold,
            replacement_ratio=ipomp_replacement_ratio,
            seed=seed,
        )
        
        controller.set_ipomp_manager(ipomp_manager)
        print(f"[Main] IPOMP manager configured:")
        print(f"  - Correlation threshold: {ipomp_correlation_threshold}")
        print(f"  - Replacement ratio: {ipomp_replacement_ratio}")
    
    # Run optimization
    controller.run_optimization(
        initial_instructions=initial_instructions,
        resume_from_checkpoint=resume_from,
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    
    if controller.old_instructions_and_scores:
        best_instruction, best_score, best_step = max(
            controller.old_instructions_and_scores, key=lambda x: x[1]
        )
        print(f"Best score: {best_score:.4f} (found at step {best_step})")
        print(f"Best instruction:\n{best_instruction}")
    
    print(f"\nResults saved to: {save_folder}")


if __name__ == "__main__":
    app.run(main)

