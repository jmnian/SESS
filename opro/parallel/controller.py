"""OPRO Controller for multi-GPU parallel optimization.

The controller:
1. Loads evaluation dataset
2. Maintains OPRO state (instructions, scores, history)
3. Uses local vLLM optimizer (GPU 0) or OpenAI API to generate candidate prompts
4. Spawns and manages 7 worker processes (GPUs 1-7)
5. Dispatches candidate prompts to workers
6. Collects answers and computes scores

Architecture (8 GPUs):
- GPU 0: Local optimizer (e.g., gpt-oss-120b via vLLM)
- GPUs 1-7: Scorer workers (vLLM with scorer model)
"""

import atexit
import collections
import json
import os
import pickle
import signal
import sys
import threading
import time
# Import Queue from worker module to use spawn-compatible queue
from opro.parallel.worker import Queue
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add OPRO root to path
OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from opro.evaluation import eval_utils, metrics
from opro.optimization import opt_utils
from opro.optimization.subset_selection import IPOMPManager
from opro.parallel.worker import start_worker
from opro.parallel.openai_optimizer import LLMOptimizer
from opro.parallel.vllm_optimizer import VLLMOptimizer


class OPROController:
    """Controller for multi-GPU OPRO optimization.
    
    Manages the OPRO loop with:
    - 7 vLLM workers (GPUs 1-7) for scoring
    - Local vLLM optimizer (GPU 0) or OpenAI API for generating new candidates
    """
    
    def __init__(
        self,
        # Worker configuration
        num_workers: int = 7,  # Default 7 workers on GPUs 1-7
        scorer_model: str = "Qwen/Qwen2.5-7B-Instruct",
        scorer_max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.90,
        use_chat_mode: bool = True,  # Chat mode: natural EOS stopping, OPRO has full control
        # Optimizer configuration
        use_local_optimizer: bool = True,  # Use local vLLM optimizer by default
        optimizer_model: str = "openai/gpt-oss-120b",  # Local model or OpenAI model
        optimizer_temperature: float = 1.0,
        optimizer_max_tokens: int = 512,
        optimizer_gpu_id: int = 0,  # GPU for local optimizer
        optimizer_quantization: Optional[str] = "fp8",  # Quantization for local optimizer
        openai_api_key: Optional[str] = None,  # Only used if use_local_optimizer=False
        # OPRO configuration
        num_candidates_per_step: int = 7,  # Default 7 candidates (one per worker)
        num_search_steps: int = 100,
        instruction_pos: str = "A_begin",
        include_qa: bool = True,  # Must be True when instruction_pos=A_begin
        old_instruction_score_threshold: float = 0.2,
        max_num_instructions: int = 20,
        num_score_buckets: int = 100,
        # Checkpointing
        checkpoint_interval: int = 5,
        save_folder: Optional[str] = None,
        # Few-shot selection
        few_shot_selection_criteria: str = "random",
        num_few_shot_examples: int = 3,
        # Debug logging
        verbose_eval_logging: bool = False,  # Log every eval detail to file
    ):
        """Initialize the OPRO controller.
        
        Args:
            num_workers: Number of GPU workers to spawn (default 7, using GPUs 1-7)
            scorer_model: Model for scoring (vLLM workers)
            scorer_max_tokens: Max tokens for scorer
            gpu_memory_utilization: vLLM GPU memory fraction
            use_chat_mode: If True (default), use vLLM's chat() API with no system
                          prompt. This provides: 1) natural EOS stopping - model stops
                          cleanly without runaway generation, 2) full OPRO control -
                          no hidden system prompt constraining optimization. Note:
                          "output only" style instructions may perform poorly, but
                          OPRO will naturally learn to avoid them.
            use_local_optimizer: If True, use local vLLM optimizer on GPU 0;
                                 if False, use OpenAI API
            optimizer_model: Model for optimization. For local: HuggingFace path 
                            (e.g., openai/gpt-oss-120b). For OpenAI: model name (e.g., gpt-4o)
            optimizer_temperature: Temperature for candidate generation
            optimizer_max_tokens: Max tokens for optimizer
            optimizer_gpu_id: GPU ID for local optimizer (default 0)
            optimizer_quantization: Quantization method for local optimizer (fp8, awq, gptq)
            openai_api_key: API key for OpenAI (only used if use_local_optimizer=False)
            num_candidates_per_step: Candidates to generate per step (default 7)
            num_search_steps: Total OPRO steps to run
            instruction_pos: Where to place instruction in prompt
            include_qa: Whether to use Q:/A: format
            old_instruction_score_threshold: Min score to include in meta-prompt
            max_num_instructions: Max instructions in meta-prompt
            num_score_buckets: Score quantization buckets
            checkpoint_interval: Steps between checkpoints
            save_folder: Directory for saving results
        """
        self.num_workers = num_workers
        self.scorer_model = scorer_model
        self.scorer_max_tokens = scorer_max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_chat_mode = use_chat_mode
        
        self.use_local_optimizer = use_local_optimizer
        self.optimizer_model = optimizer_model
        self.optimizer_temperature = optimizer_temperature
        self.optimizer_max_tokens = optimizer_max_tokens
        self.optimizer_gpu_id = optimizer_gpu_id
        self.optimizer_quantization = optimizer_quantization
        
        self.num_candidates_per_step = num_candidates_per_step
        self.num_search_steps = num_search_steps
        self.instruction_pos = instruction_pos
        self.include_qa = include_qa
        
        # Validate instruction_pos and include_qa compatibility
        if not include_qa and instruction_pos not in {"Q_begin", "Q_end"}:
            raise ValueError(
                f"When include_qa=False, instruction_pos must be 'Q_begin' or 'Q_end', "
                f"got '{instruction_pos}'"
            )
        self.old_instruction_score_threshold = old_instruction_score_threshold
        self.max_num_instructions = max_num_instructions
        self.num_score_buckets = num_score_buckets
        
        self.checkpoint_interval = checkpoint_interval
        self.save_folder = save_folder
        
        # Few-shot selection parameters
        self.few_shot_selection_criteria = few_shot_selection_criteria
        self.num_few_shot_examples = num_few_shot_examples
        self.current_step = 0  # Track current step for few-shot selection
        
        # Training evaluation sampling for debugging
        self.total_prompts_evaluated = 0  # Counter for prompts evaluated
        self.training_eval_samples = []  # Sampled evaluations for debugging
        self.training_sample_interval = 20  # Sample every N prompts
        
        # IPOMP dynamic subset manager (set via set_ipomp_manager)
        self.ipomp_manager = None
        self.use_ipomp = False
        
        # Verbose evaluation logging (logs every single evaluation)
        self.verbose_eval_logging = verbose_eval_logging
        self.verbose_eval_log = []  # Stores all detailed evaluations
        
        # Initialize optimizer based on mode
        # Note: Local optimizer initialization is deferred to start_workers()
        # to ensure proper GPU assignment after workers start
        self.optimizer = None
        if not use_local_optimizer:
            # Use OpenAI API optimizer (does not load any LLM)
            self.optimizer = LLMOptimizer(
                model=optimizer_model,
                max_tokens=optimizer_max_tokens,
                temperature=optimizer_temperature,
                num_candidates=num_candidates_per_step,
            )
        
        # Worker management
        self.workers = []
        self.input_queues = []
        self.output_queues = []
        
        # OPRO state
        self.old_instructions_and_scores = []
        self.old_instructions_and_scores_raw = []
        self.instruction_score_dict = {}
        self.old_instruction_md5_hashstrings_set = set()
        self.meta_prompts = []
        self.eval_results = []
        self.per_question_accuracies = {}  # instruction -> {idx: accuracy}
        
        # Data
        self.raw_data = None
        self.train_index = None
        self.eval_index = None
        self.dataset_name = None
        self.task_name = None
        self.prediction_treat_as_number = False
        self.prediction_treat_as_bool = False
        self.is_multiple_choice = False
        
    def load_dataset(
        self,
        dataset_name: str,
        task_name: str,
        raw_data,
        train_index: np.ndarray,
        eval_index: np.ndarray,
        prediction_treat_as_number: bool = False,
        prediction_treat_as_bool: bool = False,
        is_multiple_choice: bool = False,
    ):
        """Load and configure the evaluation dataset.
        
        Args:
            dataset_name: Name of dataset (gsm8k, bbh, mmlu)
            task_name: Task within dataset
            raw_data: Raw data (DataFrame or list)
            train_index: Indices for training evaluation
            eval_index: Indices for validation evaluation
            prediction_treat_as_number: Whether to treat predictions as numbers
            prediction_treat_as_bool: Whether to treat predictions as boolean
            is_multiple_choice: Whether task is multiple choice
        """
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.raw_data = raw_data
        self.train_index = train_index
        self.eval_index = eval_index
        self.prediction_treat_as_number = prediction_treat_as_number
        self.prediction_treat_as_bool = prediction_treat_as_bool
        self.is_multiple_choice = is_multiple_choice
        
        # Set decimal places based on dataset (MATH/GSM8K can have decimal answers)
        # GPQA is multiple choice (A/B/C/D), no decimals needed
        if dataset_name in {"math", "gsm8k"}:
            self.num_decimals = 4  # Preserve decimals like 0.3, 0.25, etc.
        else:
            self.num_decimals = 0
        
        # Pre-extract questions and answers for training
        self.train_questions = []
        self.train_answers = []
        for idx in train_index:
            q = self._get_question(idx)
            a = eval_utils.fetch_true_answer(raw_data, idx, dataset_name)
            self.train_questions.append(q)
            self.train_answers.append(a)
        
        print(f"[Controller] Loaded {len(self.train_questions)} training questions")
        
        # Initialize test data placeholders - loaded separately via load_test_dataset()
        self.test_raw_data = None
        self.test_index = None
        self.test_questions = []
        self.test_answers = []
    
    def set_ipomp_manager(self, ipomp_manager: IPOMPManager):
        """Set the IPOMP manager for dynamic subset updates.
        
        Args:
            ipomp_manager: IPOMPManager instance initialized with embeddings and initial subset
        """
        self.ipomp_manager = ipomp_manager
        self.use_ipomp = True
        print(f"[Controller] IPOMP manager set - dynamic subset updates enabled")
        print(f"[Controller] Initial subset size: {len(ipomp_manager.current_train_index)}")
    
    def _update_training_data_from_ipomp(self):
        """Update training questions and answers from IPOMP manager's current subset."""
        if not self.use_ipomp or self.ipomp_manager is None:
            return
        
        # Get the current training index from IPOMP manager
        self.train_index = self.ipomp_manager.get_current_train_index()
        
        # Re-extract questions and answers for the new subset
        self.train_questions = []
        self.train_answers = []
        for idx in self.train_index:
            q = self._get_question(idx)
            a = eval_utils.fetch_true_answer(self.raw_data, idx, self.dataset_name)
            self.train_questions.append(q)
            self.train_answers.append(a)
        
        print(f"[Controller] Updated training data from IPOMP: {len(self.train_questions)} questions")
    
    def load_test_dataset(
        self,
        test_raw_data,
        test_index: Optional[np.ndarray] = None,
    ):
        """Load the test dataset for final evaluation.
        
        For GSM8K, this should be data from gsm_test.tsv.
        For BBH, this can be the same data with different indices.
        
        Args:
            test_raw_data: Raw test data (DataFrame or list)
            test_index: Indices for test evaluation (if None, uses all)
        """
        self.test_raw_data = test_raw_data
        
        # Determine test index
        if test_index is not None:
            self.test_index = test_index
        else:
            # Use all data in test_raw_data
            if self.dataset_name in {"gsm8k", "mmlu"}:
                self.test_index = np.arange(test_raw_data.shape[0])
            else:  # bbh, math (list-based datasets)
                self.test_index = np.arange(len(test_raw_data))
        
        # Pre-extract questions and answers for test set
        self.test_questions = []
        self.test_answers = []
        for idx in self.test_index:
            q = self._get_question_from_data(test_raw_data, idx)
            a = eval_utils.fetch_true_answer(test_raw_data, idx, self.dataset_name)
            self.test_questions.append(q)
            self.test_answers.append(a)
        
        print(f"[Controller] Loaded {len(self.test_questions)} test questions")
    
    def _get_question_from_data(self, data, idx: int) -> str:
        """Get question text at given index from specified data."""
        if self.dataset_name == "gsm8k":
            return data.iloc[idx, 0]
        elif self.dataset_name in {"bbh", "math"}:
            return data[idx]["input"]
        elif self.dataset_name == "mmlu":
            return eval_utils._format_mmlu_example(data, idx)
        elif self.dataset_name == "gpqa":
            return data[idx]["input"]  # Contains formatted MC question
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
    def _get_question(self, idx: int) -> str:
        """Get question text at given index."""
        if self.dataset_name == "gsm8k":
            return self.raw_data.iloc[idx, 0]
        elif self.dataset_name in {"bbh", "math"}:
            return self.raw_data[idx]["input"]
        elif self.dataset_name == "mmlu":
            return eval_utils._format_mmlu_example(self.raw_data, idx)
        elif self.dataset_name == "gpqa":
            return self.raw_data[idx]["input"]  # Contains formatted MC question
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def start_workers(self):
        """Start all worker processes and initialize local optimizer if needed."""
        # Determine GPU assignment
        # If using local optimizer: GPU 0 = optimizer, GPUs 1-N = workers
        # If using OpenAI API: GPUs 0-(N-1) = workers
        if self.use_local_optimizer:
            worker_gpu_start = self.optimizer_gpu_id + 1  # Workers start after optimizer GPU
            print(f"[Controller] GPU {self.optimizer_gpu_id}: Local optimizer ({self.optimizer_model})")
            print(f"[Controller] GPUs {worker_gpu_start}-{worker_gpu_start + self.num_workers - 1}: {self.num_workers} scorer workers")
        else:
            worker_gpu_start = 0
            print(f"[Controller] Using OpenAI API for optimization")
            print(f"[Controller] GPUs 0-{self.num_workers - 1}: {self.num_workers} scorer workers")
        
        print(f"[Controller] Starting {self.num_workers} workers...")
        
        # Register cleanup handler to ensure workers are terminated on exit
        atexit.register(self._force_cleanup)
        
        for i in range(self.num_workers):
            gpu_id = worker_gpu_start + i
            input_q = Queue()
            output_q = Queue()
            
            worker = start_worker(
                gpu_id=gpu_id,
                input_queue=input_q,
                output_queue=output_q,
                model_name=self.scorer_model,
                max_tokens=self.scorer_max_tokens,
                gpu_memory_utilization=self.gpu_memory_utilization,
                use_chat_mode=self.use_chat_mode,
            )
            
            self.workers.append(worker)
            self.input_queues.append(input_q)
            self.output_queues.append(output_q)
            
            # Stagger worker starts to avoid port conflicts in vLLM/torch.distributed
            # Each vLLM instance needs time to allocate its network ports before
            # the next one starts
            if i < self.num_workers - 1:
                time.sleep(2.0)  # Wait 2 seconds between worker starts
        
        # Wait for all workers to be ready
        ready_count = 0
        timeout = 600  # 10 minutes for model loading
        start_time = time.time()
        ready_workers = set()
        
        while ready_count < self.num_workers:
            if time.time() - start_time > timeout:
                raise RuntimeError("Timeout waiting for workers to initialize")
            
            # Check if any workers have died
            dead_workers = [i for i, w in enumerate(self.workers) if not w.is_alive() and i not in ready_workers]
            if dead_workers:
                raise RuntimeError(
                    f"Worker(s) {dead_workers} died during initialization. "
                    "Check if all dependencies are installed (e.g., vllm)."
                )
            
            for i, output_q in enumerate(self.output_queues):
                try:
                    msg = output_q.get(timeout=1)
                    if msg.get("type") == "ready":
                        ready_count += 1
                        ready_workers.add(msg['worker_id'])
                        print(f"[Controller] Worker {msg['worker_id']} ready ({ready_count}/{self.num_workers})")
                    elif msg.get("type") == "error" and msg.get("fatal"):
                        raise RuntimeError(f"Worker {i} failed: {msg['error']}")
                except Exception:
                    pass
        
        print(f"[Controller] All {self.num_workers} workers ready")
        
        # Initialize local optimizer after workers are ready
        # This ensures GPU 0 is reserved for the optimizer
        if self.use_local_optimizer:
            print(f"[Controller] Initializing local optimizer on GPU {self.optimizer_gpu_id}...")
            # Use higher memory utilization (0.95) and reduced max_model_len (16K) 
            # for large models like gpt-oss-120b to fit on single A100
            self.optimizer = VLLMOptimizer(
                model=self.optimizer_model,
                max_tokens=self.optimizer_max_tokens,
                temperature=self.optimizer_temperature,
                num_candidates=self.num_candidates_per_step,
                gpu_id=self.optimizer_gpu_id,
                gpu_memory_utilization=0.95,  # Higher than scorers for large optimizer models
                max_model_len=16384,  # 16K is plenty for OPRO meta prompts
                quantization=self.optimizer_quantization,
            )
            self.optimizer.initialize()
            print(f"[Controller] Local optimizer ready")
    
    def stop_workers(self):
        """Stop all worker processes and shutdown local optimizer."""
        print("[Controller] Stopping workers...")
        
        # Only send shutdown to workers that are still alive
        for i, (input_q, worker) in enumerate(zip(self.input_queues, self.workers)):
            if worker.is_alive():
                try:
                    input_q.put({"type": "shutdown"}, timeout=1)
                except Exception:
                    pass
        
        # Wait briefly for graceful shutdown, then terminate
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                print(f"[Controller] Force terminating worker...")
                worker.terminate()
                worker.join(timeout=2)
        
        # Clean up queues - drain any remaining messages to prevent hangs
        for output_q in self.output_queues:
            try:
                while not output_q.empty():
                    output_q.get_nowait()
            except Exception:
                pass
        
        for input_q in self.input_queues:
            try:
                while not input_q.empty():
                    input_q.get_nowait()
            except Exception:
                pass
        
        self.workers = []
        self.input_queues = []
        self.output_queues = []
        
        print("[Controller] All workers stopped")
        
        # Shutdown local optimizer if used
        if self.use_local_optimizer and self.optimizer is not None:
            print("[Controller] Shutting down local optimizer...")
            self.optimizer.shutdown()
            print("[Controller] Local optimizer stopped")
    
    def _force_cleanup(self):
        """Force cleanup of any remaining workers and optimizer. Called by atexit."""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1)
        
        # Cleanup local optimizer
        if self.use_local_optimizer and self.optimizer is not None:
            try:
                self.optimizer.shutdown()
            except Exception:
                pass
    
    def evaluate_instruction(
        self,
        instruction: str,
        worker_id: Optional[int] = None,
    ) -> tuple[float, pd.DataFrame]:
        """Evaluate a single instruction on the training set.
        
        Args:
            instruction: The instruction/prompt to evaluate
            worker_id: Specific worker to use (optional, uses next available)
            
        Returns:
            Tuple of (average_score, detailed_results_dataframe)
        """
        if worker_id is None:
            worker_id = 0  # Default to first worker for single evaluation
        
        # Send evaluation request
        self.input_queues[worker_id].put({
            "type": "evaluate",
            "candidate_id": 0,
            "candidate_prompt": instruction,
            "questions": self.train_questions,
            "instruction_pos": self.instruction_pos,
            "include_qa": self.include_qa,
        })
        
        # Wait for result
        result = self.output_queues[worker_id].get()
        
        if result.get("type") == "error":
            raise RuntimeError(f"Worker error: {result.get('error')}")
        
        answers = result["answers"]
        
        # CRITICAL: Validate that the result matches what we sent
        returned_prompt = result.get("candidate_prompt", "")
        if returned_prompt != instruction:
            print(f"[Controller] FATAL: Training eval result mismatch!")
            print(f"  Expected: {instruction[:80]}...")
            print(f"  Got: {returned_prompt[:80]}...")
            self.stop_workers()
            raise RuntimeError("Result prompt mismatch - stopping to prevent corrupted results")
        
        # Validate answer count matches question count
        if len(answers) != len(self.train_questions):
            print(f"[Controller] FATAL: Training eval answer count mismatch!")
            print(f"  Expected {len(self.train_questions)} answers, got {len(answers)}")
            self.stop_workers()
            raise RuntimeError("Answer count mismatch - stopping to prevent corrupted results")
        
        # Compute scores
        accuracies = []
        preds = []
        # debug_count = 0
        for i, (pred, true_ans) in enumerate(zip(answers, self.train_answers)):
            # Parse prediction
            parsed_pred = metrics.get_normalized_prediction(
                pred,
                treat_as_number=self.prediction_treat_as_number,
                num_decimals=self.num_decimals,
                treat_as_bool=self.prediction_treat_as_bool,
                treat_as_multiple_choice=self.is_multiple_choice,
            )
            preds.append(parsed_pred)

            # Get accuracy
            if self.is_multiple_choice:
                input_text = self.train_questions[i]
            else:
                input_text = ""
            
            # For multiple choice, don't use "include" fallback - single letters 
            # like 'b' appear in words like "because", causing false positives.
            # Only use include fallback for free-form text answers (e.g., city names).
            treat_include_as_correct = (
                not self.prediction_treat_as_number and not self.is_multiple_choice
            )
            
            accuracy = eval_utils._get_accuracy(
                true_answer=true_ans,
                pred_answer=parsed_pred,
                input_text=input_text,
                treat_include_as_correct=treat_include_as_correct,
            )
            accuracies.append(accuracy)
            
            # DEBUG: Print first 20 examples
            # if debug_count < 20:
            #     print(f"\n{'='*60}")
            #     print(f"[DEBUG] Example {debug_count + 1}")
            #     print(f"{'='*60}")
            #     print(f"QUESTION: {self.train_questions[i]}")
            #     print(f"\nRAW OUTPUT: {pred}")
            #     print(f"\nPARSED ANSWER: {parsed_pred}")
            #     print(f"TRUE ANSWER: {true_ans}")
            #     print(f"ACCURACY: {accuracy}")
            #     print(f"{'='*60}")
            #     debug_count += 1
                
            #     if debug_count >= 20:
            #         print("\n[DEBUG] Exiting after 20 examples for debugging...")
            #         sys.exit(0)
        
        average_score = np.mean(accuracies)
        
        # Create detailed results DataFrame
        # Build raw_prompts to match worker.py's _build_prompt format exactly
        raw_prompts = []
        for q in self.train_questions:
            if self.include_qa:
                if self.instruction_pos == "before_Q":
                    prompt = f"{instruction}\nQ: {q}\n\nA:"
                elif self.instruction_pos == "Q_begin":
                    prompt = f"Q: {instruction}\n{q}\n\nA:"
                elif self.instruction_pos == "Q_end":
                    prompt = f"Q: {q}\n{instruction}\n\nA:"
                else:  # A_begin
                    prompt = f"Q: {q}\n\nA: {instruction}"
            else:
                if self.instruction_pos == "Q_begin":
                    prompt = f"{instruction}\n{q}"
                else:  # Q_end
                    prompt = f"{q}\n{instruction}"
            raw_prompts.append(prompt)
        
        detailed_results_df = pd.DataFrame({
            "index_in_raw_dataset": self.train_index,
            "raw_prompt": raw_prompts,
            "raw_answer": answers,
            "parsed_answer": [
                metrics.get_normalized_prediction(
                    a, 
                    treat_as_number=self.prediction_treat_as_number,
                    num_decimals=self.num_decimals,
                    treat_as_bool=self.prediction_treat_as_bool,
                    treat_as_multiple_choice=self.is_multiple_choice,
                )
                for a in answers
            ],
            "true_answer": self.train_answers,
            "accuracy": accuracies,
        })
        detailed_results_df.set_index("index_in_raw_dataset", inplace=True)
        
        return average_score, detailed_results_df
    
    def evaluate_candidates_parallel(
        self,
        candidates: list[str],
    ) -> list[tuple[str, float, pd.DataFrame]]:
        """Evaluate multiple candidates in parallel across workers.
        
        Args:
            candidates: List of candidate instructions to evaluate
            
        Returns:
            List of (instruction, score, detailed_results) tuples
        """
        # Dispatch candidates to workers
        active_requests = {}  # worker_id -> candidate_info
        results = []
        candidate_queue = list(enumerate(candidates))
        
        # Initially dispatch to all available workers
        for worker_id in range(min(len(candidates), self.num_workers)):
            if candidate_queue:
                cand_idx, candidate = candidate_queue.pop(0)
                self.input_queues[worker_id].put({
                    "type": "evaluate",
                    "candidate_id": cand_idx,
                    "candidate_prompt": candidate,
                    "questions": self.train_questions,
                    "instruction_pos": self.instruction_pos,
                    "include_qa": self.include_qa,
                })
                active_requests[worker_id] = (cand_idx, candidate)
        
        # Collect results and dispatch remaining
        while active_requests:
            for worker_id in list(active_requests.keys()):
                try:
                    result = self.output_queues[worker_id].get(timeout=0.1)
                except:
                    continue
                
                if result.get("type") == "error":
                    print(f"[Controller] Worker {worker_id} error: {result.get('error')}")
                    cand_idx, candidate = active_requests.pop(worker_id)
                    results.append((candidate, np.nan, None, {}))
                elif result.get("type") == "result":
                    cand_idx, candidate = active_requests.pop(worker_id)
                answers = result["answers"]
                
                # CRITICAL: Validate that the result matches what we sent
                returned_prompt = result.get("candidate_prompt", "")
                if returned_prompt != candidate:
                    print(f"[Controller] FATAL: Parallel eval result mismatch!")
                    print(f"  Expected: {candidate[:80]}...")
                    print(f"  Got: {returned_prompt[:80]}...")
                    self.stop_workers()
                    raise RuntimeError("Result prompt mismatch - stopping to prevent corrupted results")
                
                # Validate answer count matches question count
                if len(answers) != len(self.train_questions):
                    print(f"[Controller] FATAL: Parallel eval answer count mismatch!")
                    print(f"  Expected {len(self.train_questions)} answers, got {len(answers)}")
                    self.stop_workers()
                    raise RuntimeError("Answer count mismatch - stopping to prevent corrupted results")
                
                # Increment prompt counter
                self.total_prompts_evaluated += 1
                should_sample = (self.total_prompts_evaluated % self.training_sample_interval == 0)
                
                # Compute scores
                accuracies = []
                sampled_details = []  # For periodic sampling
                
                for i, (pred, true_ans) in enumerate(zip(answers, self.train_answers)):
                    parsed_pred = metrics.get_normalized_prediction(
                        pred,
                        treat_as_number=self.prediction_treat_as_number,
                        num_decimals=self.num_decimals,
                        treat_as_bool=self.prediction_treat_as_bool,
                        treat_as_multiple_choice=self.is_multiple_choice,
                    )
                    
                    if self.is_multiple_choice:
                        input_text = self.train_questions[i]
                    else:
                        input_text = ""
                    
                    # For multiple choice, don't use "include" fallback - single letters 
                    # like 'b' appear in words like "because", causing false positives.
                    treat_include_as_correct = (
                        not self.prediction_treat_as_number and not self.is_multiple_choice
                    )
                    
                    accuracy = eval_utils._get_accuracy(
                        true_answer=true_ans,
                        pred_answer=parsed_pred,
                        input_text=input_text,
                        treat_include_as_correct=treat_include_as_correct,
                    )
                    accuracies.append(accuracy)
                    
                    # Log all questions for this prompt if it's a sample interval
                    if should_sample:
                        question = self.train_questions[i]
                        # Format must match worker.py's _build_prompt exactly
                        if self.include_qa:
                            if self.instruction_pos == "before_Q":
                                full_input = f"{candidate}\nQ: {question}\n\nA:"
                            elif self.instruction_pos == "Q_begin":
                                full_input = f"Q: {candidate}\n{question}\n\nA:"
                            elif self.instruction_pos == "Q_end":
                                full_input = f"Q: {question}\n{candidate}\n\nA:"
                            else:  # A_begin
                                full_input = f"Q: {question}\n\nA: {candidate}"
                        else:
                            if self.instruction_pos == "Q_begin":
                                full_input = f"{candidate}\n{question}"
                            else:  # Q_end
                                full_input = f"{question}\n{candidate}"
                        
                        # Get parsing diagnostic
                        _, parse_diag = metrics._extract_multiple_choice_answer(
                            pred, return_diagnostic=True
                        ) if self.is_multiple_choice else (None, {"method": "not_mc"})
                        
                        sampled_details.append({
                            "question_idx": i,
                            "full_input_prompt": full_input,
                            "question_full": question,
                            "true_answer": str(true_ans),
                            "raw_output_full": pred,
                            "raw_output_length": len(pred),
                            "parsed_prediction": parsed_pred,
                            "parse_method": parse_diag.get("method"),
                            "parse_matched_text": parse_diag.get("matched_text"),
                            "correct": accuracy == 1,
                        })
                
                average_score = np.mean(accuracies)
                
                # Verbose evaluation logging - save EVERY evaluation detail
                if self.verbose_eval_logging:
                    eval_details = []
                    for i, (pred, true_ans) in enumerate(zip(answers, self.train_answers)):
                        question = self.train_questions[i]
                        
                        # Build the exact input prompt (matches _format_chat_messages or _format_prompt)
                        if self.use_chat_mode:
                            # Chat mode format
                            if self.include_qa:
                                if self.instruction_pos == "before_Q":
                                    user_content = f"{candidate}\n\nQuestion: {question}" if candidate else f"Question: {question}"
                                elif self.instruction_pos == "Q_begin":
                                    user_content = f"{candidate}\n{question}" if candidate else question
                                elif self.instruction_pos == "Q_end":
                                    user_content = f"{question}\n{candidate}" if candidate else question
                                else:  # A_begin
                                    user_content = f"{question}\n\nPlease respond starting with: {candidate}" if candidate else question
                            else:
                                if self.instruction_pos == "Q_begin":
                                    user_content = f"{candidate}\n{question}" if candidate else question
                                else:
                                    user_content = f"{question}\n{candidate}" if candidate else question
                            full_input = f"[CHAT MODE - User message]\n{user_content}"
                        else:
                            # Completion mode format
                            if self.include_qa:
                                if self.instruction_pos == "before_Q":
                                    full_input = f"{candidate}\nQ: {question}\n\nA:"
                                elif self.instruction_pos == "Q_begin":
                                    full_input = f"Q: {candidate}\n{question}\n\nA:"
                                elif self.instruction_pos == "Q_end":
                                    full_input = f"Q: {question}\n{candidate}\n\nA:"
                                else:
                                    full_input = f"Q: {question}\n\nA: {candidate}"
                            else:
                                if self.instruction_pos == "Q_begin":
                                    full_input = f"{candidate}\n{question}"
                                else:
                                    full_input = f"{question}\n{candidate}"
                        
                        # Re-parse to get the same result
                        parsed = metrics.get_normalized_prediction(
                            pred,
                            treat_as_number=self.prediction_treat_as_number,
                            num_decimals=self.num_decimals,
                            treat_as_bool=self.prediction_treat_as_bool,
                            treat_as_multiple_choice=self.is_multiple_choice,
                        )
                        
                        eval_details.append({
                            "question_idx": i,
                            "full_input_prompt": full_input,
                            "question": question[:200] + "..." if len(question) > 200 else question,
                            "true_answer": str(true_ans),
                            "raw_output": pred,
                            "raw_output_length": len(pred),
                            "parsed_prediction": parsed,
                            "correct": accuracies[i] == 1,
                        })
                    
                    verbose_entry = {
                        "step": self.current_step,
                        "instruction": candidate,
                        "overall_score": average_score,
                        "num_correct": int(sum(accuracies)),
                        "num_total": len(accuracies),
                        "use_chat_mode": self.use_chat_mode,
                        "evaluations": eval_details,
                    }
                    self.verbose_eval_log.append(verbose_entry)
                    
                    # Write to file every 10 evaluations (batched for performance)
                    if self.save_folder and len(self.verbose_eval_log) % 10 == 0:
                        verbose_log_path = os.path.join(self.save_folder, "verbose_eval_log.json")
                        with open(verbose_log_path, "w") as f:
                            json.dump(self.verbose_eval_log, f, indent=2)
                
                # Save sampled evaluation for debugging
                if should_sample and sampled_details:
                    self.training_eval_samples.append({
                        "prompt_number": self.total_prompts_evaluated,
                        "step": self.current_step,
                        "instruction": candidate,
                        "overall_score": average_score,
                        "num_correct": sum(accuracies),
                        "num_total": len(accuracies),
                        "sampled_questions": sampled_details,
                    })
                    print(f"[Controller] Sampled training eval #{self.total_prompts_evaluated}")
                    
                    # Periodically save samples to file
                    if self.save_folder and len(self.training_eval_samples) % 5 == 0:
                        self._save_training_samples()
                
                # Create detailed results
                detailed_results_df = pd.DataFrame({
                    "index_in_raw_dataset": self.train_index,
                    "raw_answer": answers,
                    "accuracy": accuracies,
                })
                detailed_results_df.set_index("index_in_raw_dataset", inplace=True)
                
                # Create per-sample accuracy dict for IPOMP (position -> accuracy)
                per_sample_acc = {i: acc for i, acc in enumerate(accuracies)}
                
                results.append((candidate, average_score, detailed_results_df, per_sample_acc))
                print(f"[Controller] Candidate '{candidate[:50]}...' score: {average_score:.4f}")
                
                # Dispatch next candidate if available
                if candidate_queue:
                    next_idx, next_candidate = candidate_queue.pop(0)
                    self.input_queues[worker_id].put({
                        "type": "evaluate",
                        "candidate_id": next_idx,
                        "candidate_prompt": next_candidate,
                        "questions": self.train_questions,
                        "instruction_pos": self.instruction_pos,
                        "include_qa": self.include_qa,
                    })
                    active_requests[worker_id] = (next_idx, next_candidate)
        
        return results
    
    def _select_few_shot_indices(self) -> list:
        """Select few-shot example indices based on the selection criteria."""
        n = self.num_few_shot_examples
        
        if self.few_shot_selection_criteria == "constant":
            # Same examples every step (fixed seed)
            np.random.seed(0)
            indices = np.sort(
                np.random.choice(self.train_index, n, replace=False)
            ).tolist()
        elif self.few_shot_selection_criteria == "current_most_frequent":
            # Select examples that are most frequently wrong across current instructions
            # This requires tracking per-question accuracies
            if hasattr(self, 'per_question_accuracies') and self.per_question_accuracies:
                # Count how many times each question was answered wrong
                wrong_counts = {}
                for ins_data in self.per_question_accuracies.values():
                    for idx, acc in ins_data.items():
                        if acc == 0.0:
                            wrong_counts[idx] = wrong_counts.get(idx, 0) + 1
                
                if wrong_counts:
                    # Sort by frequency (most wrong first)
                    sorted_indices = sorted(wrong_counts.keys(), key=lambda x: -wrong_counts[x])
                    indices = sorted_indices[:n]
                else:
                    # Fallback to random if no wrong answers tracked
                    np.random.seed(self.current_step)
                    indices = np.sort(
                        np.random.choice(self.train_index, n, replace=False)
                    ).tolist()
            else:
                # Fallback to random if no tracking data
                np.random.seed(self.current_step)
                indices = np.sort(
                    np.random.choice(self.train_index, n, replace=False)
                ).tolist()
        else:  # "random" (default)
            # Different random sample each step
            np.random.seed(self.current_step)
            indices = np.sort(
                np.random.choice(self.train_index, n, replace=False)
            ).tolist()
        
        return indices
    
    def generate_meta_prompt(self) -> str:
        """Generate the meta-prompt for the optimizer."""
        few_shot_indices = self._select_few_shot_indices()
        
        return opt_utils.gen_meta_prompt(
            old_instructions_and_scores=self.old_instructions_and_scores,
            instruction_pos=self.instruction_pos,
            optimizer_llm_name=self.optimizer_model,
            old_instruction_score_threshold=self.old_instruction_score_threshold,
            max_num_instructions=self.max_num_instructions,
            meta_prompt_type="both_instructions_and_exemplars",
            few_shot_qa_pairs=True,
            include_qa=self.include_qa,
            data=self.raw_data,
            few_shot_index_list=few_shot_indices,
            instructions_before_exemplars=True,
            num_score_buckets=self.num_score_buckets,
            dataset_name=self.dataset_name,
            task_name=self.task_name,
        )
    
    def evaluate_on_test_set_parallel(
        self,
        instructions: list[str],
    ) -> list[tuple[str, float, list[float]]]:
        """Evaluate instructions on the test set in parallel across workers.
        
        Args:
            instructions: List of instructions to evaluate
            
        Returns:
            List of (instruction, average_score, per_question_accuracies) tuples
        """
        if not self.test_questions:
            print("[Controller] Warning: No test questions loaded, skipping test evaluation")
            return []
        
        # Dispatch instructions to workers
        active_requests = {}  # worker_id -> (idx, instruction)
        results_by_idx = {}  # idx -> (instruction, score, accuracies) - maintain order
        instruction_queue = list(enumerate(instructions))
        
        # Initially dispatch to all available workers
        for worker_id in range(min(len(instructions), self.num_workers)):
            if instruction_queue:
                idx, instruction = instruction_queue.pop(0)
                self.input_queues[worker_id].put({
                    "type": "evaluate",
                    "candidate_id": idx,
                    "candidate_prompt": instruction,
                    "questions": self.test_questions,  # Use test questions
                    "instruction_pos": self.instruction_pos,
                    "include_qa": self.include_qa,
                })
                active_requests[worker_id] = (idx, instruction)
        
        # Collect results and dispatch remaining
        while active_requests:
            for worker_id in list(active_requests.keys()):
                try:
                    result = self.output_queues[worker_id].get(timeout=0.1)
                except:
                    continue
                
                if result.get("type") == "error":
                    print(f"[Controller] Worker {worker_id} error: {result.get('error')}")
                    idx, instruction = active_requests.pop(worker_id)
                    results_by_idx[idx] = (instruction, np.nan, [])
                elif result.get("type") == "result":
                    idx, instruction = active_requests.pop(worker_id)
                answers = result["answers"]
                
                # CRITICAL: Validate that the result matches what we sent
                returned_prompt = result.get("candidate_prompt", "")
                if returned_prompt != instruction:
                    print(f"[Controller] FATAL: Test eval result mismatch!")
                    print(f"  Expected: {instruction[:80]}...")
                    print(f"  Got: {returned_prompt[:80]}...")
                    self.stop_workers()
                    raise RuntimeError("Result prompt mismatch - stopping to prevent corrupted results")
                
                # Validate answer count matches question count
                if len(answers) != len(self.test_questions):
                    print(f"[Controller] FATAL: Test eval answer count mismatch!")
                    print(f"  Expected {len(self.test_questions)} answers, got {len(answers)}")
                    self.stop_workers()
                    raise RuntimeError("Answer count mismatch - stopping to prevent corrupted results")
                
                # Compute scores using test answers
                accuracies = []
                detailed_results = []  # For baseline logging
                is_baseline = instruction.strip().lower() == "let's solve the problem."
                
                for i, (pred, true_ans) in enumerate(zip(answers, self.test_answers)):
                    parsed_pred = metrics.get_normalized_prediction(
                        pred,
                        treat_as_number=self.prediction_treat_as_number,
                        num_decimals=self.num_decimals,
                        treat_as_bool=self.prediction_treat_as_bool,
                        treat_as_multiple_choice=self.is_multiple_choice,
                    )
                    
                    if self.is_multiple_choice:
                        input_text = self.test_questions[i]
                    else:
                        input_text = ""
                    
                    # For multiple choice, don't use "include" fallback - single letters 
                    # like 'b' appear in words like "because", causing false positives.
                    treat_include_as_correct = (
                        not self.prediction_treat_as_number and not self.is_multiple_choice
                    )
                    
                    accuracy = eval_utils._get_accuracy(
                        true_answer=true_ans,
                        pred_answer=parsed_pred,
                        input_text=input_text,
                        treat_include_as_correct=treat_include_as_correct,
                    )
                    accuracies.append(accuracy)
                    
                    # Collect detailed results for baseline prompt
                    if is_baseline:
                        # Build the full input prompt as sent to the model
                        # Format must match worker.py's _build_prompt exactly
                        question = self.test_questions[i]
                        if self.include_qa:
                            if self.instruction_pos == "before_Q":
                                full_input_prompt = f"{instruction}\nQ: {question}\n\nA:"
                            elif self.instruction_pos == "Q_begin":
                                full_input_prompt = f"Q: {instruction}\n{question}\n\nA:"
                            elif self.instruction_pos == "Q_end":
                                full_input_prompt = f"Q: {question}\n{instruction}\n\nA:"
                            else:  # A_begin
                                full_input_prompt = f"Q: {question}\n\nA: {instruction}"
                        else:
                            if self.instruction_pos == "Q_begin":
                                full_input_prompt = f"{instruction}\n{question}"
                            else:  # Q_end
                                full_input_prompt = f"{question}\n{instruction}"
                        
                        # Get parsing diagnostic info
                        _, parse_diagnostic = metrics._extract_multiple_choice_answer(
                            pred, return_diagnostic=True
                        ) if self.is_multiple_choice else (None, {"method": "not_mc", "matched_text": None, "position": None})
                        
                        detailed_results.append({
                            "idx": i,
                            "full_input_prompt": full_input_prompt,  # Exact prompt sent to model
                            "question_full": question,  # Full question with options
                            "true_answer": str(true_ans),
                            "raw_output_full": pred,  # Full raw output (not truncated)
                            "raw_output_length": len(pred),
                            "parsed_prediction": parsed_pred,
                            "parse_method": parse_diagnostic.get("method"),
                            "parse_matched_text": parse_diagnostic.get("matched_text"),
                            "parse_position": parse_diagnostic.get("position"),
                            "correct": accuracy == 1,
                        })
                
                average_score = np.mean(accuracies)
                results_by_idx[idx] = (instruction, average_score, accuracies)
                print(f"[Controller] Test eval '{instruction[:50]}...' score: {average_score:.4f}")
                print(f"[Controller]   Worker: {worker_id}, Candidate ID: {idx}")
                
                # Save detailed baseline outputs to file for debugging
                if is_baseline and self.save_folder:
                    baseline_log_path = os.path.join(self.save_folder, "baseline_detailed_outputs.json")
                    baseline_log = {
                        "instruction": instruction,
                        "worker_id": worker_id,
                        "total_questions": len(self.test_questions),
                        "accuracy": average_score,
                        "num_correct": sum(accuracies),
                        "num_wrong": len(accuracies) - sum(accuracies),
                        # Save first 20 wrong answers for analysis
                        "sample_wrong_answers": [r for r in detailed_results if not r["correct"]][:20],
                        # Save first 10 correct answers for comparison
                        "sample_correct_answers": [r for r in detailed_results if r["correct"]][:10],
                        # Summary statistics on raw predictions
                        "prediction_length_stats": {
                            "min": min(len(pred) for pred in answers),
                            "max": max(len(pred) for pred in answers),
                            "mean": sum(len(pred) for pred in answers) / len(answers),
                        },
                    }
                    with open(baseline_log_path, "w") as f:
                        json.dump(baseline_log, f, indent=2)
                    print(f"[Controller] Saved baseline detailed outputs to {baseline_log_path}")
                
                # Dispatch next instruction if available
                if instruction_queue:
                    next_idx, next_instruction = instruction_queue.pop(0)
                    self.input_queues[worker_id].put({
                        "type": "evaluate",
                        "candidate_id": next_idx,
                        "candidate_prompt": next_instruction,
                        "questions": self.test_questions,
                        "instruction_pos": self.instruction_pos,
                        "include_qa": self.include_qa,
                    })
                    active_requests[worker_id] = (next_idx, next_instruction)
        
        # Return results in original order (sorted by idx)
        return [results_by_idx[i] for i in sorted(results_by_idx.keys())]
    
    def _select_prompts_for_final_evaluation(self, final_step: int, max_prompts: int = 21) -> list[tuple[str, float, int]]:
        """Select prompts for final test evaluation.
        
        Selection strategy:
        1. All initial prompts (step -1) - always included for baseline comparison
        2. All prompts from the final step
        3. The overall best prompt (by training score)
        4. Fill remaining slots (up to max_prompts) with highest scoring prompts from other steps
        
        Args:
            final_step: The final step number
            max_prompts: Maximum number of prompts to evaluate on test set (default: 10)
            
        Returns:
            List of (instruction, train_score, step) tuples to evaluate
        """
        selected = []
        selected_instructions = set()
        
        # 1. Always include initial prompts (step -1) for baseline comparison
        initial_prompts = [
            (ins, score, step) for ins, score, step in self.old_instructions_and_scores
            if step == -1 and not np.isnan(score)
        ]
        for prompt in initial_prompts:
            selected.append(prompt)
            selected_instructions.add(prompt[0])
        
        # 2. Add all prompts from final step
        final_step_prompts = [
            (ins, score, step) for ins, score, step in self.old_instructions_and_scores
            if step == final_step and not np.isnan(score) and ins not in selected_instructions
        ]
        for prompt in final_step_prompts:
            selected.append(prompt)
            selected_instructions.add(prompt[0])
        
        # 3. Add best prompt overall (if not already included)
        if self.old_instructions_and_scores:
            valid_prompts = [(ins, score, step) for ins, score, step in self.old_instructions_and_scores if not np.isnan(score)]
            if valid_prompts:
                best_prompt = max(valid_prompts, key=lambda x: x[1])
                if best_prompt[0] not in selected_instructions:
                    selected.append(best_prompt)
                    selected_instructions.add(best_prompt[0])
        
        # 4. Fill remaining slots with highest scoring prompts from other steps (up to max_prompts total)
        if len(selected) < max_prompts:
            other_prompts = [
                (ins, score, step) for ins, score, step in self.old_instructions_and_scores
                if ins not in selected_instructions and not np.isnan(score)
            ]
            other_prompts_sorted = sorted(other_prompts, key=lambda x: x[1], reverse=True)
            
            for prompt in other_prompts_sorted:
                if len(selected) >= max_prompts:
                    break
                selected.append(prompt)
                selected_instructions.add(prompt[0])
        
        print(f"[Controller] Selected {len(selected)} prompts for final test evaluation:")
        for i, (ins, score, step) in enumerate(selected):
            step_label = "INITIAL" if step == -1 else f"Step {step}"
            print(f"  {i+1}. {step_label}, train_score={score:.4f}: {ins[:60]}...")
        
        return selected
    
    def _run_final_test_evaluation(self, final_step: int):
        """Run final evaluation on test set and save results.
        
        Args:
            final_step: The final optimization step number
        """
        if not hasattr(self, 'test_questions') or not self.test_questions:
            print("[Controller] No test set available, skipping final evaluation")
            return
        
        # Select prompts for evaluation
        prompts_to_evaluate = self._select_prompts_for_final_evaluation(final_step)
        
        if not prompts_to_evaluate:
            print("[Controller] No prompts to evaluate on test set")
            return
        
        # Extract just the instructions
        instructions = [ins for ins, _, _ in prompts_to_evaluate]
        
        # Evaluate on test set
        print(f"\n[Controller] Evaluating {len(instructions)} prompts on {len(self.test_questions)} test questions...")
        test_results = self.evaluate_on_test_set_parallel(instructions)
        
        # Compile final results
        final_results = {
            "dataset": self.dataset_name,
            "task": self.task_name,
            "num_test_questions": len(self.test_questions),
            "num_train_questions": len(self.train_questions),
            "final_step": final_step,
            "evaluated_prompts": []
        }
        
        for (instruction, train_score, train_step), (_, test_score, test_accuracies) in zip(
            prompts_to_evaluate, test_results
        ):
            prompt_result = {
                "instruction": instruction,
                "train_score": float(train_score) if not np.isnan(train_score) else None,
                "train_step": int(train_step),
                "test_score": float(test_score) if not np.isnan(test_score) else None,
                "is_initial": train_step == -1,  # Baseline/initial prompt
                "is_best_train": (instruction == max(
                    self.old_instructions_and_scores, key=lambda x: x[1]
                )[0]) if self.old_instructions_and_scores else False,
                "is_final_step": train_step == final_step,
            }
            final_results["evaluated_prompts"].append(prompt_result)
        
        # Sort by test score
        final_results["evaluated_prompts"].sort(
            key=lambda x: x["test_score"] if x["test_score"] is not None else -1,
            reverse=True
        )
        
        # Add summary statistics
        test_scores = [p["test_score"] for p in final_results["evaluated_prompts"] if p["test_score"] is not None]
        if test_scores:
            final_results["summary"] = {
                "best_test_score": max(test_scores),
                "avg_test_score": float(np.mean(test_scores)),
                "best_test_instruction": final_results["evaluated_prompts"][0]["instruction"],
            }
        
        # Save to JSON
        if self.save_folder:
            json_path = os.path.join(self.save_folder, "test_evaluation_results.json")
            with open(json_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"[Controller] Test evaluation results saved to: {json_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Evaluation Summary")
        print("=" * 60)
        for p in final_results["evaluated_prompts"]:
            marker = ""
            if p.get("is_initial"):
                marker += " [INITIAL/BASELINE]"
            if p["is_best_train"]:
                marker += " [BEST_TRAIN]"
            if p["is_final_step"]:
                marker += " [FINAL_STEP]"
            step_label = "INITIAL" if p["train_step"] == -1 else f"Step {p['train_step']}"
            print(f"  Test: {p['test_score']:.4f} | Train: {p['train_score']:.4f} | {step_label}{marker}")
            print(f"    {p['instruction'][:80]}...")
        
        if "summary" in final_results:
            print(f"\nBest test score: {final_results['summary']['best_test_score']:.4f}")
            print(f"Average test score: {final_results['summary']['avg_test_score']:.4f}")
    
    def run_optimization(
        self,
        initial_instructions: list[str],
        resume_from_checkpoint: Optional[str] = None,
    ):
        """Run the full OPRO optimization loop.
        
        Args:
            initial_instructions: Starting instructions to evaluate
            resume_from_checkpoint: Path to checkpoint file to resume from
        """
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n[Controller] Received interrupt signal, saving checkpoint...")
            self._save_checkpoint(step=-1, final=True)
            self.stop_workers()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Resume from checkpoint if specified
        start_step = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            start_step = self._load_checkpoint(resume_from_checkpoint)
            print(f"[Controller] Resumed from step {start_step}")
        
        # Start workers
        self.start_workers()
        
        try:
            # Evaluate initial instructions if not resuming
            if start_step == 0:
                print("\n[Controller] Evaluating initial instructions...")
                for instruction in initial_instructions:
                    print(f"[Controller] Evaluating: {instruction}")
                    score, detailed_df = self.evaluate_instruction(instruction)
                    print(f"[Controller] Score: {score:.4f}")
                    
                    self.old_instructions_and_scores.append((instruction, score, -1))
                    self.old_instructions_and_scores_raw.append((instruction, score, -1))
                    self.instruction_score_dict[instruction] = score
                    self.old_instruction_md5_hashstrings_set.add(
                        eval_utils.instruction_to_filename(instruction, md5_hashing=True)
                    )
                    
                    # Save results
                    if self.save_folder:
                        filename = eval_utils.instruction_to_filename(instruction)
                        filepath = os.path.join(
                            self.save_folder, "result_by_instruction", f"{filename}.csv"
                        )
                        detailed_df.to_csv(filepath)
            
            # Main optimization loop
            for step in tqdm(range(start_step, self.num_search_steps), desc="OPRO Steps"):
                self.current_step = step  # Update for few-shot selection
                print(f"\n[Controller] ===== Step {step} =====")
                
                # Generate meta-prompt
                meta_prompt = self.generate_meta_prompt()
                self.meta_prompts.append((meta_prompt, step))
                
                # Generate new candidates using optimizer (local vLLM or OpenAI API)
                optimizer_type = "local vLLM" if self.use_local_optimizer else "OpenAI API"
                print(f"[Controller] Generating candidates via {optimizer_type}...")
                raw_candidates = self.optimizer.generate_and_parse_candidates(
                    meta_prompt=meta_prompt,
                    instruction_pos=self.instruction_pos,
                    num_candidates=self.num_candidates_per_step,
                    temperature=self.optimizer_temperature,
                )
                
                # Save optimizer generation log for debugging
                if self.save_folder and hasattr(self.optimizer, 'last_generation_log'):
                    self._save_optimizer_log(step, meta_prompt, self.optimizer.last_generation_log)
                
                # Filter duplicates and invalid candidates
                candidates_to_evaluate = []
                for candidate in raw_candidates:
                    candidate = eval_utils.polish_sentence(candidate)
                    md5_hash = eval_utils.instruction_to_filename(candidate, md5_hashing=True)
                    
                    if md5_hash in self.old_instruction_md5_hashstrings_set:
                        print(f"[Controller] Skipping duplicate: {candidate[:50]}...")
                        continue
                    
                    if len(candidate) > 500:
                        print(f"[Controller] Skipping too long: {candidate[:50]}...")
                        continue
                    
                    if "INS" in candidate:
                        print(f"[Controller] Skipping contains INS: {candidate[:50]}...")
                        continue
                    
                    # Skip instructions containing numbers for GSM8K/MATH (matches original)
                    if self.dataset_name in {"gsm8k", "math"} and any(char.isdigit() for char in candidate):
                        print(f"[Controller] Skipping contains numbers: {candidate[:50]}...")
                        continue
                    
                    candidates_to_evaluate.append(candidate)
                    self.old_instruction_md5_hashstrings_set.add(md5_hash)
                
                print(f"[Controller] Evaluating {len(candidates_to_evaluate)} candidates...")
                
                # Evaluate candidates in parallel
                if candidates_to_evaluate:
                    results = self.evaluate_candidates_parallel(candidates_to_evaluate)
                    
                    # Collect per-sample accuracies for IPOMP
                    ipomp_per_sample_accuracies = []
                    evaluated_candidates = []
                    
                    for instruction, score, detailed_df, per_sample_acc in results:
                        if not np.isnan(score):
                            self.old_instructions_and_scores.append((instruction, score, step))
                            self.instruction_score_dict[instruction] = score
                            
                            # Track per-question accuracies for few-shot selection
                            if detailed_df is not None and 'accuracy' in detailed_df.columns:
                                self.per_question_accuracies[instruction] = dict(
                                    zip(detailed_df.index, detailed_df['accuracy'])
                                )
                            
                            # Save detailed results
                            if self.save_folder and detailed_df is not None:
                                filename = eval_utils.instruction_to_filename(instruction)
                                filepath = os.path.join(
                                    self.save_folder, "result_by_instruction", f"{filename}.csv"
                                )
                                detailed_df.to_csv(filepath)
                            
                            # Collect for IPOMP
                            if per_sample_acc:
                                ipomp_per_sample_accuracies.append(per_sample_acc)
                                evaluated_candidates.append(instruction)
                        
                        self.old_instructions_and_scores_raw.append((instruction, score, step))
                    
                    # IPOMP: Record performance and potentially update subset
                    if self.use_ipomp and self.ipomp_manager is not None and ipomp_per_sample_accuracies:
                        # Record this iteration's performance
                        self.ipomp_manager.record_iteration_performance(
                            step=step,
                            candidate_prompts=evaluated_candidates,
                            per_sample_accuracies=ipomp_per_sample_accuracies,
                        )
                        
                        # Update subset every step (can be changed to less frequent)
                        # Paper suggests updating after every iteration
                        new_train_index, num_replaced = self.ipomp_manager.update_subset(step)
                        
                        if num_replaced > 0:
                            print(f"[Controller] IPOMP: Replaced {num_replaced} samples at step {step}")
                            # Update training data with new subset
                            self._update_training_data_from_ipomp()
                
                # Checkpoint
                if step % self.checkpoint_interval == 0:
                    self._save_checkpoint(step)
                
                # Report best so far
                if self.old_instructions_and_scores:
                    best_instruction, best_score, best_step = max(
                        self.old_instructions_and_scores, key=lambda x: x[1]
                    )
                    print(f"[Controller] Best so far (step {best_step}): {best_score:.4f}")
                    print(f"[Controller] Best instruction: {best_instruction[:100]}...")
            
            # Final save
            self._save_checkpoint(self.num_search_steps - 1, final=True)
            
            # Run final evaluation on test set
            print("\n[Controller] ===== Final Test Evaluation =====")
            self._run_final_test_evaluation(final_step=self.num_search_steps - 1)
            
        finally:
            self.stop_workers()
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """Save checkpoint to disk."""
        if not self.save_folder:
            return
        
        checkpoint = {
            "step": step,
            "old_instructions_and_scores": self.old_instructions_and_scores,
            "old_instructions_and_scores_raw": self.old_instructions_and_scores_raw,
            "instruction_score_dict": self.instruction_score_dict,
            "meta_prompts": self.meta_prompts,
            "eval_results": self.eval_results,
        }
        
        # Save pickle
        checkpoint_path = os.path.join(self.save_folder, "results_dict.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Save JSON for readability
        json_checkpoint = {
            "step": step,
            "best_instructions": sorted(
                self.old_instructions_and_scores, key=lambda x: x[1], reverse=True
            )[:10],
        }
        json_path = os.path.join(self.save_folder, "checkpoint.json")
        with open(json_path, "w") as f:
            json.dump(json_checkpoint, f, indent=2)
        
        # Generate training plot in background thread (non-blocking)
        plot_thread = threading.Thread(target=self._save_training_plot, daemon=True)
        plot_thread.start()
        
        # Save training samples if any
        self._save_training_samples()
        
        print(f"[Controller] Checkpoint saved at step {step}")
    
    def _save_optimizer_log(self, step: int, meta_prompt: str, generation_log: list):
        """Save optimizer generation log for debugging.
        
        Saves the meta-prompt, raw optimizer outputs, and parsing results.
        This helps diagnose issues where garbage instructions get through.
        """
        if not self.save_folder:
            return
        
        log_dir = os.path.join(self.save_folder, "optimizer_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, f"step_{step:03d}.json")
        with open(log_path, "w") as f:
            json.dump({
                "step": step,
                "meta_prompt_length": len(meta_prompt),
                "meta_prompt_preview": meta_prompt[:2000] + "..." if len(meta_prompt) > 2000 else meta_prompt,
                "num_generations": len(generation_log),
                "num_accepted": sum(1 for g in generation_log if g.get("status") == "accepted"),
                "num_filtered": sum(1 for g in generation_log if g.get("status") == "filtered"),
                "num_errors": sum(1 for g in generation_log if g.get("status") == "error"),
                "generations": generation_log,
            }, f, indent=2)
    
    def _save_training_samples(self):
        """Save sampled training evaluations to file for debugging."""
        if not self.save_folder or not self.training_eval_samples:
            return
        
        samples_path = os.path.join(self.save_folder, "training_eval_samples.json")
        with open(samples_path, "w") as f:
            json.dump({
                "total_prompts_evaluated": self.total_prompts_evaluated,
                "sample_interval": self.training_sample_interval,
                "num_samples": len(self.training_eval_samples),
                "samples": self.training_eval_samples,
            }, f, indent=2)
        print(f"[Controller] Saved {len(self.training_eval_samples)} training eval samples")
    
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint from disk."""
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        self.old_instructions_and_scores = checkpoint["old_instructions_and_scores"]
        self.old_instructions_and_scores_raw = checkpoint["old_instructions_and_scores_raw"]
        self.instruction_score_dict = checkpoint["instruction_score_dict"]
        self.meta_prompts = checkpoint.get("meta_prompts", [])
        self.eval_results = checkpoint.get("eval_results", [])
        
        # Rebuild MD5 set
        for instruction, _, _ in self.old_instructions_and_scores:
            md5_hash = eval_utils.instruction_to_filename(instruction, md5_hashing=True)
            self.old_instruction_md5_hashstrings_set.add(md5_hash)
        
        return checkpoint["step"] + 1
    
    def _save_training_plot(self):
        """Save training progress plot."""
        if not self.save_folder:
            return
        
        try:
            steps = []
            scores = []
            for instruction, score, step in self.old_instructions_and_scores:
                if not np.isnan(score):
                    steps.append(step)
                    scores.append(score)
            
            if not steps:
                return
            
            plt.figure(figsize=(10, 6))
            plt.scatter(steps, scores, alpha=0.3, s=50, label='Individual scores')
            
            # Average line
            unique_steps = sorted(set(steps))
            avg_scores = []
            best_scores = []
            current_best = 0
            for step in unique_steps:
                step_scores = [s for st, s in zip(steps, scores) if st == step]
                avg_scores.append(np.mean(step_scores))
                current_best = max(current_best, max(step_scores))
                best_scores.append(current_best)
            
            plt.plot(unique_steps, avg_scores, 'b-', linewidth=2, label='Average', marker='o', markersize=6)
            plt.plot(unique_steps, best_scores, 'r-', linewidth=2, label='Best so far', marker='s', markersize=5)
            
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Training Accuracy', fontsize=12)
            plt.title('OPRO Optimization Progress', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(self.save_folder, 'training_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"[Controller] Warning: Could not save training plot: {e}")

