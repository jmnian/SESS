"""Utilities for selecting train/eval/test subsets from datasets."""

import gc
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Default number of GPUs for tensor parallelism
DEFAULT_TENSOR_PARALLEL_SIZE = 1

# Number of attention heads for common embedding models (for TP validation)
# If not listed, defaults to 32 (common for 7B/8B models)
EMBEDDING_MODEL_NUM_HEADS = {
    "Qwen/Qwen3-Embedding-8B": 32,
    "intfloat/e5-mistral-7b-instruct": 32,
    "BAAI/bge-large-en-v1.5": 16,
    "sentence-transformers/all-MiniLM-L6-v2": 12,
}


def _get_valid_tp_size(requested_tp: int, num_heads: int = 32) -> int:
    """Get a valid tensor parallel size that divides num_heads evenly.
    
    Args:
        requested_tp: Requested tensor parallel size
        num_heads: Number of attention heads in the model (default: 32)
        
    Returns:
        Largest valid TP size that is <= requested_tp and divides num_heads
    """
    # Find the largest divisor of num_heads that is <= requested_tp
    for tp in range(requested_tp, 0, -1):
        if num_heads % tp == 0:
            return tp
    return 1  # Fallback to 1

def random_subset(num_examples, train_ratio, eval_ratio, seed=42):
  """Randomly select train and eval indices from a dataset.
  
  Args:
    num_examples: Total number of examples in the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    seed: Random seed for reproducibility (default: 42)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  # Random selection with fixed seed for reproducibility
  np.random.seed(seed)
  
  train_index = np.sort(
      np.array(
          np.random.choice(
              num_examples, size=int(train_ratio * num_examples), replace=False
          )
      )
  )
  
  eval_and_test_index = np.sort(
      np.array(list(set(np.arange(num_examples)) - set(train_index)))
  )
  
  eval_index = np.sort(
      np.array(
          np.random.choice(
              eval_and_test_index,
              size=int(eval_ratio * num_examples),
              replace=False,
          )
      )
  )
  
  return train_index, eval_index


def _generate_embeddings_vllm(
    questions, 
    dataset_name, 
    embeddings_dir, 
    embedding_model="Qwen/Qwen3-Embedding-8B",
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Generate or load embeddings for questions using local vLLM.
  
  Args:
    questions: List of question strings
    dataset_name: Name of the dataset
    embeddings_dir: Directory to save/load embeddings
    embedding_model: Model name for embeddings (default: Qwen3-Embedding-8B)
    tensor_parallel_size: Number of GPUs for tensor parallelism
    gpu_memory_utilization: Fraction of GPU memory to use
    
  Returns:
    numpy array of embeddings, shape (num_questions, embedding_dim)
  """
  from vllm import LLM
  
  # Create safe model name for cache file
  safe_model_name = embedding_model.replace("/", "_").replace(":", "_")
  embeddings_file = os.path.join(
      embeddings_dir, 
      f"{dataset_name}_{safe_model_name}_embeddings.npy"
  )
  
  # Check if embeddings already exist
  if os.path.exists(embeddings_file):
    cached_embeddings = np.load(embeddings_file)
    # Validate cache size matches current data
    if cached_embeddings.shape[0] == len(questions):
      print(f"Loading existing embeddings from {embeddings_file}")
      return cached_embeddings
    else:
      print(f"Cache size mismatch: cached {cached_embeddings.shape[0]} vs current {len(questions)}. Regenerating...")
      del cached_embeddings
  
  # Force TP=1 for embedding generation to avoid NCCL issues
  # Embedding models are small and TP=1 is sufficient for one-time preprocessing
  # Using TP>1 can cause NCCL conflicts when running before/after other multi-GPU ops
  embedding_tp = 1
  if tensor_parallel_size > 1:
    print(f"Using tensor_parallel_size=1 for embeddings (avoids NCCL conflicts, embedding is one-time)")
  
  print(f"Generating embeddings for {len(questions)} questions using vLLM...")
  print(f"Embedding model: {embedding_model}")
  print(f"Tensor parallel size: {embedding_tp}")
  
  # Initialize vLLM for embeddings (no task parameter in vLLM 0.13+)
  llm = LLM(
      model=embedding_model,
      tensor_parallel_size=embedding_tp,
      gpu_memory_utilization=gpu_memory_utilization,
      trust_remote_code=True,
  )
  
  # Model-specific preprocessing
  # For E5 models: add instruction prefix
  # For Qwen3-Embedding: no prefix needed (it handles instructions internally)
  if "e5" in embedding_model.lower():
    processed_questions = [
        f"Instruct: Retrieve semantically similar questions\nQuery: {q.strip()}"
        for q in questions
    ]
  else:
    processed_questions = [q.strip() for q in questions]
  
  # Generate embeddings
  print("Generating embeddings...")
  outputs = llm.embed(processed_questions)
  
  # Extract embedding vectors
  # vLLM embed() returns EmbeddingRequestOutput objects
  # The structure varies by vLLM version, so handle multiple formats
  all_embeddings = []
  for output in outputs:
    # Try different attribute paths for compatibility
    if hasattr(output, 'outputs') and hasattr(output.outputs, 'embedding'):
      # Newer vLLM: output.outputs.embedding
      emb = output.outputs.embedding
    elif hasattr(output, 'outputs') and hasattr(output.outputs, 'data'):
      # Older vLLM: output.outputs.data
      emb = output.outputs.data
    elif hasattr(output, 'embedding'):
      # Direct: output.embedding
      emb = output.embedding
    else:
      raise AttributeError(f"Unknown embedding output format: {type(output)}, attrs: {dir(output)}")
    
    # Convert to numpy if needed
    if hasattr(emb, 'cpu'):
      emb = emb.cpu().numpy()
    elif hasattr(emb, 'numpy'):
      emb = emb.numpy()
    all_embeddings.append(emb)
  
  embeddings_array = np.array(all_embeddings, dtype=np.float32)
  
  # Save embeddings
  os.makedirs(embeddings_dir, exist_ok=True)
  np.save(embeddings_file, embeddings_array)
  print(f"Saved embeddings to {embeddings_file}")
  print(f"Embedding shape: {embeddings_array.shape}")
  
  # Clean up GPU memory thoroughly to avoid interference with subsequent vLLM instances
  del llm
  gc.collect()  # Force garbage collection
  import torch
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
    torch.cuda.empty_cache()
  
  return embeddings_array


def _compute_dense_similarity(embeddings, dataset_name, sim_matrices_dir, embedding_model=""):
  """Compute dense similarity matrix from embeddings.
  
  Args:
    embeddings: numpy array of embeddings, shape (N, dim)
    dataset_name: Name of the dataset
    sim_matrices_dir: Directory to save/load similarity matrices
    embedding_model: Name of embedding model used (for cache file naming)
    
  Returns:
    Dense similarity matrix, shape (N, N), values in [0, 1]
  """
  # Include embedding model in filename to avoid cache collisions
  safe_model_name = embedding_model.replace("/", "_").replace(":", "_") if embedding_model else "default"
  sim_file = os.path.join(sim_matrices_dir, f"{dataset_name}_{safe_model_name}_dense_similarity.npy")
  
  # Check if similarity matrix already exists
  if os.path.exists(sim_file):
    cached_sim = np.load(sim_file)
    # Validate cache size matches current embeddings
    if cached_sim.shape[0] == embeddings.shape[0]:
      print(f"Loading existing dense similarity matrix from {sim_file}")
      return cached_sim
    else:
      print(f"Cache size mismatch: cached {cached_sim.shape[0]} vs current {embeddings.shape[0]}. Regenerating...")
      del cached_sim
  
  print("Computing dense similarity matrix...")
  
  # Normalize embeddings
  norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
  norm = np.maximum(norm, 1e-8)  # Avoid divide by zero
  E_norm = embeddings / norm
  
  # Compute cosine similarity
  cos_matrix = E_norm @ E_norm.T
  
  # Normalize to [0, 1] range
  sim_dense = (cos_matrix + 1.0) / 2.0
  
  # Save similarity matrix
  os.makedirs(sim_matrices_dir, exist_ok=True)
  np.save(sim_file, sim_dense)
  print(f"Saved dense similarity matrix to {sim_file}")
  
  return sim_dense


def _compute_tfidf_similarity(questions, dataset_name, sim_matrices_dir):
  """Compute TF-IDF similarity matrix from questions.
  
  Args:
    questions: List of question strings
    dataset_name: Name of the dataset
    sim_matrices_dir: Directory to save/load similarity matrices
    
  Returns:
    TF-IDF similarity matrix, shape (N, N), values in [0, 1]
  """
  sim_file = os.path.join(sim_matrices_dir, f"{dataset_name}_tfidf_similarity.npy")
  
  # Check if similarity matrix already exists
  if os.path.exists(sim_file):
    cached_sim = np.load(sim_file)
    # Validate cache size matches current questions
    if cached_sim.shape[0] == len(questions):
      print(f"Loading existing TF-IDF similarity matrix from {sim_file}")
      return cached_sim
    else:
      print(f"Cache size mismatch: cached {cached_sim.shape[0]} vs current {len(questions)}. Regenerating...")
      del cached_sim
  
  print("Computing TF-IDF similarity matrix...")
  
  # Compute TF-IDF vectors
  tfidf_vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
  tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
  
  # Compute cosine similarity
  tfidf_sim = cosine_similarity(tfidf_matrix).astype(np.float32)
  
  # Save similarity matrix
  os.makedirs(sim_matrices_dir, exist_ok=True)
  np.save(sim_file, tfidf_sim)
  print(f"Saved TF-IDF similarity matrix to {sim_file}")
  
  return tfidf_sim


def _row_minmax_normalize(M):
  """Apply row-wise min-max normalization.
  
  Args:
    M: Input matrix
    
  Returns:
    Row-wise normalized matrix
  """
  mn = M.min(axis=1, keepdims=True)
  mx = M.max(axis=1, keepdims=True)
  return (M - mn) / (mx - mn + 1e-8)


def _greedy_subset_selection(M, k, verbose=True):
  """Greedy subset selection using facility location objective.
  
  Args:
    M: Similarity matrix of shape (N, N), values in [0,1]
    k: Subset size
    verbose: Print progress if True
    
  Returns:
    List of selected indices
  """
  N = M.shape[0]
  S = []  # Selected subset
  remaining = set(range(N))  # Candidates not yet selected
  current_max = np.zeros(N)  # Current coverage for each point
  
  if verbose:
    print(f"Running greedy selection to select {k} samples...")
  
  for iteration in tqdm(range(k), desc="Greedy selection", disable=not verbose):
    best_gain = -1.0
    best_i = None
    
    # Find the point that maximizes marginal gain
    for i in remaining:
      row_i = M[i]  # Similarity from i to all points
      gain_vector = np.maximum(current_max, row_i) - current_max
      delta = gain_vector.sum()  # Marginal gain
      
      if delta > best_gain:
        best_gain = delta
        best_i = i
    
    # Add best point to subset
    i_star = best_i
    S.append(i_star)
    remaining.remove(i_star)
    
    # Update coverage
    current_max = np.maximum(current_max, M[i_star])
  
  if verbose:
    print("Greedy selection complete.")
  
  return S


def _extract_questions_from_raw_data(dataset_name, raw_data):
  """Extract questions from raw_data based on dataset type.
  
  Args:
    dataset_name: Name of the dataset
    raw_data: Raw data (format depends on dataset)
    
  Returns:
    List of question strings
  """
  if dataset_name == "gsm8k":
    # For gsm8k, raw_data is a DataFrame with questions in column 0
    if isinstance(raw_data, pd.DataFrame):
      questions = raw_data[0].tolist()
    else:
      # If it's a list, questions are in index 0
      questions = raw_data[0] if isinstance(raw_data, list) else raw_data
    return [str(q).strip() for q in questions]
  elif dataset_name == "bbh":
    # For BBH, raw_data is a list of dictionaries
    return [str(item.get("input", "")).strip() for item in raw_data]
  elif dataset_name == "mmlu":
    # For MMLU, raw_data is a DataFrame
    if isinstance(raw_data, pd.DataFrame):
      questions = raw_data[0].tolist()
    else:
      questions = raw_data
    return [str(q).strip() for q in questions]
  elif dataset_name == "math":
    # For MATH, raw_data is a list of dictionaries with "problem" key
    return [str(item.get("input", "")).strip() for item in raw_data]
  elif dataset_name == "gpqa":
    # For GPQA, raw_data is a list of dictionaries
    # Use question_only for embedding similarity (without choices)
    return [str(item.get("question_only", item.get("input", ""))).strip() for item in raw_data]
  else:
    raise ValueError(f"Unknown dataset: {dataset_name}")


def representative_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio, 
    alpha=0.9, 
    seed=42,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select train and eval indices using representative subset selection.
  
  This method uses a combination of dense (embedding-based) and lexical (TF-IDF)
  similarity to select a diverse and representative subset of examples.
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    alpha: Weight for dense similarity (1-alpha for lexical). Default 0.9.
    seed: Random seed for reproducibility (default: 42)
    embedding_model: Model for generating embeddings (default: Qwen3-Embedding-8B)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for saving embeddings and similarity matrices
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  cache_dir = os.path.join(opro_root, "embeddings_and_sim_matrices", dataset_name)
  embeddings_dir = cache_dir
  sim_matrices_dir = cache_dir
  
  # Extract questions from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  print(f"\n{'='*60}")
  print(f"Representative Subset Selection for {dataset_name}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Alpha (dense weight): {alpha}")
  print(f"Embedding model: {embedding_model}")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"Cache directory: {cache_dir}")
  print(f"{'='*60}\n")
  
  # Step 1: Generate or load embeddings using vLLM
  embeddings = _generate_embeddings_vllm(
      questions, 
      dataset_name, 
      embeddings_dir,
      embedding_model=embedding_model,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Step 2: Compute or load dense similarity matrix
  sim_dense = _compute_dense_similarity(embeddings, dataset_name, sim_matrices_dir, embedding_model)
  
  # Step 3: Compute or load TF-IDF similarity matrix
  sim_tfidf = _compute_tfidf_similarity(questions, dataset_name, sim_matrices_dir)
  
  # Step 4: Apply normalizations as per notebook
  print("Applying row-wise min-max normalization...")
  sim_dense_norm = _row_minmax_normalize(sim_dense)
  sim_tfidf_norm = _row_minmax_normalize(sim_tfidf)
  
  # Apply square root to TF-IDF to smooth out the distribution
  sim_tfidf_norm_sqrt = np.sqrt(sim_tfidf_norm)
  
  # Step 5: Mix the similarity matrices
  print(f"Mixing similarity matrices with alpha={alpha}...")
  M_mixed = alpha * sim_dense_norm + (1 - alpha) * sim_tfidf_norm_sqrt
  
  # Step 6: Run greedy subset selection for training set
  k_train = int(train_ratio * num_examples)
  train_index = _greedy_subset_selection(M_mixed, k_train, verbose=True)
  train_index = np.sort(np.array(train_index))
  
  # Step 7: Select eval indices from remaining examples
  remaining_indices = list(set(range(num_examples)) - set(train_index))
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  print(f"\n{'='*60}")
  print(f"Selection complete!")
  print(f"Training set size: {len(train_index)}")
  print(f"Eval set size: {len(eval_index)}")
  print(f"{'='*60}\n")
  
  return train_index, eval_index


def _extract_full_answers_from_raw_data(dataset_name, raw_data):
  """Extract full answers (with reasoning) from raw_data based on dataset type.
  
  Args:
    dataset_name: Name of the dataset
    raw_data: Raw data (format depends on dataset)
    
  Returns:
    List of answer strings (may include reasoning/solution steps)
  """
  if dataset_name == "gsm8k":
    # For gsm8k, raw_data is a DataFrame or list with full answers in column/index 2
    if isinstance(raw_data, pd.DataFrame):
      answers = raw_data[2].tolist()
    else:
      answers = raw_data[2] if isinstance(raw_data, list) else raw_data
    return [str(a).strip() for a in answers]
  elif dataset_name == "bbh":
    # For BBH, raw_data is a list of dictionaries with "target" key
    return [str(item.get("target", "")).strip() for item in raw_data]
  elif dataset_name == "mmlu":
    # For MMLU, raw_data is a DataFrame with answer in column 5
    if isinstance(raw_data, pd.DataFrame):
      answers = raw_data[5].tolist()
    else:
      answers = raw_data
    return [str(a).strip() for a in answers]
  elif dataset_name == "math":
    # For MATH, raw_data is a list of dictionaries with "solution" key for full answer
    return [str(item.get("solution", "")).strip() for item in raw_data]
  else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

def _extract_short_answers_from_raw_data(dataset_name, raw_data):
  """Extract answers from raw_data based on dataset type.
  
  Args:
    dataset_name: Name of the dataset
    raw_data: Raw data (format depends on dataset)
    
  Returns:
    List of answer strings
  """
  if dataset_name == "gsm8k":
    # For gsm8k, raw_data is a DataFrame or list with answers in column/index 1
    if isinstance(raw_data, pd.DataFrame):
      answers = raw_data[1].tolist()
    else:
      answers = raw_data[1] if isinstance(raw_data, list) else raw_data
    return [str(a).strip() for a in answers]
  elif dataset_name == "bbh":
    # For BBH, raw_data is a list of dictionaries with "target" key
    return [str(item.get("target", "")).strip() for item in raw_data]
  elif dataset_name == "mmlu":
    # For MMLU, raw_data is a DataFrame with answer in column 5 (the correct answer letter)
    if isinstance(raw_data, pd.DataFrame):
      answers = raw_data[5].tolist()
    else:
      answers = raw_data
    return [str(a).strip() for a in answers]
  elif dataset_name == "math":
    # For MATH, raw_data is a list of dictionaries with "answer" key
    return [str(item.get("target", "")).strip() for item in raw_data]
  elif dataset_name == "gpqa":
    # For GPQA, raw_data is a list of dictionaries with "target" key (letter A/B/C/D)
    return [str(item.get("target", "")).strip() for item in raw_data]
  else:
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _is_multiple_choice_dataset(dataset_name):
  """Check if a dataset uses multiple choice format (A/B/C/D answers).
  
  Args:
    dataset_name: Name of the dataset
    
  Returns:
    True if the dataset uses multiple choice format, False otherwise
  """
  # Datasets that use A/B/C/D multiple choice format
  multiple_choice_datasets = {
      "gpqa",
      "mmlu",
      # BBH has mixed formats, but many subtasks are multiple choice
      # Add other multiple choice datasets here as needed
  }
  
  # Check if dataset_name starts with or matches any multiple choice dataset
  dataset_lower = dataset_name.lower()
  for mc_dataset in multiple_choice_datasets:
    if dataset_lower.startswith(mc_dataset) or mc_dataset in dataset_lower:
      return True
  
  return False


def _compute_confidence_scores_vllm(
    questions, 
    answers, 
    model_name, 
    dataset_name, 
    cache_dir,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Compute or load confidence scores using vLLM with tensor parallelism.
  
  This computes the log probability of the answer tokens given the question
  as context using vLLM's prompt_logprobs feature.
  
  For multiple choice datasets (GPQA, MMLU), uses a prompt appropriate for
  selecting A/B/C/D. For numeric datasets (GSM8K, MATH), uses a numeric prompt.
  
  Args:
    questions: List of question strings
    answers: List of answer strings
    model_name: Model name for scoring (via vLLM)
    dataset_name: Name of dataset (for caching)
    cache_dir: Directory to save/load confidence scores
    tensor_parallel_size: Number of GPUs for tensor parallelism
    gpu_memory_utilization: Fraction of GPU memory to use
    
  Returns:
    numpy array of log likelihood scores, shape (num_examples,)
  """
  from vllm import LLM, SamplingParams
  import torch
  
  # Create a safe filename from the model name
  safe_model_name = model_name.replace("/", "_").replace(":", "_")
  scores_file = os.path.join(
      cache_dir, 
      f"{dataset_name}_{safe_model_name}_confidence_scores_vllm.npy"
  )
  
  # Check if scores already exist
  if os.path.exists(scores_file):
    cached_scores = np.load(scores_file)
    # Validate cache size matches current data
    if cached_scores.shape[0] == len(questions):
      print(f"Loading existing confidence scores from {scores_file}")
      return cached_scores
    else:
      print(f"Cache size mismatch: cached {cached_scores.shape[0]} vs current {len(questions)}. Regenerating...")
      del cached_scores
  
  # Determine if this is a multiple choice dataset
  is_multiple_choice = _is_multiple_choice_dataset(dataset_name)
  
  print(f"Computing confidence scores for {len(questions)} examples using vLLM...")
  print(f"Model: {model_name}")
  print(f"Dataset type: {'multiple choice (A/B/C/D)' if is_multiple_choice else 'numeric/free-form'}")
  
  # Force tensor_parallel_size=1 to avoid NCCL conflicts with other multi-GPU operations
  # (e.g., optimizer on GPU 0, workers on GPUs 1-7). The 7B model fits on a single GPU
  # and this is a one-time operation, so single-GPU is acceptable.
  actual_tp_size = 1
  print(f"Tensor parallel size: {actual_tp_size} (forced single-GPU to avoid NCCL conflicts)")
  
  # Initialize vLLM
  llm = LLM(
      model=model_name,
      tensor_parallel_size=actual_tp_size,
      gpu_memory_utilization=gpu_memory_utilization,
      trust_remote_code=True,
  )
  
  # Get tokenizer for computing answer token lengths
  tokenizer = llm.get_tokenizer()
  
  # Build prompts with answers appended (we'll compute logprobs of the full sequence)
  # Use different prompt templates based on dataset type
  prompts = []
  answer_token_counts = []
  
  for q, a in zip(questions, answers):
    if is_multiple_choice:
      # Multiple choice prompt for GPQA, MMLU, etc.
      prompt = f"Directly give the choice A or B or C or D: {q}\nAnswer: "
    else:
      # Numeric/free-form prompt for GSM8K, MATH, etc.
      prompt = f"Directly give the numeric answer to the following question: {q}\n\nAnswer: "
    
    full_text = prompt + a
    prompts.append(full_text)
    
    # Count answer tokens
    answer_tokens = tokenizer.encode(a, add_special_tokens=False)
    answer_token_counts.append(len(answer_tokens))
  
  # Use prompt_logprobs to get logprobs for all tokens including the prompt
  # We set max_tokens=1 since we just want to compute logprobs, not generate
  sampling_params = SamplingParams(
      max_tokens=1,
      prompt_logprobs=1,  # Get logprobs for each prompt token
      temperature=0.0,
  )
  
  print("Computing log probabilities...")
  outputs = llm.generate(prompts, sampling_params)
  
  # Extract log likelihoods for answer tokens
  all_log_likelihoods = []
  
  for i, (output, num_answer_tokens) in enumerate(zip(outputs, answer_token_counts)):
    prompt_logprobs = output.prompt_logprobs
    
    if prompt_logprobs is None or len(prompt_logprobs) == 0:
      all_log_likelihoods.append(-100.0)
      continue
    
    # prompt_logprobs is a list where each element corresponds to a token position
    # The last num_answer_tokens positions are the answer tokens
    # Each element is a dict mapping token_id -> Logprob object (or None for first token)
    
    answer_log_probs = []
    total_tokens = len(prompt_logprobs)
    answer_start = total_tokens - num_answer_tokens
    
    for pos in range(answer_start, total_tokens):
      if prompt_logprobs[pos] is not None:
        # Get the logprob of the actual token at this position
        # prompt_logprobs[pos] is a dict {token_id: Logprob}
        # We need the logprob of the token that was actually there
        for token_id, logprob_obj in prompt_logprobs[pos].items():
          answer_log_probs.append(logprob_obj.logprob)
          break  # Only one entry per position in prompt_logprobs
    
    # Debug: print first example
    if i == 0:
      print(f"\n[DEBUG] First example:")
      print(f"Question: {questions[0][:100]}...")
      print(f"Answer: {answers[0]}")
      print(f"Total tokens: {total_tokens}")
      print(f"Answer tokens: {num_answer_tokens}")
      print(f"Answer log probs: {answer_log_probs}")
      if answer_log_probs:
        print(f"Average log likelihood: {np.mean(answer_log_probs):.4f}")
    
    # Average log probability across answer tokens
    avg_log_likelihood = np.mean(answer_log_probs) if answer_log_probs else -100.0
    all_log_likelihoods.append(avg_log_likelihood)
  
  scores_array = np.array(all_log_likelihoods, dtype=np.float32)
  
  # Save scores
  os.makedirs(cache_dir, exist_ok=True)
  np.save(scores_file, scores_array)
  print(f"Saved confidence scores to {scores_file}")
  
  # Clean up GPU memory thoroughly to avoid interference with subsequent vLLM instances
  del llm
  gc.collect()  # Force garbage collection
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
    torch.cuda.empty_cache()
  
  return scores_array


def least_confident_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio, 
    scorer_llm_name, 
    seed=42,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select train and eval indices based on scorer LLM confidence using vLLM.
  
  This method selects examples where the scorer LLM is least confident about
  the correct answer. Lower confidence (lower log likelihood) means the model
  is more uncertain, so these are typically harder examples.
  
  The confidence is measured by computing the log likelihood of the correct
  answer tokens given the question as context using vLLM with tensor parallelism.
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    scorer_llm_name: Name of the LLM to use for scoring confidence (via vLLM)
    seed: Random seed for reproducibility (default: 42)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for saving confidence scores
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  cache_dir = os.path.join(opro_root, "confidence_scores", dataset_name)
  
  # Extract questions and answers from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  answers = _extract_short_answers_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  if len(answers) != num_examples:
    raise ValueError(
        f"Mismatch between number of questions ({num_examples}) "
        f"and answers ({len(answers)})"
    )
  
  print(f"\n{'='*60}")
  print(f"Least Confident Subset Selection for {dataset_name}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Scorer LLM: {scorer_llm_name} (via vLLM)")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"Cache directory: {cache_dir}")
  print(f"{'='*60}\n")
  
  # Step 1: Compute or load confidence scores using vLLM
  confidence_scores = _compute_confidence_scores_vllm(
      questions, 
      answers, 
      scorer_llm_name, 
      dataset_name, 
      cache_dir,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Step 2: Sort by confidence (ascending = least confident first)
  sorted_indices = np.argsort(confidence_scores)
  
  # Step 3: Select training set from least confident examples
  k_train = int(train_ratio * num_examples)
  train_index = np.sort(sorted_indices[:k_train])
  
  # Step 4: Select eval indices from remaining examples
  remaining_indices = sorted_indices[k_train:]
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  # Print statistics
  train_scores = confidence_scores[train_index]
  print(f"\n{'='*60}")
  print(f"Selection complete!")
  print(f"Training set size: {len(train_index)}")
  print(f"Eval set size: {len(eval_index)}")
  print(f"Training set confidence scores (log likelihood):")
  print(f"  Mean: {train_scores.mean():.4f}")
  print(f"  Min: {train_scores.min():.4f}")
  print(f"  Max: {train_scores.max():.4f}")
  print(f"{'='*60}\n")
  
  # Sanity check: Print first 5 selected training examples with their confidence scores
  print(f"\n{'='*60}")
  print("SANITY CHECK: First 5 selected training examples")
  print(f"{'='*60}")
  num_to_show = min(5, len(train_index))
  for i in range(num_to_show):
    idx = train_index[i]
    score = confidence_scores[idx]
    question = questions[idx]
    answer = answers[idx]
    
    print(f"\nExample {i+1} (Index {idx}):")
    print(f"  Question: {question}")
    print(f"  Answer: {answer}")
    print(f"  Confidence score: {score:.4f}")
  
  print(f"\n{'='*60}\n")
  
  return train_index, eval_index


def _compute_verbal_confidence_scores_vllm(
    questions, 
    model_name, 
    dataset_name, 
    cache_dir, 
    k=4,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Compute verbal confidence scores using vLLM with tensor parallelism.
  
  Uses the verbalized confidence approach from "Just Ask for Calibration" 
  (Tian et al., 2023) where the model is asked to provide k guesses with 
  probabilities. For robustness, we:
  
  1. Generate 10 samples per question (with temperature=0.5)
  2. For each sample, parse all probabilities (P1-Pk) and take the maximum
  3. Average the max probabilities across all 10 samples
  
  This averaging approach gives a more stable confidence estimate than a 
  single deterministic sample, reducing noise from stochastic generation.
  
  Note: The probabilities do NOT need to sum to 1. Each P_i represents the
  model's independent estimate of the probability that guess G_i is correct.
  
  Args:
    questions: List of question strings
    model_name: Model name for vLLM
    dataset_name: Name of dataset (for caching)
    cache_dir: Directory to save/load confidence scores
    k: Number of guesses to request
    tensor_parallel_size: Number of GPUs for tensor parallelism
    gpu_memory_utilization: Fraction of GPU memory to use
    
  Returns:
    numpy array of confidence scores (averaged max probabilities), shape (num_examples,)
  """
  from vllm import LLM, SamplingParams
  import torch
  
  # Create a safe filename from the model name
  safe_model_name = model_name.replace("/", "_").replace(":", "_")
  scores_file = os.path.join(
      cache_dir, 
      f"{dataset_name}_{safe_model_name}_verbal_confidence_scores_k{k}.npy"
  )
  
  # Check if scores already exist
  if os.path.exists(scores_file):
    cached_scores = np.load(scores_file)
    # Validate cache size matches current data
    if cached_scores.shape[0] == len(questions):
      print(f"Loading existing verbal confidence scores from {scores_file}")
      return cached_scores
    else:
      print(f"Cache size mismatch: cached {cached_scores.shape[0]} vs current {len(questions)}. Regenerating...")
      del cached_scores
  
  # Determine if this is a multiple choice dataset
  is_multiple_choice = _is_multiple_choice_dataset(dataset_name)
  
  print(f"Computing verbal confidence scores for {len(questions)} examples...")
  print(f"Model: {model_name}")
  print(f"Dataset type: {'multiple choice (A/B/C/D)' if is_multiple_choice else 'numeric/free-form'}")
  
  # Build prompt template based on the paper's Verb. 1S top-k method
  # Reference: https://arxiv.org/pdf/2305.14975
  # Use different examples for multiple choice vs numeric datasets
  
  if is_multiple_choice:
    # Multiple choice prompt template with A/B/C/D example
    # Based on "Just Ask for Calibration" (Tian et al., 2023) - Verb. 1S top-k method
    prompt_template = f"""Provide your {k} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:

G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>
P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>
G2: <second most likely guess, as short as possible; not a complete sentence, just the guess!>
P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>
G3: <third most likely guess, as short as possible; not a complete sentence, just the guess!>
P3: <the probability between 0.0 and 1.0 that G3 is correct, without any extra commentary whatsoever; just the probability!>
G4: <fourth most likely guess, as short as possible; not a complete sentence, just the guess!>
P4: <the probability between 0.0 and 1.0 that G4 is correct, without any extra commentary whatsoever; just the probability!>

The question is: """
  else:
    # Numeric/free-form prompt template
    # Based on "Just Ask for Calibration" (Tian et al., 2023) - Verb. 1S top-k method
    prompt_template = f"""Provide your {k} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:

G1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>
P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>
G2: <second most likely guess, as short as possible; not a complete sentence, just the guess!>
P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>
G3: <third most likely guess, as short as possible; not a complete sentence, just the guess!>
P3: <the probability between 0.0 and 1.0 that G3 is correct, without any extra commentary whatsoever; just the probability!>
G4: <fourth most likely guess, as short as possible; not a complete sentence, just the guess!>
P4: <the probability between 0.0 and 1.0 that G4 is correct, without any extra commentary whatsoever; just the probability!>

The question is: """
  
  prompts = [prompt_template + q for q in questions]
  
  # Force tensor_parallel_size=1 to avoid NCCL conflicts with other multi-GPU operations
  # (e.g., optimizer on GPU 0, workers on GPUs 1-7). The 7B model fits on a single GPU
  # and this is a one-time operation, so single-GPU is acceptable.
  actual_tp_size = 1
  print(f"Initializing vLLM... (tensor_parallel_size={actual_tp_size}, forced single-GPU)")
  llm = LLM(
      model=model_name, 
      tensor_parallel_size=actual_tp_size,
      gpu_memory_utilization=gpu_memory_utilization,
      trust_remote_code=True
  )
  # Generate 10 samples per prompt and average the max probabilities
  # This gives a more robust confidence estimate than a single sample
  num_samples = 10
  sampling_params = SamplingParams(
      temperature=0.5,  # Moderate temperature for diversity while keeping reasonable outputs
      max_tokens=300,   # k=4 guesses need ~100-150 tokens max
      n=num_samples,    # Generate 10 samples per prompt
  )
  
  # Pattern to match P followed by a digit and a number (e.g., P1: 0.92, P2: 0.05)
  prob_pattern = re.compile(r'P\d:\s*([0-9]*\.?[0-9]+)')
  
  def parse_probabilities(response_text):
    """Parse probabilities from response. Returns (max_prob, sum_prob, success)."""
    matches = prob_pattern.findall(response_text)
    if not matches:
      return None, None, False
    try:
      probabilities = []
      for match in matches:
        prob = float(match)
        # Clamp to [0, 1]
        prob = max(0.0, min(1.0, prob))
        probabilities.append(prob)
      return max(probabilities), sum(probabilities), True
    except ValueError:
      return None, None, False
  
  # Generate responses (10 samples per prompt)
  print(f"Generating {num_samples} samples per prompt with vLLM (temperature={sampling_params.temperature})...")
  outputs = llm.generate(prompts, sampling_params)
  
  # For each question, parse all 10 samples and average the max probabilities
  # This gives a more robust confidence estimate
  confidence_scores = []
  total_parse_failures = 0
  sum_warnings = 0
  
  for i, output in enumerate(outputs):
    max_probs = []
    sample_parse_failures = 0
    
    # Parse all n samples for this prompt
    for sample_output in output.outputs:
      response = sample_output.text
      max_prob, sum_prob, success = parse_probabilities(response)
      
      if success:
        max_probs.append(max_prob)
        # Warn if probabilities sum to > 1.5 (indicates poor calibration)
        if sum_prob > 1.5:
          sum_warnings += 1
          if sum_warnings <= 3:
            print(f"Warning: Probabilities sum to {sum_prob:.2f} for example {i} (>1.5 suggests poor calibration)")
      else:
        sample_parse_failures += 1
    
    # Average the successfully parsed max probabilities
    if max_probs:
      avg_confidence = sum(max_probs) / len(max_probs)
      confidence_scores.append(avg_confidence)
      if sample_parse_failures > 0:
        total_parse_failures += sample_parse_failures
        if i < 5:  # Show first few examples with partial failures
          print(f"Example {i}: {len(max_probs)}/{num_samples} samples parsed successfully, avg confidence = {avg_confidence:.4f}")
    else:
      # All samples failed to parse - use default of 0.5
      confidence_scores.append(0.5)
      total_parse_failures += num_samples
      print(f"Warning: All {num_samples} samples failed to parse for example {i}. Using default 0.5.")
      if output.outputs:
        print(f"  Sample response: {output.outputs[0].text[:200]}...")
  
  if total_parse_failures > 0:
    print(f"Total parse failures across all samples: {total_parse_failures}/{len(questions) * num_samples}")
  
  if sum_warnings > 3:
    print(f"Total samples with probability sum > 1.5: {sum_warnings}")
  
  scores_array = np.array(confidence_scores, dtype=np.float32)
  
  # Save scores
  os.makedirs(cache_dir, exist_ok=True)
  np.save(scores_file, scores_array)
  print(f"Saved verbal confidence scores to {scores_file}")
  
  # Clean up GPU memory thoroughly to avoid interference with subsequent vLLM instances
  del llm
  gc.collect()  # Force garbage collection
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
    torch.cuda.empty_cache()
  
  return scores_array


def verbal_least_confident_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio,
    scorer_llm_name, 
    k=4, 
    seed=42,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select train and eval indices based on verbalized confidence using vLLM.
  
  This method uses the verbalized confidence approach from the paper
  "Just Ask for Calibration" (Tian et al., 2023, https://arxiv.org/pdf/2305.14975)
  where the model is prompted to provide k guesses with probabilities. 
  The maximum probability across all guesses (max of P1...Pk) is used as 
  the confidence score, which is robust to ordering variations in model output.
  
  Note: Probabilities do NOT need to sum to 1 - each P_i is the model's
  independent estimate of the probability that guess G_i is correct.
  
  Examples where the model is least confident are selected for training,
  as these are typically harder examples that benefit most from optimization.
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    scorer_llm_name: Name of the LLM to use for scoring (via vLLM)
    k: Number of guesses to request from the model (default: 4)
    seed: Random seed for reproducibility (default: 42)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for saving confidence scores
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  cache_dir = os.path.join(opro_root, "verbal_confidence_scores", dataset_name)
  
  # Extract questions from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  print(f"\n{'='*60}")
  print(f"Verbal Least Confident Subset Selection for {dataset_name}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Scorer LLM: {scorer_llm_name} (via vLLM)")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"Number of guesses (k): {k}")
  print(f"Cache directory: {cache_dir}")
  print(f"{'='*60}\n")
  
  # Step 1: Compute or load verbal confidence scores using vLLM
  confidence_scores = _compute_verbal_confidence_scores_vllm(
      questions, 
      scorer_llm_name, 
      dataset_name, 
      cache_dir, 
      k=k,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Step 2: Sort by confidence (ascending = least confident first)
  sorted_indices = np.argsort(confidence_scores)
  
  # Step 3: Select training set from least confident examples
  k_train = int(train_ratio * num_examples)
  train_index = np.sort(sorted_indices[:k_train])
  
  # Step 4: Select eval indices from remaining examples
  remaining_indices = sorted_indices[k_train:]
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  # Print statistics
  train_scores = confidence_scores[train_index]
  print(f"\n{'='*60}")
  print(f"Selection complete!")
  print(f"Training set size: {len(train_index)}")
  print(f"Eval set size: {len(eval_index)}")
  print(f"Training set confidence scores (avg of max verbalized probabilities over 10 samples):")
  print(f"  Mean: {train_scores.mean():.4f}")
  print(f"  Min: {train_scores.min():.4f}")
  print(f"  Max: {train_scores.max():.4f}")
  print(f"{'='*60}\n")
  
  # Sanity check: Print first 5 selected training examples with their confidence scores
  print(f"\n{'='*60}")
  print("SANITY CHECK: First 5 selected training examples")
  print(f"{'='*60}")
  num_to_show = min(5, len(train_index))
  for i in range(num_to_show):
    idx = train_index[i]
    score = confidence_scores[idx]
    question = questions[idx]
    
    print(f"\nExample {i+1} (Index {idx}):")
    print(f"  Question: {question}")
    print(f"  Verbal confidence (avg of 10 samples): {score:.4f}")
  
  print(f"\n{'='*60}\n")
  
  return train_index, eval_index


def _weighted_greedy_subset_selection(M, weights, k, verbose=True):
  """Greedy subset selection with importance weights on coverage.
  
  Modified facility location objective:
    max_S Σ_i w_i * max_{j∈S} sim(i,j)
  
  This prioritizes covering high-weight (hard) examples.
  
  Args:
    M: Similarity matrix of shape (N, N), values in [0,1]
    weights: Importance weights of shape (N,), higher = more important to cover
    k: Subset size
    verbose: Print progress if True
    
  Returns:
    List of selected indices
  """
  N = M.shape[0]
  S = []  # Selected subset
  remaining = set(range(N))  # Candidates not yet selected
  current_max = np.zeros(N)  # Current coverage for each point
  
  if verbose:
    print(f"Running weighted greedy selection to select {k} samples...")
  
  for iteration in tqdm(range(k), desc="Weighted greedy selection", disable=not verbose):
    best_gain = -1.0
    best_i = None
    
    # Find the point that maximizes weighted marginal gain
    for i in remaining:
      row_i = M[i]  # Similarity from i to all points
      gain_vector = np.maximum(current_max, row_i) - current_max
      # Weight by importance: prioritize covering hard examples
      delta = (weights * gain_vector).sum()
      
      if delta > best_gain:
        best_gain = delta
        best_i = i
    
    # Add best point to subset
    i_star = best_i
    S.append(i_star)
    remaining.remove(i_star)
    
    # Update coverage
    current_max = np.maximum(current_max, M[i_star])
  
  if verbose:
    print("Weighted greedy selection complete.")
  
  return S


def confidence_weighted_representative_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio,
    scorer_llm_name, 
    alpha=0.9, 
    confidence_weight=0.5, 
    seed=42,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select subset that is both diverse AND focuses on hard examples.
  
  This method combines representative selection (diversity) with logit-based
  confidence scoring (difficulty) using a weighted facility location objective.
  
  The objective becomes:
    max_S Σ_i w_i * max_{j∈S} sim(i,j)
  
  where w_i = (1 - confidence_weight) + confidence_weight * (1 - normalized_confidence_i)
  
  This means:
  - Examples the model is confident about get lower weight in coverage
  - Examples the model struggles with get higher weight
  - We maximize coverage while prioritizing hard examples
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    scorer_llm_name: Name of the LLM to use for logit-based confidence scoring (via vLLM)
    alpha: Weight for dense similarity (1-alpha for lexical). Default 0.9.
    confidence_weight: Balance between diversity and difficulty. 
                       0 = pure representative (diversity only)
                       1 = heavily weight hard examples
                       Default 0.5 for balanced mix.
    seed: Random seed for reproducibility (default: 42)
    embedding_model: Model for generating embeddings (default: Qwen3-Embedding-8B)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for caching
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  rep_cache_dir = os.path.join(opro_root, "embeddings_and_sim_matrices", dataset_name)
  conf_cache_dir = os.path.join(opro_root, "confidence_scores", dataset_name)
  
  # Extract questions and answers from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  answers = _extract_short_answers_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  print(f"\n{'='*60}")
  print(f"Confidence-Weighted Representative Subset Selection")
  print(f"Dataset: {dataset_name}")
  print(f"{'='*60}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Alpha (dense vs lexical): {alpha}")
  print(f"Confidence weight: {confidence_weight}")
  print(f"  (0 = pure diversity, 1 = heavily weight hard examples)")
  print(f"Scorer LLM: {scorer_llm_name}")
  print(f"Embedding model: {embedding_model}")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"{'='*60}\n")
  
  # ===== Step 1: Build similarity matrix (from representative selection) =====
  print("=== Building similarity matrix ===")
  
  # Generate or load embeddings using vLLM
  embeddings = _generate_embeddings_vllm(
      questions, 
      dataset_name, 
      rep_cache_dir,
      embedding_model=embedding_model,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Compute or load dense similarity matrix
  sim_dense = _compute_dense_similarity(embeddings, dataset_name, rep_cache_dir, embedding_model)
  
  # Compute or load TF-IDF similarity matrix
  sim_tfidf = _compute_tfidf_similarity(questions, dataset_name, rep_cache_dir)
  
  # Apply normalizations
  print("Applying row-wise min-max normalization...")
  sim_dense_norm = _row_minmax_normalize(sim_dense)
  sim_tfidf_norm = _row_minmax_normalize(sim_tfidf)
  
  # Apply square root to TF-IDF to smooth out the distribution
  sim_tfidf_norm_sqrt = np.sqrt(sim_tfidf_norm)
  
  # Mix the similarity matrices
  print(f"Mixing similarity matrices with alpha={alpha}...")
  M_mixed = alpha * sim_dense_norm + (1 - alpha) * sim_tfidf_norm_sqrt
  
  # ===== Step 2: Get logit-based confidence scores =====
  print("\n=== Computing confidence scores via log probability ===")
  
  confidence_scores = _compute_confidence_scores_vllm(
      questions, 
      answers,
      scorer_llm_name, 
      dataset_name, 
      conf_cache_dir,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # ===== Step 3: Compute importance weights =====
  print("\n=== Computing importance weights ===")
  
  # Normalize confidence to [0, 1]
  conf_min = confidence_scores.min()
  conf_max = confidence_scores.max()
  conf_normalized = (confidence_scores - conf_min) / (conf_max - conf_min + 1e-8)
  
  # Compute importance weights: higher for low-confidence (hard) examples
  # w_i = (1 - confidence_weight) + confidence_weight * (1 - normalized_confidence)
  # When confidence_weight=0: all weights are 1.0 (pure diversity)
  # When confidence_weight=1: weights are (1 - normalized_confidence)
  importance_weights = (1 - confidence_weight) + confidence_weight * (1 - conf_normalized)
  
  print(f"Importance weight stats:")
  print(f"  Min: {importance_weights.min():.4f}")
  print(f"  Max: {importance_weights.max():.4f}")
  print(f"  Mean: {importance_weights.mean():.4f}")
  
  # ===== Step 4: Run weighted greedy selection =====
  print("\n=== Running weighted greedy selection ===")
  
  k_train = int(train_ratio * num_examples)
  train_index = _weighted_greedy_subset_selection(M_mixed, importance_weights, k_train, verbose=True)
  train_index = np.sort(np.array(train_index))
  
  # ===== Step 5: Select eval indices from remaining examples =====
  remaining_indices = list(set(range(num_examples)) - set(train_index))
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  # ===== Print statistics =====
  train_confidence = confidence_scores[train_index]
  print(f"\n{'='*60}")
  print(f"Selection complete!")
  print(f"Training set size: {len(train_index)}")
  print(f"Eval set size: {len(eval_index)}")
  print(f"Training set confidence (log likelihood):")
  print(f"  Mean: {train_confidence.mean():.4f}")
  print(f"  Min: {train_confidence.min():.4f}")
  print(f"  Max: {train_confidence.max():.4f}")
  print(f"{'='*60}\n")
  
  # Sanity check: Print first 5 selected training examples
  print(f"\n{'='*60}")
  print("SANITY CHECK: First 5 selected training examples")
  print(f"{'='*60}")
  num_to_show = min(5, len(train_index))
  for i in range(num_to_show):
    idx = train_index[i]
    conf = confidence_scores[idx]
    weight = importance_weights[idx]
    question = questions[idx]
    answer = answers[idx]
    
    print(f"\nExample {i+1} (Index {idx}):")
    print(f"  Question: {question[:200]}...")
    print(f"  Answer: {answer}")
    print(f"  Confidence (log likelihood): {conf:.4f}")
    print(f"  Importance weight: {weight:.4f}")
  
  print(f"\n{'='*60}\n")
  
  return train_index, eval_index


# ============================================================================
# IPOMP: Model Performance-Guided Evaluation Data Selection
# Based on: "Model Performance-Guided Evaluation Data Selection for Effective
#           Prompt Optimization" (Wu et al.)
# ============================================================================

def _detect_boundary_points(embeddings, k_neighbors=5):
    """Detect boundary points using Local Outlier Factor (LOF)-like approach.
    
    Boundary points are samples that lie on the edges of clusters,
    characterized by having neighbors from diverse clusters.
    
    Args:
        embeddings: numpy array of embeddings, shape (N, dim)
        k_neighbors: Number of neighbors to consider
        
    Returns:
        numpy array of boundary point indices
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    
    N = embeddings.shape[0]
    
    # First, cluster the data to identify cluster assignments
    n_clusters = min(10, N // 5)  # At most 10 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Find k nearest neighbors for each point
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, N), metric='cosine')
    nn.fit(embeddings)
    _, neighbor_indices = nn.kneighbors(embeddings)
    
    # Compute boundary score: diversity of cluster labels among neighbors
    boundary_scores = []
    for i in range(N):
        neighbors = neighbor_indices[i, 1:]  # Exclude self
        neighbor_labels = cluster_labels[neighbors]
        # Boundary score = number of unique clusters among neighbors
        unique_clusters = len(set(neighbor_labels))
        boundary_scores.append(unique_clusters)
    
    boundary_scores = np.array(boundary_scores)
    
    # Select points with high boundary scores (top 30%)
    threshold = np.percentile(boundary_scores, 70)
    boundary_indices = np.where(boundary_scores >= threshold)[0]
    
    return boundary_indices


def _select_furthest_pairs(embeddings, boundary_indices, num_pairs):
    """Select pairs of samples that are furthest apart in semantic distance.
    
    For efficiency, only considers pairs among boundary points.
    
    Args:
        embeddings: numpy array of embeddings, shape (N, dim)
        boundary_indices: Indices of boundary points to consider
        num_pairs: Number of samples to select (will select num_pairs individual samples)
        
    Returns:
        List of selected indices
    """
    if len(boundary_indices) == 0:
        return []
    
    # Compute pairwise distances among boundary points
    boundary_embeddings = embeddings[boundary_indices]
    
    # Normalize embeddings for cosine distance
    norms = np.linalg.norm(boundary_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    boundary_embeddings_norm = boundary_embeddings / norms
    
    # Compute cosine similarity matrix
    sim_matrix = boundary_embeddings_norm @ boundary_embeddings_norm.T
    # Convert to distance (dissimilarity)
    dist_matrix = 1 - sim_matrix
    
    # Greedy selection of diverse points
    selected_boundary_idx = []
    remaining = set(range(len(boundary_indices)))
    
    # Start with the two most distant points
    if len(boundary_indices) >= 2:
        # Find max distance pair
        max_dist = -1
        max_i, max_j = 0, 1
        for i in range(len(boundary_indices)):
            for j in range(i + 1, len(boundary_indices)):
                if dist_matrix[i, j] > max_dist:
                    max_dist = dist_matrix[i, j]
                    max_i, max_j = i, j
        
        selected_boundary_idx.append(max_i)
        selected_boundary_idx.append(max_j)
        remaining.discard(max_i)
        remaining.discard(max_j)
    
    # Greedily add points that are furthest from current selection
    while len(selected_boundary_idx) < num_pairs and remaining:
        best_idx = None
        best_min_dist = -1
        
        for idx in remaining:
            # Min distance to any selected point
            min_dist = min(dist_matrix[idx, sel_idx] for sel_idx in selected_boundary_idx)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        if best_idx is not None:
            selected_boundary_idx.append(best_idx)
            remaining.discard(best_idx)
        else:
            break
    
    # Convert boundary indices back to original indices
    selected_indices = [boundary_indices[i] for i in selected_boundary_idx]
    
    return selected_indices


def ipomp_initial_subset(
    dataset_name,
    raw_data,
    train_ratio,
    eval_ratio,
    seed=42,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90,
    n_clusters=10,
):
    """Initial subset selection for IPOMP method (Algorithm 1 from the paper).
    
    This method:
    1. Uses Sentence-BERT to encode all training questions
    2. Clusters using K-means with k=10
    3. Samples proportionately from each cluster (half the subset)
    4. Selects boundary cases (furthest pairs) for the other half
    
    Args:
        dataset_name: Name of the dataset
        raw_data: Raw data of the dataset
        train_ratio: Ratio of examples to use for training (0 to 1)
        eval_ratio: Ratio of examples to use for evaluation (0 to 1)
        seed: Random seed for reproducibility
        embedding_model: Model for generating embeddings
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        n_clusters: Number of clusters for K-means (default: 10)
        
    Returns:
        Tuple of (train_index, eval_index, embeddings) - embeddings returned for use by IPOMPManager
    """
    from sklearn.cluster import KMeans
    
    np.random.seed(seed)
    
    # Determine paths for saving embeddings
    opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    cache_dir = os.path.join(opro_root, "embeddings_and_sim_matrices", dataset_name)
    
    # Extract questions from raw_data
    questions = _extract_questions_from_raw_data(dataset_name, raw_data)
    num_examples = len(questions)
    
    print(f"\n{'='*60}")
    print(f"IPOMP Initial Subset Selection for {dataset_name}")
    print(f"Number of examples: {num_examples}")
    print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
    print(f"Embedding model: {embedding_model}")
    print(f"Number of clusters: {n_clusters}")
    print(f"{'='*60}\n")
    
    # Step 1: Generate or load embeddings
    embeddings = _generate_embeddings_vllm(
        questions,
        dataset_name,
        cache_dir,
        embedding_model=embedding_model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    # Step 2: K-means clustering
    print(f"Clustering into {n_clusters} clusters...")
    actual_n_clusters = min(n_clusters, num_examples)
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Count samples per cluster
    cluster_counts = np.bincount(cluster_labels, minlength=actual_n_clusters)
    print(f"Cluster sizes: {cluster_counts}")
    
    # Calculate target subset size
    k_train = int(train_ratio * num_examples)
    half_size = k_train // 2
    
    # Step 3: Proportional sampling from clusters (first half)
    print(f"Selecting {half_size} samples via proportional cluster sampling...")
    cluster_samples = []
    total_in_clusters = num_examples
    
    for cluster_id in range(actual_n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        # Proportional allocation
        n_from_cluster = int((len(cluster_indices) / total_in_clusters) * half_size)
        n_from_cluster = max(1, n_from_cluster)  # At least 1 from each cluster
        n_from_cluster = min(n_from_cluster, len(cluster_indices))  # Can't exceed cluster size
        
        # Randomly sample from this cluster
        selected = np.random.choice(cluster_indices, size=n_from_cluster, replace=False)
        cluster_samples.extend(selected)
    
    # Ensure we have exactly half_size samples (adjust if needed)
    cluster_samples = list(set(cluster_samples))[:half_size]
    
    # Step 4: Boundary case selection (second half)
    remaining_samples_needed = k_train - len(cluster_samples)
    print(f"Selecting {remaining_samples_needed} boundary cases...")
    
    # Detect boundary points
    boundary_indices = _detect_boundary_points(embeddings)
    print(f"Detected {len(boundary_indices)} boundary points")
    
    # Remove already selected samples from boundary indices
    boundary_indices = np.array([i for i in boundary_indices if i not in cluster_samples])
    
    # Select furthest pairs among boundary points
    boundary_samples = _select_furthest_pairs(
        embeddings, boundary_indices, remaining_samples_needed
    )
    
    # If not enough boundary samples, fill with random remaining samples
    if len(boundary_samples) < remaining_samples_needed:
        remaining_indices = list(set(range(num_examples)) - set(cluster_samples) - set(boundary_samples))
        additional_needed = remaining_samples_needed - len(boundary_samples)
        if remaining_indices and additional_needed > 0:
            additional = np.random.choice(
                remaining_indices,
                size=min(additional_needed, len(remaining_indices)),
                replace=False
            )
            boundary_samples.extend(additional)
    
    # Combine both halves
    train_index = np.sort(np.array(list(set(cluster_samples) | set(boundary_samples))))
    
    # Ensure we have exactly k_train samples
    if len(train_index) > k_train:
        train_index = train_index[:k_train]
    elif len(train_index) < k_train:
        remaining = list(set(range(num_examples)) - set(train_index))
        additional = np.random.choice(remaining, size=k_train - len(train_index), replace=False)
        train_index = np.sort(np.concatenate([train_index, additional]))
    
    # Step 5: Select eval indices from remaining examples
    remaining_indices = list(set(range(num_examples)) - set(train_index))
    k_eval = int(eval_ratio * num_examples)
    
    if k_eval > 0 and len(remaining_indices) > 0:
        eval_index = np.sort(
            np.array(
                np.random.choice(
                    remaining_indices,
                    size=min(k_eval, len(remaining_indices)),
                    replace=False,
                )
            )
        )
    else:
        eval_index = np.array([], dtype=int)
    
    print(f"\n{'='*60}")
    print(f"IPOMP Initial Selection complete!")
    print(f"Training set size: {len(train_index)}")
    print(f"  - From cluster sampling: {len(cluster_samples)}")
    print(f"  - From boundary cases: {len(boundary_samples)}")
    print(f"Eval set size: {len(eval_index)}")
    print(f"{'='*60}\n")
    
    # Return embeddings too for use by IPOMPManager
    return train_index, eval_index, embeddings


class IPOMPManager:
    """Manager for IPOMP dynamic subset updates during optimization.
    
    This class implements Algorithm 2 from the paper:
    1. Track performance of each sample across candidate prompts (using logits)
    2. Use hierarchical clustering to identify redundant samples (correlation > 0.9)
    3. Replace a portion (beta) of redundant samples with most dissimilar ones
    
    Attributes:
        embeddings: Pre-computed embeddings for all training samples
        full_train_index: Original full training index (for replacement sampling)
        current_train_index: Current active training subset
        performance_history: List of performance matrices from each iteration
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        full_train_index: np.ndarray,
        initial_train_index: np.ndarray,
        questions: list,
        answers: list,
        correlation_threshold: float = 0.9,
        replacement_ratio: float = 0.1,  # beta in the paper
        seed: int = 42,
    ):
        """Initialize IPOMP manager.
        
        Args:
            embeddings: Pre-computed embeddings for all examples, shape (N, dim)
            full_train_index: Full training index (all available training samples)
            initial_train_index: Initial subset selected by ipomp_initial_subset
            questions: List of all questions
            answers: List of all answers
            correlation_threshold: CT in the paper (default: 0.9)
            replacement_ratio: beta, fraction of redundant samples to replace (default: 0.1)
            seed: Random seed
        """
        self.embeddings = embeddings
        self.full_train_index = full_train_index
        self.current_train_index = np.array(initial_train_index, copy=True)
        self.questions = questions
        self.answers = answers
        self.correlation_threshold = correlation_threshold
        self.replacement_ratio = replacement_ratio
        self.seed = seed
        
        np.random.seed(seed)
        
        # Build HNSW index for efficient dissimilarity search
        self._build_hnsw_index()
        
        # Performance tracking
        self.performance_history = []  # List of (step, performance_matrix) tuples
        
    def _build_hnsw_index(self):
        """Build HNSW index for efficient nearest neighbor search.
        
        Uses hnswlib for fast approximate nearest neighbor search.
        We invert the similarity to find most dissimilar samples.
        """
        try:
            import hnswlib
        except ImportError:
            print("Warning: hnswlib not installed. Install with: pip install hnswlib")
            print("Falling back to brute-force search (slower)")
            self.hnsw_index = None
            return
        
        dim = self.embeddings.shape[1]
        num_elements = len(self.full_train_index)
        
        # Create HNSW index with cosine distance (inner product on normalized vectors)
        self.hnsw_index = hnswlib.Index(space='cosine', dim=dim)
        self.hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # Add embeddings for full training set
        full_embeddings = self.embeddings[self.full_train_index]
        self.hnsw_index.add_items(full_embeddings, self.full_train_index)
        
        # Set ef for search (higher = more accurate but slower)
        self.hnsw_index.set_ef(50)
        
        print(f"[IPOMPManager] Built HNSW index with {num_elements} elements")
    
    def get_current_questions(self) -> list:
        """Get questions for current training subset."""
        return [self.questions[i] for i in self.current_train_index]
    
    def get_current_answers(self) -> list:
        """Get answers for current training subset."""
        return [self.answers[i] for i in self.current_train_index]
    
    def get_current_train_index(self) -> np.ndarray:
        """Get current training indices."""
        return self.current_train_index.copy()
    
    def record_iteration_performance(
        self,
        step: int,
        candidate_prompts: list,
        per_sample_accuracies: list,
    ):
        """Record performance of samples across candidate prompts for this iteration.
        
        Args:
            step: Current optimization step
            candidate_prompts: List of candidate prompts evaluated this iteration
            per_sample_accuracies: List of dicts, each mapping sample_position -> accuracy
                                   for each candidate prompt
        """
        # Build performance matrix: rows = samples in current subset, cols = num_prompts
        # Use accuracy (0/1) as the performance proxy (simpler than logits)
        
        num_samples = len(self.current_train_index)
        num_prompts = len(candidate_prompts)
        
        if num_prompts == 0:
            return
        
        performance_matrix = np.zeros((num_samples, num_prompts))
        
        for prompt_idx, acc_dict in enumerate(per_sample_accuracies):
            for sample_pos in range(num_samples):
                if sample_pos in acc_dict:
                    performance_matrix[sample_pos, prompt_idx] = acc_dict[sample_pos]
        
        self.performance_history.append((step, performance_matrix))
        
        print(f"[IPOMPManager] Recorded performance for step {step}: "
              f"{num_samples} samples x {num_prompts} prompts")
    
    def update_subset(self, step: int) -> tuple:
        """Update the training subset based on performance patterns.
        
        Implements Algorithm 2 from the paper:
        1. Identify redundant samples using hierarchical clustering on performance correlation
        2. Select a portion (beta) of redundant samples for replacement
        3. Replace them with most dissimilar samples from the full training set
        
        Args:
            step: Current optimization step
            
        Returns:
            Tuple of (new_train_index, samples_replaced_count)
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        if len(self.performance_history) < 2:
            print(f"[IPOMPManager] Not enough history for update (need >= 2 iterations)")
            return self.current_train_index, 0
        
        # Use the most recent performance matrix
        _, perf_matrix = self.performance_history[-1]
        
        if perf_matrix.shape[1] < 2:
            print(f"[IPOMPManager] Not enough prompts for correlation analysis")
            return self.current_train_index, 0
        
        num_samples = perf_matrix.shape[0]
        
        if num_samples < 3:
            print(f"[IPOMPManager] Not enough samples for clustering")
            return self.current_train_index, 0
        
        # Compute correlation matrix between samples based on their performance patterns
        # Standardize rows (each sample's performance across prompts)
        perf_std = perf_matrix - perf_matrix.mean(axis=1, keepdims=True)
        std_dev = perf_matrix.std(axis=1, keepdims=True)
        std_dev = np.maximum(std_dev, 1e-8)  # Avoid divide by zero
        perf_std = perf_std / std_dev
        
        # Compute correlation matrix
        corr_matrix = perf_std @ perf_std.T / perf_matrix.shape[1]
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Convert correlation to distance for hierarchical clustering
        # High correlation -> low distance (redundant)
        dist_matrix = 1 - np.abs(corr_matrix)
        np.fill_diagonal(dist_matrix, 0)
        
        # Ensure distance matrix is valid (symmetric, no negative values)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        dist_matrix = np.maximum(dist_matrix, 0)
        
        # Hierarchical clustering
        try:
            condensed_dist = squareform(dist_matrix)
            linkage_matrix = linkage(condensed_dist, method='average')
            
            # Form clusters at correlation threshold
            # Distance threshold = 1 - correlation_threshold
            dist_threshold = 1 - self.correlation_threshold
            cluster_labels = fcluster(linkage_matrix, t=dist_threshold, criterion='distance')
        except Exception as e:
            print(f"[IPOMPManager] Clustering failed: {e}")
            return self.current_train_index, 0
        
        # Identify redundant samples: samples in clusters with > 1 member
        cluster_counts = np.bincount(cluster_labels)
        redundant_clusters = np.where(cluster_counts > 1)[0]
        
        # Collect redundant sample positions
        redundant_positions = []
        for cluster_id in redundant_clusters:
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            # Keep one sample from each cluster, mark others as redundant
            redundant_positions.extend(cluster_members[1:])  # Keep first, remove rest
        
        if not redundant_positions:
            print(f"[IPOMPManager] No redundant samples found")
            return self.current_train_index, 0
        
        # Select a portion (beta) of redundant samples for replacement
        num_to_replace = max(1, int(len(redundant_positions) * self.replacement_ratio))
        positions_to_replace = np.random.choice(
            redundant_positions, 
            size=min(num_to_replace, len(redundant_positions)),
            replace=False
        )
        
        samples_to_replace = self.current_train_index[positions_to_replace]
        
        print(f"[IPOMPManager] Found {len(redundant_positions)} redundant samples, "
              f"replacing {len(positions_to_replace)}")
        
        # Find most dissimilar replacements using HNSW
        replacement_samples = self._find_dissimilar_replacements(
            samples_to_replace,
            exclude_set=set(self.current_train_index)
        )
        
        # Update the training index
        new_train_index = self.current_train_index.copy()
        for pos, replacement in zip(positions_to_replace, replacement_samples):
            new_train_index[pos] = replacement
        
        self.current_train_index = np.sort(new_train_index)
        
        print(f"[IPOMPManager] Updated subset: replaced {len(replacement_samples)} samples")
        
        return self.current_train_index, len(replacement_samples)
    
    def _find_dissimilar_replacements(
        self,
        samples_to_replace: np.ndarray,
        exclude_set: set,
    ) -> list:
        """Find most dissimilar samples from the full training set.
        
        For each sample to replace, find the sample in the full training set
        that is most dissimilar (furthest in embedding space).
        
        Args:
            samples_to_replace: Indices of samples to replace
            exclude_set: Set of indices to exclude (current training subset)
            
        Returns:
            List of replacement sample indices
        """
        replacements = []
        available_indices = set(self.full_train_index) - exclude_set - set(replacements)
        
        for sample_idx in samples_to_replace:
            if not available_indices:
                break
            
            sample_embedding = self.embeddings[sample_idx:sample_idx+1]
            
            if self.hnsw_index is not None:
                # Use HNSW for efficient search
                # Note: HNSW finds nearest neighbors, so we query more and pick the furthest
                k = min(len(available_indices), 100)
                labels, distances = self.hnsw_index.knn_query(sample_embedding, k=k)
                
                # Find the most distant among available indices
                best_replacement = None
                best_distance = -1
                
                for label, dist in zip(labels[0], distances[0]):
                    if label in available_indices and dist > best_distance:
                        best_distance = dist
                        best_replacement = label
                
                if best_replacement is not None:
                    replacements.append(best_replacement)
                    available_indices.discard(best_replacement)
            else:
                # Brute force fallback
                available_list = list(available_indices)
                available_embeddings = self.embeddings[available_list]
                
                # Compute cosine distances
                sample_norm = sample_embedding / (np.linalg.norm(sample_embedding) + 1e-8)
                available_norms = available_embeddings / (np.linalg.norm(available_embeddings, axis=1, keepdims=True) + 1e-8)
                similarities = (sample_norm @ available_norms.T).flatten()
                
                # Find most dissimilar (lowest similarity)
                most_dissimilar_idx = np.argmin(similarities)
                replacement = available_list[most_dissimilar_idx]
                
                replacements.append(replacement)
                available_indices.discard(replacement)
        
        return replacements
    
    def get_stats(self) -> dict:
        """Get statistics about the current state."""
        return {
            "current_subset_size": len(self.current_train_index),
            "full_train_size": len(self.full_train_index),
            "num_iterations_recorded": len(self.performance_history),
            "correlation_threshold": self.correlation_threshold,
            "replacement_ratio": self.replacement_ratio,
        }


# ============================================================================
# Anchor Points: Benchmarking Models with Much Fewer Examples
# Based on: "Anchor Points: Benchmarking Models with Much Fewer Examples"
# (Vivek et al., 2023) - https://arxiv.org/abs/2309.08638
# ============================================================================

# Default list of diverse small source models for Anchor Points
# These models are chosen to be:
# 1. Small enough to fit on a single GPU
# 2. Diverse in architecture and training data
# 3. Publicly available on HuggingFace
# The scorer model will be automatically excluded from this list.
DEFAULT_ANCHOR_SOURCE_MODELS = [
    # Qwen2.5 family (various sizes for diversity)
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # Llama 3.2 family
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # Gemma family
    "google/gemma-2-2b-it",
    # Phi family
    "microsoft/Phi-3.5-mini-instruct",
    # SmolLM family
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # OLMo family
    "allenai/OLMo-1B-hf",
    # Pythia family (diverse training)
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    # StableLM family
    "stabilityai/stablelm-2-1_6b-chat",
    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # Falcon
    "tiiuae/falcon-rw-1b",
    # MiniCPM
    "openbmb/MiniCPM-1B-sft-bf16",
]


def _compute_single_model_confidence_worker(
    gpu_id: int,
    model_name: str,
    questions: list,
    answers: list,
    is_multiple_choice: bool,
    gpu_memory_utilization: float,
    result_queue,
    model_index: int,
):
    """Worker function to compute confidences for a single model on a specific GPU.
    
    This function runs in a separate process to compute correct-class confidences
    for one source model.
    
    Args:
        gpu_id: GPU to use (0-indexed)
        model_name: HuggingFace model name to load
        questions: List of question strings
        answers: List of correct answer strings
        is_multiple_choice: Whether this is a multiple choice dataset
        gpu_memory_utilization: Fraction of GPU memory to use
        result_queue: Queue to put results into
        model_index: Index of this model in the source models list
    """
    import os
    import signal
    
    # Reset signal handlers to default for worker process
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    
    # Set CUDA_VISIBLE_DEVICES before importing torch/vllm
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Set unique MASTER_PORT to avoid conflicts
    base_port = 29500
    os.environ["MASTER_PORT"] = str(base_port + gpu_id * 100 + 50)  # Offset from main workers
    
    print(f"[AnchorWorker {gpu_id}] Starting for model {model_name}")
    
    try:
        from vllm import LLM, SamplingParams
        import numpy as np
        
        # Initialize vLLM
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        tokenizer = llm.get_tokenizer()
        
        # Build prompts
        prompts = []
        answer_token_counts = []
        
        for q, a in zip(questions, answers):
            if is_multiple_choice:
                prompt = f"Directly give the choice A or B or C or D: {q}\nAnswer: "
            else:
                prompt = f"Directly give the numeric answer to the following question: {q}\n\nAnswer: "
            
            full_text = prompt + a
            prompts.append(full_text)
            
            answer_tokens = tokenizer.encode(a, add_special_tokens=False)
            answer_token_counts.append(len(answer_tokens))
        
        # Compute confidences
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0.1,  # Low temperature for more deterministic outputs
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        confidences = np.zeros(len(questions))
        
        for i, (output, num_answer_tokens) in enumerate(zip(outputs, answer_token_counts)):
            prompt_logprobs = output.prompt_logprobs
            
            if prompt_logprobs is None or len(prompt_logprobs) == 0:
                confidences[i] = 0.5  # Default
                continue
            
            # Get log probs for answer tokens
            answer_log_probs = []
            total_tokens = len(prompt_logprobs)
            answer_start = total_tokens - num_answer_tokens
            
            for pos in range(answer_start, total_tokens):
                if prompt_logprobs[pos] is not None:
                    for token_id, logprob_obj in prompt_logprobs[pos].items():
                        answer_log_probs.append(logprob_obj.logprob)
                        break
            
            # Convert log prob to probability (confidence)
            if answer_log_probs:
                avg_log_prob = np.mean(answer_log_probs)
                confidence = np.exp(avg_log_prob)
                confidence = np.clip(confidence, 1e-6, 1 - 1e-6)
            else:
                confidence = 0.5
            
            confidences[i] = confidence
        
        # Cleanup
        del llm
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        print(f"[AnchorWorker {gpu_id}] Completed {model_name}")
        result_queue.put({
            "model_index": model_index,
            "model_name": model_name,
            "confidences": confidences,
            "status": "success",
        })
        
    except Exception as e:
        print(f"[AnchorWorker {gpu_id}] Error with {model_name}: {e}")
        result_queue.put({
            "model_index": model_index,
            "model_name": model_name,
            "confidences": None,
            "status": "error",
            "error": str(e),
        })


def _compute_source_model_confidences_parallel(
    questions,
    answers,
    source_models,
    dataset_name,
    cache_dir,
    is_multiple_choice=False,
    num_workers=7,
    gpu_memory_utilization=0.90,
):
    """Compute correct-class confidences from multiple diverse source models in parallel.
    
    This function uses actual different small LMs as source models, processing
    them in parallel batches based on the number of available workers/GPUs.
    
    Args:
        questions: List of question strings
        answers: List of correct answer strings
        source_models: List of model names to use as source models
        dataset_name: Name of dataset (for caching)
        cache_dir: Directory to save/load confidence scores
        is_multiple_choice: Whether this is a multiple choice dataset
        num_workers: Number of parallel workers (GPUs) to use
        gpu_memory_utilization: Fraction of GPU memory to use
        
    Returns:
        numpy array of shape (num_examples, num_source_models) containing
        correct-class confidences
    """
    import multiprocessing as mp
    
    num_source_models = len(source_models)
    
    # Create a cache key based on the source models used
    model_hash = "_".join([m.split("/")[-1][:10] for m in source_models[:5]])
    cache_file = os.path.join(
        cache_dir,
        f"{dataset_name}_anchor_confidences_n{num_source_models}_{model_hash}.npy"
    )
    
    # Check if fully cached
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        if cached.shape == (len(questions), num_source_models):
            print(f"Loading cached anchor point confidences from {cache_file}")
            return cached
        else:
            print(f"Cache shape mismatch: {cached.shape} vs ({len(questions)}, {num_source_models}), regenerating...")
    
    # Check for individual model caches
    all_confidences = np.zeros((len(questions), num_source_models))
    models_to_compute = []
    
    for idx, model_name in enumerate(source_models):
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        model_cache_file = os.path.join(
            cache_dir,
            f"{dataset_name}_{safe_model_name}_confidences.npy"
        )
        
        if os.path.exists(model_cache_file):
            cached = np.load(model_cache_file)
            if len(cached) == len(questions):
                print(f"Loading cached confidences for {model_name}")
                all_confidences[:, idx] = cached
                continue
        
        models_to_compute.append((idx, model_name))
    
    if not models_to_compute:
        # All models were cached
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_file, all_confidences)
        return all_confidences
    
    print(f"\n{'='*60}")
    print(f"Computing anchor point confidences for {len(questions)} examples")
    print(f"Using {num_source_models} diverse source models")
    print(f"Models to compute: {len(models_to_compute)} (others cached)")
    print(f"Parallel workers: {num_workers}")
    print(f"{'='*60}\n")
    
    # Process models in batches based on num_workers
    # Use spawn context for proper CUDA isolation
    ctx = mp.get_context("spawn")
    
    batch_idx = 0
    while batch_idx < len(models_to_compute):
        batch = models_to_compute[batch_idx:batch_idx + num_workers]
        
        print(f"\nProcessing batch {batch_idx // num_workers + 1}/{(len(models_to_compute) + num_workers - 1) // num_workers}")
        print(f"Models in this batch: {[m[1] for m in batch]}")
        
        result_queue = ctx.Queue()
        processes = []
        
        for worker_idx, (model_idx, model_name) in enumerate(batch):
            gpu_id = worker_idx  # Use GPUs 0, 1, 2, ... for anchor computation
            
            p = ctx.Process(
                target=_compute_single_model_confidence_worker,
                args=(
                    gpu_id,
                    model_name,
                    questions,
                    answers,
                    is_multiple_choice,
                    gpu_memory_utilization,
                    result_queue,
                    model_idx,
                )
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results_collected = 0
        while results_collected < len(batch):
            try:
                result = result_queue.get(timeout=1800)  # 30 min timeout per model
                results_collected += 1
                
                if result["status"] == "success":
                    model_idx = result["model_index"]
                    model_name = result["model_name"]
                    confidences = result["confidences"]
                    
                    all_confidences[:, model_idx] = confidences
                    
                    # Save individual model cache
                    safe_model_name = model_name.replace("/", "_").replace(":", "_")
                    model_cache_file = os.path.join(
                        cache_dir,
                        f"{dataset_name}_{safe_model_name}_confidences.npy"
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(model_cache_file, confidences)
                    
                    print(f"Collected results for {model_name} ({results_collected}/{len(batch)})")
                else:
                    print(f"Error for model {result['model_name']}: {result.get('error', 'unknown')}")
                    # Use default confidence of 0.5 for failed models
                    all_confidences[:, result["model_index"]] = 0.5
                    
            except Exception as e:
                print(f"Timeout or error collecting results: {e}")
                break
        
        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join()
        
        batch_idx += num_workers
    
    # Save combined cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_file, all_confidences)
    print(f"\nSaved combined anchor point confidences to {cache_file}")
    
    return all_confidences


def _compute_anchor_correlation_matrix(confidences):
    """Compute Pearson correlation matrix from logit-transformed confidences.
    
    Following the paper: apply logit transform to scale [0,1] to [-inf, +inf],
    then compute Pearson correlation between sample pairs across source models.
    
    Args:
        confidences: numpy array of shape (num_examples, num_source_models)
        
    Returns:
        Correlation matrix of shape (num_examples, num_examples)
    """
    # Logit transform: logit(p) = log(p / (1-p))
    # Clip to avoid inf values
    eps = 1e-6
    conf_clipped = np.clip(confidences, eps, 1 - eps)
    logits = np.log(conf_clipped / (1 - conf_clipped))
    
    # Standardize each row (sample) across source models
    mean = logits.mean(axis=1, keepdims=True)
    std = logits.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-8)  # Avoid divide by zero
    logits_std = (logits - mean) / std
    
    # Compute Pearson correlation matrix
    # corr(i, j) = (1/n) * sum_k((x_ik - mean_i)/std_i * (x_jk - mean_j)/std_j)
    n_sources = logits.shape[1]
    corr_matrix = (logits_std @ logits_std.T) / n_sources
    
    # Ensure diagonal is 1 and values are in [-1, 1]
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    return corr_matrix


def _kmedoids_pam(distance_matrix, n_clusters, max_iter=100, random_state=42):
    """K-Medoids clustering using PAM (Partitioning Around Medoids) algorithm.
    
    Args:
        distance_matrix: Symmetric distance matrix of shape (n, n)
        n_clusters: Number of medoids to select
        max_iter: Maximum iterations
        random_state: Random seed
        
    Returns:
        Tuple of (medoid_indices, labels) where:
        - medoid_indices: array of indices of selected medoids
        - labels: cluster assignment for each sample
    """
    np.random.seed(random_state)
    n = distance_matrix.shape[0]
    
    if n_clusters >= n:
        # If requesting more clusters than samples, return all samples
        return np.arange(n), np.arange(n)
    
    # Initialize medoids randomly
    medoid_indices = np.random.choice(n, size=n_clusters, replace=False)
    
    def assign_clusters(medoids):
        """Assign each point to nearest medoid."""
        distances_to_medoids = distance_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        return labels
    
    def compute_cost(medoids, labels):
        """Compute total cost (sum of distances to medoids)."""
        total_cost = 0
        for i, label in enumerate(labels):
            total_cost += distance_matrix[i, medoids[label]]
        return total_cost
    
    labels = assign_clusters(medoid_indices)
    current_cost = compute_cost(medoid_indices, labels)
    
    for iteration in range(max_iter):
        improved = False
        
        # Try swapping each medoid with each non-medoid
        for m_idx in range(n_clusters):
            current_medoid = medoid_indices[m_idx]
            best_swap = None
            best_cost = current_cost
            
            non_medoids = [i for i in range(n) if i not in medoid_indices]
            
            for candidate in non_medoids:
                # Try swapping
                new_medoids = medoid_indices.copy()
                new_medoids[m_idx] = candidate
                new_labels = assign_clusters(new_medoids)
                new_cost = compute_cost(new_medoids, new_labels)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_swap = candidate
            
            if best_swap is not None:
                medoid_indices[m_idx] = best_swap
                labels = assign_clusters(medoid_indices)
                current_cost = best_cost
                improved = True
        
        if not improved:
            break
    
    return medoid_indices, labels


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name to a canonical form for comparison.
    
    Handles variations like:
    - "Qwen/Qwen2.5-7B-Instruct" vs "Qwen2.5-7B-Instruct"
    - Case differences
    
    Args:
        model_name: Model name (may be HuggingFace path or just model name)
        
    Returns:
        Normalized model name (lowercase, just the model portion)
    """
    # Get just the model name part (after the last /)
    name = model_name.split("/")[-1].lower()
    return name


def anchor_points_subset(
    dataset_name,
    raw_data,
    train_ratio,
    eval_ratio,
    scorer_llm_name,
    seed=42,
    num_source_models=None,
    num_workers=7,
    gpu_memory_utilization=0.90,
    source_models=None,
):
    """Select anchor points using the method from Vivek et al., 2023.
    
    This method:
    1. Computes correct-class confidences from multiple diverse source models
       (actual different small LMs, NOT the scorer model)
    2. Applies logit transform and computes Pearson correlations
    3. Uses K-medoids (PAM) to select representative anchor points
    
    Args:
        dataset_name: Name of the dataset
        raw_data: Raw data of the dataset
        train_ratio: Ratio of examples to use for training (0 to 1)
        eval_ratio: Ratio of examples to use for evaluation (0 to 1)
        scorer_llm_name: The scorer model (will be EXCLUDED from source models)
        seed: Random seed for reproducibility
        num_source_models: Number of source models to use. If None, uses all
            available source models (excluding scorer). Default: None.
        num_workers: Number of parallel workers (GPUs) to use for computing
            confidences. Default: 7.
        gpu_memory_utilization: Fraction of GPU memory to use
        source_models: Optional list of source models to use. If None, uses
            DEFAULT_ANCHOR_SOURCE_MODELS. The scorer model will be automatically
            excluded from this list.
        
    Returns:
        Tuple of (train_index, eval_index, cluster_sizes) where:
        - train_index: indices of selected anchor points
        - eval_index: indices for evaluation
        - cluster_sizes: size of each cluster (for APW scoring)
    """
    np.random.seed(seed)
    
    # Determine cache directory
    opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    cache_dir = os.path.join(opro_root, "anchor_point_scores", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get source models list
    if source_models is None:
        source_models = DEFAULT_ANCHOR_SOURCE_MODELS.copy()
    else:
        source_models = list(source_models)  # Make a copy
    
    # Filter out the scorer model from source models
    # The scorer model should NOT be a source model
    scorer_normalized = _normalize_model_name(scorer_llm_name)
    original_count = len(source_models)
    source_models = [
        m for m in source_models 
        if _normalize_model_name(m) != scorer_normalized
    ]
    
    if len(source_models) < original_count:
        print(f"Excluded scorer model '{scorer_llm_name}' from source models")
    
    # Limit to num_source_models if specified
    if num_source_models is not None and num_source_models < len(source_models):
        source_models = source_models[:num_source_models]
    
    # Extract questions and answers
    questions = _extract_questions_from_raw_data(dataset_name, raw_data)
    answers = _extract_short_answers_from_raw_data(dataset_name, raw_data)
    num_examples = len(questions)
    
    is_multiple_choice = _is_multiple_choice_dataset(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"Anchor Points Subset Selection for {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of examples: {num_examples}")
    print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
    print(f"Scorer LLM: {scorer_llm_name} (EXCLUDED from source models)")
    print(f"Number of source models: {len(source_models)}")
    print(f"Parallel workers: {num_workers}")
    print(f"Dataset type: {'multiple choice' if is_multiple_choice else 'numeric'}")
    print(f"\nSource models:")
    for i, m in enumerate(source_models, 1):
        print(f"  {i:2d}. {m}")
    print(f"{'='*60}\n")
    
    # Step 1: Compute confidences from source models in parallel
    confidences = _compute_source_model_confidences_parallel(
        questions=questions,
        answers=answers,
        source_models=source_models,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        is_multiple_choice=is_multiple_choice,
        num_workers=num_workers,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    print(f"\nConfidence matrix shape: {confidences.shape}")
    print(f"Confidence stats: mean={confidences.mean():.4f}, std={confidences.std():.4f}")
    
    # Step 2: Compute correlation matrix
    print("Computing Pearson correlation matrix on logit-transformed confidences...")
    corr_matrix = _compute_anchor_correlation_matrix(confidences)
    
    # Step 3: Convert correlation to distance (1 - correlation)
    distance_matrix = 1 - corr_matrix
    # Ensure non-negative distances
    distance_matrix = np.maximum(distance_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)
    
    print(f"Distance matrix: min={distance_matrix.min():.4f}, max={distance_matrix.max():.4f}")
    
    # Step 4: K-medoids to select anchor points
    k_train = int(train_ratio * num_examples)
    print(f"Selecting {k_train} anchor points using K-medoids (PAM)...")
    
    try:
        # Try to use sklearn_extra if available (more efficient)
        from sklearn_extra.cluster import KMedoids
        kmedoids = KMedoids(
            n_clusters=k_train,
            metric='precomputed',
            init='k-medoids++',
            max_iter=100,
            random_state=seed,
        )
        kmedoids.fit(distance_matrix)
        train_index = kmedoids.medoid_indices_
        labels = kmedoids.labels_
        print("Using sklearn_extra.cluster.KMedoids")
    except ImportError:
        # Fallback to custom PAM implementation
        print("sklearn_extra not available, using custom PAM implementation")
        train_index, labels = _kmedoids_pam(
            distance_matrix=distance_matrix,
            n_clusters=k_train,
            max_iter=100,
            random_state=seed,
        )
    
    train_index = np.sort(train_index)
    
    # Compute cluster sizes for APW scoring
    cluster_sizes = np.bincount(labels, minlength=k_train)
    
    # Step 5: Select eval indices from non-anchor points
    remaining_indices = list(set(range(num_examples)) - set(train_index))
    k_eval = int(eval_ratio * num_examples)
    
    if k_eval > 0 and len(remaining_indices) > 0:
        eval_index = np.sort(
            np.array(
                np.random.choice(
                    remaining_indices,
                    size=min(k_eval, len(remaining_indices)),
                    replace=False,
                )
            )
        )
    else:
        eval_index = np.array([], dtype=int)
    
    print(f"\n{'='*60}")
    print(f"Anchor Points Selection complete!")
    print(f"Number of anchor points: {len(train_index)}")
    print(f"Cluster size stats: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")
    print(f"Eval set size: {len(eval_index)}")
    print(f"{'='*60}\n")
    
    # Print some anchor points for sanity check
    print(f"Sample anchor points (first 5):")
    for i, idx in enumerate(train_index[:5]):
        conf_str = ", ".join([f"{c:.3f}" for c in confidences[idx][:5]])
        if confidences.shape[1] > 5:
            conf_str += ", ..."
        print(f"  {i+1}. Index {idx}: confidences=[{conf_str}]")
        print(f"     Question: {questions[idx][:100]}...")
    
    return train_index, eval_index, cluster_sizes


# ============================================================================
# ABLATION METHODS: Opposite/Worst-Case Subset Selection
# These methods intentionally select suboptimal subsets to demonstrate
# that the original methods make a meaningful difference.
# ============================================================================


def most_confident_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio, 
    scorer_llm_name, 
    seed=42,
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select train and eval indices based on scorer LLM confidence - OPPOSITE of least_confident.
  
  This is an ABLATION method that selects the EASIEST examples (highest confidence).
  The model is most confident about these examples, so they provide the least
  informative signal for optimization.
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    scorer_llm_name: Name of the LLM to use for scoring confidence (via vLLM)
    seed: Random seed for reproducibility (default: 42)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for saving confidence scores
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  cache_dir = os.path.join(opro_root, "confidence_scores", dataset_name)
  
  # Extract questions and answers from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  answers = _extract_short_answers_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  if len(answers) != num_examples:
    raise ValueError(
        f"Mismatch between number of questions ({num_examples}) "
        f"and answers ({len(answers)})"
    )
  
  print(f"\n{'='*60}")
  print(f"[ABLATION] Most Confident Subset Selection for {dataset_name}")
  print(f"This selects the EASIEST examples (opposite of least_confident)")
  print(f"{'='*60}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Scorer LLM: {scorer_llm_name} (via vLLM)")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"Cache directory: {cache_dir}")
  print(f"{'='*60}\n")
  
  # Step 1: Compute or load confidence scores using vLLM
  confidence_scores = _compute_confidence_scores_vllm(
      questions, 
      answers, 
      scorer_llm_name, 
      dataset_name, 
      cache_dir,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Step 2: Sort by confidence DESCENDING (most confident first) - OPPOSITE of least_confident
  sorted_indices = np.argsort(-confidence_scores)  # Negative for descending order
  
  # Step 3: Select training set from MOST confident examples (easiest)
  k_train = int(train_ratio * num_examples)
  train_index = np.sort(sorted_indices[:k_train])
  
  # Step 4: Select eval indices from remaining examples
  remaining_indices = sorted_indices[k_train:]
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  # Print statistics
  train_scores = confidence_scores[train_index]
  print(f"\n{'='*60}")
  print(f"[ABLATION] Selection complete!")
  print(f"Training set size: {len(train_index)}")
  print(f"Eval set size: {len(eval_index)}")
  print(f"Training set confidence scores (log likelihood) - HIGH = EASY:")
  print(f"  Mean: {train_scores.mean():.4f}")
  print(f"  Min: {train_scores.min():.4f}")
  print(f"  Max: {train_scores.max():.4f}")
  print(f"{'='*60}\n")
  
  # Sanity check: Print first 5 selected training examples
  print(f"\n{'='*60}")
  print("[ABLATION] SANITY CHECK: First 5 selected training examples (EASIEST)")
  print(f"{'='*60}")
  num_to_show = min(5, len(train_index))
  for i in range(num_to_show):
    idx = train_index[i]
    score = confidence_scores[idx]
    question = questions[idx]
    answer = answers[idx]
    
    print(f"\nExample {i+1} (Index {idx}):")
    print(f"  Question: {question}")
    print(f"  Answer: {answer}")
    print(f"  Confidence score (HIGH = easy): {score:.4f}")
  
  print(f"\n{'='*60}\n")
  
  return train_index, eval_index


def _anti_greedy_subset_selection(M, k, verbose=True):
  """Anti-greedy subset selection - selects LEAST representative examples.
  
  This is the OPPOSITE of greedy facility location. Instead of maximizing
  coverage, it selects points that MINIMIZE coverage (most redundant/outlier).
  
  Strategy: Select points with minimum marginal coverage contribution.
  
  Args:
    M: Similarity matrix of shape (N, N), values in [0,1]
    k: Subset size
    verbose: Print progress if True
    
  Returns:
    List of selected indices (least representative)
  """
  N = M.shape[0]
  S = []  # Selected subset
  remaining = set(range(N))  # Candidates not yet selected
  current_max = np.zeros(N)  # Current coverage for each point
  
  if verbose:
    print(f"Running anti-greedy selection to select {k} LEAST representative samples...")
  
  for iteration in tqdm(range(k), desc="Anti-greedy selection", disable=not verbose):
    best_gain = float('inf')  # We want MINIMUM gain
    best_i = None
    
    # Find the point that MINIMIZES marginal gain (least coverage contribution)
    for i in remaining:
      row_i = M[i]  # Similarity from i to all points
      gain_vector = np.maximum(current_max, row_i) - current_max
      delta = gain_vector.sum()  # Marginal gain
      
      if delta < best_gain:  # MINIMUM instead of maximum
        best_gain = delta
        best_i = i
    
    # Add point with minimum gain to subset
    i_star = best_i
    S.append(i_star)
    remaining.remove(i_star)
    
    # Update coverage (even though we're selecting poorly, we track coverage)
    current_max = np.maximum(current_max, M[i_star])
  
  if verbose:
    print("Anti-greedy selection complete.")
  
  return S


def least_representative_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio, 
    alpha=0.9, 
    seed=42,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select train and eval indices using LEAST representative subset selection.
  
  This is an ABLATION method - the OPPOSITE of representative selection.
  Instead of maximizing coverage/diversity, it selects the LEAST representative
  examples (outliers, redundant points that don't cover the dataset well).
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    alpha: Weight for dense similarity (1-alpha for lexical). Default 0.9.
    seed: Random seed for reproducibility (default: 42)
    embedding_model: Model for generating embeddings (default: Qwen3-Embedding-8B)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for saving embeddings and similarity matrices
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  cache_dir = os.path.join(opro_root, "embeddings_and_sim_matrices", dataset_name)
  embeddings_dir = cache_dir
  sim_matrices_dir = cache_dir
  
  # Extract questions from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  print(f"\n{'='*60}")
  print(f"[ABLATION] Least Representative Subset Selection for {dataset_name}")
  print(f"This selects the LEAST diverse examples (opposite of representative)")
  print(f"{'='*60}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Alpha (dense weight): {alpha}")
  print(f"Embedding model: {embedding_model}")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"Cache directory: {cache_dir}")
  print(f"{'='*60}\n")
  
  # Step 1: Generate or load embeddings using vLLM
  embeddings = _generate_embeddings_vllm(
      questions, 
      dataset_name, 
      embeddings_dir,
      embedding_model=embedding_model,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Step 2: Compute or load dense similarity matrix
  sim_dense = _compute_dense_similarity(embeddings, dataset_name, sim_matrices_dir, embedding_model)
  
  # Step 3: Compute or load TF-IDF similarity matrix
  sim_tfidf = _compute_tfidf_similarity(questions, dataset_name, sim_matrices_dir)
  
  # Step 4: Apply normalizations as per notebook
  print("Applying row-wise min-max normalization...")
  sim_dense_norm = _row_minmax_normalize(sim_dense)
  sim_tfidf_norm = _row_minmax_normalize(sim_tfidf)
  
  # Apply square root to TF-IDF to smooth out the distribution
  sim_tfidf_norm_sqrt = np.sqrt(sim_tfidf_norm)
  
  # Step 5: Mix the similarity matrices
  print(f"Mixing similarity matrices with alpha={alpha}...")
  M_mixed = alpha * sim_dense_norm + (1 - alpha) * sim_tfidf_norm_sqrt
  
  # Step 6: Run ANTI-greedy subset selection for training set (OPPOSITE of greedy)
  k_train = int(train_ratio * num_examples)
  train_index = _anti_greedy_subset_selection(M_mixed, k_train, verbose=True)
  train_index = np.sort(np.array(train_index))
  
  # Step 7: Select eval indices from remaining examples
  remaining_indices = list(set(range(num_examples)) - set(train_index))
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  print(f"\n{'='*60}")
  print(f"[ABLATION] Selection complete!")
  print(f"Training set size: {len(train_index)} (least representative)")
  print(f"Eval set size: {len(eval_index)}")
  print(f"{'='*60}\n")
  
  return train_index, eval_index


def _weighted_anti_greedy_subset_selection(M, weights, k, verbose=True):
  """Anti-greedy subset selection with INVERTED importance weights.
  
  This is the OPPOSITE of weighted greedy selection. It:
  1. Minimizes coverage (anti-greedy)
  2. Uses INVERTED weights: prioritizes covering EASY examples (less useful)
  
  Objective: min_S Σ_i w_i * max_{j∈S} sim(i,j)
  where w_i = (1 - confidence_weight) + confidence_weight * normalized_confidence_i
  (Higher weight for confident/easy examples)
  
  Args:
    M: Similarity matrix of shape (N, N), values in [0,1]
    weights: Importance weights (higher = more important to cover, but inverted)
    k: Subset size
    verbose: Print progress if True
    
  Returns:
    List of selected indices
  """
  N = M.shape[0]
  S = []  # Selected subset
  remaining = set(range(N))  # Candidates not yet selected
  current_max = np.zeros(N)  # Current coverage for each point
  
  if verbose:
    print(f"Running weighted anti-greedy selection to select {k} samples...")
  
  for iteration in tqdm(range(k), desc="Weighted anti-greedy selection", disable=not verbose):
    best_gain = float('inf')  # We want MINIMUM gain
    best_i = None
    
    # Find the point that MINIMIZES weighted marginal gain
    for i in remaining:
      row_i = M[i]  # Similarity from i to all points
      gain_vector = np.maximum(current_max, row_i) - current_max
      # Weight by importance (inverted: prioritize easy examples)
      delta = (weights * gain_vector).sum()
      
      if delta < best_gain:  # MINIMUM instead of maximum
        best_gain = delta
        best_i = i
    
    # Add point with minimum gain to subset
    i_star = best_i
    S.append(i_star)
    remaining.remove(i_star)
    
    # Update coverage
    current_max = np.maximum(current_max, M[i_star])
  
  if verbose:
    print("Weighted anti-greedy selection complete.")
  
  return S


def confidence_weighted_least_representative_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio,
    scorer_llm_name, 
    alpha=0.9, 
    confidence_weight=0.5, 
    seed=42,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select subset that is LEAST diverse AND focuses on EASY examples.
  
  This is an ABLATION method - the OPPOSITE of confidence_weighted_representative.
  
  Instead of:
  - Maximizing coverage (diversity) -> MINIMIZES coverage (redundant/outlier)
  - Prioritizing hard examples -> Prioritizes EASY examples (high confidence)
  
  The objective becomes:
    min_S Σ_i w_i * max_{j∈S} sim(i,j)
  
  where w_i = (1 - confidence_weight) + confidence_weight * normalized_confidence_i
  
  This means:
  - Examples the model is confident about get HIGHER weight (opposite)
  - We MINIMIZE coverage while prioritizing easy examples
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    scorer_llm_name: Name of the LLM to use for logit-based confidence scoring (via vLLM)
    alpha: Weight for dense similarity (1-alpha for lexical). Default 0.9.
    confidence_weight: Balance between diversity and difficulty (inverted). 
                       0 = pure anti-representative (least diversity)
                       1 = heavily weight easy examples
                       Default 0.5 for balanced mix.
    seed: Random seed for reproducibility (default: 42)
    embedding_model: Model for generating embeddings (default: Qwen3-Embedding-8B)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for caching
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  rep_cache_dir = os.path.join(opro_root, "embeddings_and_sim_matrices", dataset_name)
  conf_cache_dir = os.path.join(opro_root, "confidence_scores", dataset_name)
  
  # Extract questions and answers from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  answers = _extract_short_answers_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  print(f"\n{'='*60}")
  print(f"[ABLATION] Confidence-Weighted LEAST Representative Subset Selection")
  print(f"This selects EASY + LEAST DIVERSE examples (opposite of CWR)")
  print(f"Dataset: {dataset_name}")
  print(f"{'='*60}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Alpha (dense vs lexical): {alpha}")
  print(f"Confidence weight (inverted): {confidence_weight}")
  print(f"  (0 = pure anti-diversity, 1 = heavily weight easy examples)")
  print(f"Scorer LLM: {scorer_llm_name}")
  print(f"Embedding model: {embedding_model}")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"{'='*60}\n")
  
  # ===== Step 1: Build similarity matrix (from representative selection) =====
  print("=== Building similarity matrix ===")
  
  # Generate or load embeddings using vLLM
  embeddings = _generate_embeddings_vllm(
      questions, 
      dataset_name, 
      rep_cache_dir,
      embedding_model=embedding_model,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Compute or load dense similarity matrix
  sim_dense = _compute_dense_similarity(embeddings, dataset_name, rep_cache_dir, embedding_model)
  
  # Compute or load TF-IDF similarity matrix
  sim_tfidf = _compute_tfidf_similarity(questions, dataset_name, rep_cache_dir)
  
  # Apply normalizations
  print("Applying row-wise min-max normalization...")
  sim_dense_norm = _row_minmax_normalize(sim_dense)
  sim_tfidf_norm = _row_minmax_normalize(sim_tfidf)
  
  # Apply square root to TF-IDF to smooth out the distribution
  sim_tfidf_norm_sqrt = np.sqrt(sim_tfidf_norm)
  
  # Mix the similarity matrices
  print(f"Mixing similarity matrices with alpha={alpha}...")
  M_mixed = alpha * sim_dense_norm + (1 - alpha) * sim_tfidf_norm_sqrt
  
  # ===== Step 2: Get logit-based confidence scores =====
  print("\n=== Computing confidence scores via log probability ===")
  
  confidence_scores = _compute_confidence_scores_vllm(
      questions, 
      answers,
      scorer_llm_name, 
      dataset_name, 
      conf_cache_dir,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # ===== Step 3: Compute INVERTED importance weights =====
  print("\n=== Computing INVERTED importance weights (prioritize EASY examples) ===")
  
  # Normalize confidence to [0, 1]
  conf_min = confidence_scores.min()
  conf_max = confidence_scores.max()
  conf_normalized = (confidence_scores - conf_min) / (conf_max - conf_min + 1e-8)
  
  # Compute INVERTED importance weights: higher for HIGH-confidence (easy) examples
  # w_i = (1 - confidence_weight) + confidence_weight * normalized_confidence
  # When confidence_weight=0: all weights are 1.0 (pure anti-diversity)
  # When confidence_weight=1: weights are normalized_confidence (easy examples)
  # This is OPPOSITE of the original which uses (1 - normalized_confidence)
  importance_weights = (1 - confidence_weight) + confidence_weight * conf_normalized
  
  print(f"INVERTED importance weight stats (higher = easier examples):")
  print(f"  Min: {importance_weights.min():.4f}")
  print(f"  Max: {importance_weights.max():.4f}")
  print(f"  Mean: {importance_weights.mean():.4f}")
  
  # ===== Step 4: Run weighted ANTI-greedy selection =====
  print("\n=== Running weighted ANTI-greedy selection ===")
  
  k_train = int(train_ratio * num_examples)
  train_index = _weighted_anti_greedy_subset_selection(M_mixed, importance_weights, k_train, verbose=True)
  train_index = np.sort(np.array(train_index))
  
  # ===== Step 5: Select eval indices from remaining examples =====
  remaining_indices = list(set(range(num_examples)) - set(train_index))
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  # ===== Print statistics =====
  train_confidence = confidence_scores[train_index]
  print(f"\n{'='*60}")
  print(f"[ABLATION] Selection complete!")
  print(f"Training set size: {len(train_index)}")
  print(f"Eval set size: {len(eval_index)}")
  print(f"Training set confidence (log likelihood) - HIGH = EASY:")
  print(f"  Mean: {train_confidence.mean():.4f}")
  print(f"  Min: {train_confidence.min():.4f}")
  print(f"  Max: {train_confidence.max():.4f}")
  print(f"{'='*60}\n")
  
  # Sanity check: Print first 5 selected training examples
  print(f"\n{'='*60}")
  print("[ABLATION] SANITY CHECK: First 5 selected training examples (EASY + LEAST DIVERSE)")
  print(f"{'='*60}")
  num_to_show = min(5, len(train_index))
  for i in range(num_to_show):
    idx = train_index[i]
    conf = confidence_scores[idx]
    weight = importance_weights[idx]
    question = questions[idx]
    answer = answers[idx]
    
    print(f"\nExample {i+1} (Index {idx}):")
    print(f"  Question: {question[:200]}...")
    print(f"  Answer: {answer}")
    print(f"  Confidence (HIGH = easy): {conf:.4f}")
    print(f"  Inverted importance weight: {weight:.4f}")
  
  print(f"\n{'='*60}\n")
  
  return train_index, eval_index


# ============================================================================
# Verbal Confidence-Weighted Representative Subset Selection
# Uses verbal confidence instead of logit-based confidence
# ============================================================================

def verbal_confidence_weighted_representative_subset(
    dataset_name, 
    raw_data, 
    train_ratio, 
    eval_ratio,
    scorer_llm_name, 
    alpha=0.9, 
    confidence_weight=0.5, 
    k=4,
    seed=42,
    embedding_model="Qwen/Qwen3-Embedding-8B",
    tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.90
):
  """Select subset that is both diverse AND focuses on hard examples using VERBAL confidence.
  
  This method combines representative selection (diversity) with VERBAL confidence
  scoring (difficulty) using a weighted facility location objective.
  
  Unlike confidence_weighted_representative which uses logit-based confidence
  (P(correct answer | question)), this uses verbal confidence from the
  "Just Ask for Calibration" approach where the model explicitly states
  its confidence via k guesses with probabilities.
  
  The objective becomes:
    max_S Σ_i w_i * max_{j∈S} sim(i,j)
  
  where w_i = (1 - confidence_weight) + confidence_weight * (1 - normalized_verbal_confidence_i)
  
  This means:
  - Examples the model verbally reports high confidence on get lower weight in coverage
  - Examples the model verbally reports struggling with get higher weight
  - We maximize coverage while prioritizing hard examples
  
  Args:
    dataset_name: Name of the dataset (e.g., "gsm8k", "bbh", "mmlu")
    raw_data: Raw data of the dataset
    train_ratio: Ratio of examples to use for training (0 to 1)
    eval_ratio: Ratio of examples to use for evaluation (0 to 1)
    scorer_llm_name: Name of the LLM to use for verbal confidence scoring (via vLLM)
    alpha: Weight for dense similarity (1-alpha for lexical). Default 0.9.
    confidence_weight: Balance between diversity and difficulty. 
                       0 = pure representative (diversity only)
                       1 = heavily weight hard examples
                       >1 = even more aggressive weighting on hard examples
                       Default 0.5 for balanced mix.
    k: Number of guesses to request for verbal confidence (default: 4)
    seed: Random seed for reproducibility (default: 42)
    embedding_model: Model for generating embeddings (default: Qwen3-Embedding-8B)
    tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    gpu_memory_utilization: Fraction of GPU memory to use (default: 0.90)
    
  Returns:
    Tuple of (train_index, eval_index) where each is a sorted numpy array
    of indices
  """
  np.random.seed(seed)
  
  # Determine paths for caching
  opro_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
  rep_cache_dir = os.path.join(opro_root, "embeddings_and_sim_matrices", dataset_name)
  verbal_conf_cache_dir = os.path.join(opro_root, "verbal_confidence_scores", dataset_name)
  
  # Extract questions from raw_data
  questions = _extract_questions_from_raw_data(dataset_name, raw_data)
  num_examples = len(questions)
  
  print(f"\n{'='*60}")
  print(f"Verbal Confidence-Weighted Representative Subset Selection")
  print(f"Dataset: {dataset_name}")
  print(f"{'='*60}")
  print(f"Number of examples: {num_examples}")
  print(f"Train ratio: {train_ratio}, Eval ratio: {eval_ratio}")
  print(f"Alpha (dense vs lexical): {alpha}")
  print(f"Confidence weight: {confidence_weight}")
  print(f"  (0 = pure diversity, 1 = heavily weight hard examples)")
  print(f"Scorer LLM: {scorer_llm_name}")
  print(f"Embedding model: {embedding_model}")
  print(f"Verbal confidence k (num guesses): {k}")
  print(f"Tensor parallel size: {tensor_parallel_size}")
  print(f"{'='*60}\n")
  
  # ===== Step 1: Build similarity matrix (from representative selection) =====
  print("=== Building similarity matrix ===")
  
  # Generate or load embeddings using vLLM
  embeddings = _generate_embeddings_vllm(
      questions, 
      dataset_name, 
      rep_cache_dir,
      embedding_model=embedding_model,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # Compute or load dense similarity matrix
  sim_dense = _compute_dense_similarity(embeddings, dataset_name, rep_cache_dir, embedding_model)
  
  # Compute or load TF-IDF similarity matrix
  sim_tfidf = _compute_tfidf_similarity(questions, dataset_name, rep_cache_dir)
  
  # Apply normalizations
  print("Applying row-wise min-max normalization...")
  sim_dense_norm = _row_minmax_normalize(sim_dense)
  sim_tfidf_norm = _row_minmax_normalize(sim_tfidf)
  
  # Apply square root to TF-IDF to smooth out the distribution
  sim_tfidf_norm_sqrt = np.sqrt(sim_tfidf_norm)
  
  # Mix the similarity matrices
  print(f"Mixing similarity matrices with alpha={alpha}...")
  M_mixed = alpha * sim_dense_norm + (1 - alpha) * sim_tfidf_norm_sqrt
  
  # ===== Step 2: Get VERBAL confidence scores =====
  print("\n=== Computing VERBAL confidence scores ===")
  
  confidence_scores = _compute_verbal_confidence_scores_vllm(
      questions, 
      scorer_llm_name, 
      dataset_name, 
      verbal_conf_cache_dir,
      k=k,
      tensor_parallel_size=tensor_parallel_size,
      gpu_memory_utilization=gpu_memory_utilization
  )
  
  # ===== Step 3: Compute importance weights =====
  print("\n=== Computing importance weights ===")
  
  # Normalize confidence to [0, 1]
  conf_min = confidence_scores.min()
  conf_max = confidence_scores.max()
  conf_normalized = (confidence_scores - conf_min) / (conf_max - conf_min + 1e-8)
  
  # Compute importance weights: higher for low-confidence (hard) examples
  # w_i = (1 - confidence_weight) + confidence_weight * (1 - normalized_confidence)
  # When confidence_weight=0: all weights are 1.0 (pure diversity)
  # When confidence_weight=1: weights are (1 - normalized_confidence)
  importance_weights = (1 - confidence_weight) + confidence_weight * (1 - conf_normalized)
  
  print(f"Importance weight stats:")
  print(f"  Min: {importance_weights.min():.4f}")
  print(f"  Max: {importance_weights.max():.4f}")
  print(f"  Mean: {importance_weights.mean():.4f}")
  
  # ===== Step 4: Run weighted greedy selection =====
  print("\n=== Running weighted greedy selection ===")
  
  k_train = int(train_ratio * num_examples)
  train_index = _weighted_greedy_subset_selection(M_mixed, importance_weights, k_train, verbose=True)
  train_index = np.sort(np.array(train_index))
  
  # ===== Step 5: Select eval indices from remaining examples =====
  remaining_indices = list(set(range(num_examples)) - set(train_index))
  k_eval = int(eval_ratio * num_examples)
  
  if k_eval > 0 and len(remaining_indices) > 0:
    eval_index = np.sort(
        np.array(
            np.random.choice(
                remaining_indices,
                size=min(k_eval, len(remaining_indices)),
                replace=False,
            )
        )
    )
  else:
    eval_index = np.array([], dtype=int)
  
  # ===== Print statistics =====
  train_confidence = confidence_scores[train_index]
  print(f"\n{'='*60}")
  print(f"Selection complete!")
  print(f"Training set size: {len(train_index)}")
  print(f"Eval set size: {len(eval_index)}")
  print(f"Training set VERBAL confidence (avg of max probabilities):")
  print(f"  Mean: {train_confidence.mean():.4f}")
  print(f"  Min: {train_confidence.min():.4f}")
  print(f"  Max: {train_confidence.max():.4f}")
  print(f"{'='*60}\n")
  
  # Sanity check: Print first 5 selected training examples
  print(f"\n{'='*60}")
  print("SANITY CHECK: First 5 selected training examples")
  print(f"{'='*60}")
  num_to_show = min(5, len(train_index))
  for i in range(num_to_show):
    idx = train_index[i]
    conf = confidence_scores[idx]
    weight = importance_weights[idx]
    question = questions[idx]
    
    print(f"\nExample {i+1} (Index {idx}):")
    print(f"  Question: {question[:200]}...")
    print(f"  Verbal confidence: {conf:.4f}")
    print(f"  Importance weight: {weight:.4f}")
  
  print(f"\n{'='*60}\n")
  
  return train_index, eval_index
