"""vLLM-based scorer worker process for multi-GPU OPRO.

Each worker:
- Pins to exactly one GPU via CUDA_VISIBLE_DEVICES
- Loads vLLM scorer model once at startup
- Repeatedly receives (candidate_prompt, questions) and returns answers
- Uses batched inference for all questions per candidate
"""

import os
import signal
import traceback
import multiprocessing

# Use 'spawn' to create worker processes to avoid CUDA reinitialization issues.
# When using 'fork' (the default on Linux), child processes inherit the parent's
# CUDA state, which causes "Cannot re-initialize CUDA in forked subprocess" errors
# if CUDA was initialized in the parent (e.g., by subset selection using vLLM).
mp_context = multiprocessing.get_context('spawn')
Process = mp_context.Process
Queue = mp_context.Queue
from typing import Optional


class ScorerWorker:
    """vLLM-based scorer worker that runs on a single GPU.
    
    The worker loads the model once and processes evaluation requests
    by batching all questions for a candidate prompt into a single
    vLLM chat() call for proper instruction-following.
    """
    
    def __init__(
        self,
        gpu_id: int,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.90,
        use_chat_mode: bool = True,  # Chat mode: natural EOS stopping, no system prompt
    ):
        """Initialize the scorer worker.
        
        Args:
            gpu_id: The GPU ID to pin this worker to (0-7)
            model_name: HuggingFace model name
            max_tokens: Maximum tokens to generate per response
            gpu_memory_utilization: Fraction of GPU memory for vLLM
            use_chat_mode: If True, use vLLM's chat() API with proper chat
                          template. This is recommended for instruction-tuned
                          models like Qwen2.5-Instruct to prevent hallucinations
                          and system prompt artifacts in outputs.
        """
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_chat_mode = use_chat_mode
        self.llm = None
        self.sampling_params = None
    
    def initialize(self):
        """Initialize vLLM model. Must be called after GPU pinning."""
        # Import vLLM after CUDA_VISIBLE_DEVICES is set
        from vllm import LLM, SamplingParams
        
        mode_str = "chat" if self.use_chat_mode else "completion"
        print(f"[Worker {self.gpu_id}] Loading vLLM model: {self.model_name} (mode: {mode_str})")
        
        self.llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=False,  # Disable to prevent cache corruption issues
            trust_remote_code=True,
            seed=42,  # Global seed for determinism
            enforce_eager=True,  # Disable CUDA graphs for more deterministic behavior
        )
        
        # Stop sequences - in chat mode, the model is better behaved but we still
        # want to stop on certain patterns to prevent runaway generation
        stop_sequences = [
            "\n\nQ:",       # New question (Q&A format)
            "\nQuestion:",  # New question (explicit)
            "\n\n---",      # Section break
            "\n\nNote:",    # Side notes
            "\n\nHint:",    # Hints (shouldn't appear)
        ]
        
        if not self.use_chat_mode:
            # In completion mode, add more aggressive stops to prevent artifacts
            stop_sequences.extend([
                "\nQ:",         # New question on new line
                ". Q:",         # New question after sentence
                ".\nQ:",        # New question after sentence + newline
                ") Q:",         # New question after closing paren
                "A: Let",       # Start of a new answer
                "A: The",       # Start of a new answer
                "A: To",        # Start of a new answer
                "\n\nA:",       # New answer after blank line
                "You are an AI",  # System prompt artifact
                "\nYou are",      # System prompt artifact
            ])
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            # Seed for deterministic sampling (even with temperature=0, vLLM can have
            # non-determinism due to CUDA kernel operations and batching)
            seed=42,
            stop=stop_sequences,
        )
        
        print(f"[Worker {self.gpu_id}] Model loaded successfully")
    
    def evaluate_candidate(
        self,
        candidate_prompt: str,
        questions: list[str],
        instruction_pos: str = "A_begin",
        include_qa: bool = True,
    ) -> list[str]:
        """Evaluate a candidate prompt on all questions with batched inference.
        
        Args:
            candidate_prompt: The instruction/prompt to evaluate
            questions: List of questions to evaluate
            instruction_pos: Position of instruction in prompt
            include_qa: Whether to include Q:/A: format
            
        Returns:
            List of generated answers for each question
        """
        if self.llm is None:
            raise RuntimeError("Worker not initialized. Call initialize() first.")
        
        if self.use_chat_mode:
            # Use chat() API with proper chat template
            # This prevents model from outputting system prompt artifacts
            messages_batch = []
            for q in questions:
                messages = self._format_chat_messages(
                    candidate_prompt, q, instruction_pos, include_qa
                )
                messages_batch.append(messages)
            
            # Batched chat call
            outputs = self.llm.chat(messages_batch, self.sampling_params)
            
            # Extract and return answers
            answers = [o.outputs[0].text.strip() for o in outputs]
        else:
            # Fall back to completion mode (raw text prompts)
            prompts = []
            for q in questions:
                prompt = self._format_prompt(
                    candidate_prompt, q, instruction_pos, include_qa
                )
                prompts.append(prompt)
            
            outputs = self.llm.generate(prompts, self.sampling_params)
            answers = [o.outputs[0].text.strip() for o in outputs]
        
        return answers
    
    def _format_chat_messages(
        self,
        instruction: str,
        question: str,
        instruction_pos: str,
        include_qa: bool,
    ) -> list[dict]:
        """Format a prompt as OpenAI-style chat messages for vLLM.chat().
        
        Uses chat template to properly format for instruction-tuned models,
        preventing system prompt artifacts in outputs.
        
        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        # Build the user message content based on instruction position
        if include_qa:
            if instruction_pos == "before_Q":
                if instruction:
                    content = f"{instruction}\n\nQuestion: {question}"
                else:
                    content = f"Question: {question}"
            elif instruction_pos == "Q_begin":
                if instruction:
                    content = f"{instruction}\n{question}"
                else:
                    content = question
            elif instruction_pos == "Q_end":
                if instruction:
                    content = f"{question}\n{instruction}"
                else:
                    content = question
            else:  # instruction_pos == "A_begin"
                # For A_begin, instruction is prepended to the expected answer
                # We put instruction in system and question in user
                if instruction:
                    content = f"{question}\n\nPlease respond starting with: {instruction}"
                else:
                    content = question
        else:
            if instruction_pos == "Q_begin":
                if instruction:
                    content = f"{instruction}\n{question}"
                else:
                    content = question
            else:  # instruction_pos == "Q_end"
                if instruction:
                    content = f"{question}\n{instruction}"
                else:
                    content = question
        
        # Use chat mode with NO custom system prompt.
        # This lets the model use its default behavior, and OPRO can discover
        # through optimization which instructions lead to better reasoning.
        # If "output only the answer" instructions perform poorly (wrong answers),
        # OPRO will naturally evolve away from them.
        #
        # Note: We don't add a system message at all - vLLM will use the model's
        # default chat template which may include a default system prompt.
        messages = [
            {"role": "user", "content": content}
        ]
        
        return messages
    
    def _format_prompt(
        self,
        instruction: str,
        question: str,
        instruction_pos: str,
        include_qa: bool,
    ) -> str:
        """Format a single prompt with instruction and question (completion mode).
        
        Important: The instruction (candidate_prompt) prefix must be
        byte-for-byte identical across all prompts to enable vLLM
        prefix caching.
        """
        if include_qa:
            if instruction_pos == "before_Q":
                if instruction:
                    prompt = f"{instruction}\nQ: {question}\n\nA:"
                else:
                    prompt = f"Q: {question}\n\nA:"
            elif instruction_pos == "Q_begin":
                if instruction:
                    prompt = f"Q: {instruction}\n{question}\n\nA:"
                else:
                    prompt = f"Q: {question}\n\nA:"
            elif instruction_pos == "Q_end":
                if instruction:
                    prompt = f"Q: {question}\n{instruction}\n\nA:"
                else:
                    prompt = f"Q: {question}\n\nA:"
            else:  # instruction_pos == "A_begin"
                if instruction:
                    prompt = f"Q: {question}\n\nA: {instruction}"
                else:
                    prompt = f"Q: {question}\n\nA:"
        else:
            if instruction_pos == "Q_begin":
                if instruction:
                    prompt = f"{instruction}\n{question}"
                else:
                    prompt = question
            else:  # instruction_pos == "Q_end"
                if instruction:
                    prompt = f"{question}\n{instruction}"
                else:
                    prompt = question
        
        return prompt


def worker_process(
    gpu_id: int,
    input_queue: Queue,
    output_queue: Queue,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.90,
    use_chat_mode: bool = True,  # Chat mode: natural EOS stopping, no system prompt
):
    """Worker process main function.
    
    This function runs in a separate process and:
    1. Sets CUDA_VISIBLE_DEVICES before importing vLLM/torch
    2. Loads the model once
    3. Processes requests from input_queue
    4. Sends results to output_queue
    
    Args:
        gpu_id: GPU to pin to (0-7)
        input_queue: Queue to receive work from controller
        output_queue: Queue to send results to controller
        model_name: Model to load
        max_tokens: Max generation tokens
        gpu_memory_utilization: vLLM memory fraction
        use_chat_mode: If True, use vLLM's chat() API with proper template
    """
    # Reset signal handlers to default - workers should not inherit parent's handlers
    # This prevents the controller's signal handler from running in worker processes
    # which would cause "can only test a child process" errors
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    
    # CRITICAL: Set environment variables BEFORE importing vLLM or torch
    # 1. Set CUDA_VISIBLE_DEVICES to pin to specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Set unique MASTER_PORT for each worker to avoid port conflicts
    # vLLM/torch.distributed uses this port for distributed initialization
    # Using different ports ensures multiple workers can start simultaneously
    base_port = 29500  # PyTorch default is 29500
    os.environ["MASTER_PORT"] = str(base_port + gpu_id * 100)
    
    print(f"[Worker {gpu_id}] Starting on GPU {gpu_id} with MASTER_PORT={os.environ['MASTER_PORT']}")
    
    try:
        # Initialize worker
        worker = ScorerWorker(
            gpu_id=gpu_id,
            model_name=model_name,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            use_chat_mode=use_chat_mode,
        )
        worker.initialize()
        
        # Signal ready
        output_queue.put({
            "type": "ready",
            "worker_id": gpu_id,
        })
        
        # Main processing loop
        while True:
            try:
                message = input_queue.get()
                
                if message is None or message.get("type") == "shutdown":
                    print(f"[Worker {gpu_id}] Received shutdown signal")
                    break
                
                if message.get("type") == "evaluate":
                    candidate_id = message["candidate_id"]
                    candidate_prompt = message["candidate_prompt"]
                    questions = message["questions"]
                    instruction_pos = message.get("instruction_pos", "A_begin")
                    include_qa = message.get("include_qa", True)
                    
                    # Evaluate the candidate
                    answers = worker.evaluate_candidate(
                        candidate_prompt=candidate_prompt,
                        questions=questions,
                        instruction_pos=instruction_pos,
                        include_qa=include_qa,
                    )
                    
                    # Send result back
                    output_queue.put({
                        "type": "result",
                        "worker_id": gpu_id,
                        "candidate_id": candidate_id,
                        "candidate_prompt": candidate_prompt,
                        "answers": answers,
                    })
                    
            except Exception as e:
                print(f"[Worker {gpu_id}] Error processing request: {e}")
                traceback.print_exc()
                output_queue.put({
                    "type": "error",
                    "worker_id": gpu_id,
                    "error": str(e),
                })
                
    except Exception as e:
        print(f"[Worker {gpu_id}] Fatal error: {e}")
        traceback.print_exc()
        output_queue.put({
            "type": "error",
            "worker_id": gpu_id,
            "error": str(e),
            "fatal": True,
        })
    
    print(f"[Worker {gpu_id}] Exiting")


def start_worker(
    gpu_id: int,
    input_queue: Queue,
    output_queue: Queue,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.90,
    use_chat_mode: bool = True,  # Chat mode: natural EOS stopping, no system prompt
) -> Process:
    """Start a worker process.
    
    Args:
        gpu_id: GPU to assign to worker
        input_queue: Queue for sending work to worker
        output_queue: Queue for receiving results from worker
        model_name: Model to load
        max_tokens: Max generation tokens
        gpu_memory_utilization: vLLM memory fraction
        use_chat_mode: If True, use vLLM's chat() API with proper template
        
    Returns:
        The started Process object
    """
    p = Process(
        target=worker_process,
        args=(
            gpu_id,
            input_queue,
            output_queue,
            model_name,
            max_tokens,
            gpu_memory_utilization,
            use_chat_mode,
        ),
        daemon=False,  # Non-daemon so vLLM can spawn child processes if needed
    )
    p.start()
    return p

