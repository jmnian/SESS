"""vLLM-based local optimizer for generating candidate prompts in OPRO.

This module provides the optimizer component that uses a locally-hosted LLM
via vLLM to generate new candidate prompts based on previous instructions
and scores. This replaces the OpenAI API-based optimizer.

The optimizer runs on GPU 0, while scorer workers run on GPUs 1-7.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

# Note: vLLM import is deferred to avoid loading before GPU is set

# =============================================================================
# DEBUG FLAG - Set to False to disable verbose debug output
# =============================================================================
DEBUG_OPTIMIZER = False
# =============================================================================


def _debug_print(*args, **kwargs):
    """Print only if DEBUG_OPTIMIZER is True."""
    if DEBUG_OPTIMIZER:
        print(*args, **kwargs)


@dataclass
class VLLMOptimizerConfig:
    """Configuration for the vLLM-based optimizer."""
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 512
    temperature: float = 1.0
    num_candidates: int = 7  # Default to 7 since we have 7 scorer workers
    gpu_id: int = 0  # GPU for the optimizer
    gpu_memory_utilization: float = 0.95  # High utilization for large models
    max_model_len: int = 16384  # Reduced from 131K - OPRO meta prompts are short
    quantization: Optional[str] = None  # gpt-oss uses built-in MXFP4 (no extra quant needed)
    tensor_parallel_size: int = 1  # Single GPU for optimizer


class VLLMOptimizer:
    """vLLM-based optimizer for generating candidate prompts.
    
    Uses a locally-hosted LLM (e.g., GPT-OSS-120B) via vLLM to generate
    new candidate prompts based on the OPRO meta-prompt containing previous
    instructions and scores.
    
    This runs on a dedicated GPU (default: GPU 0) while the scorer workers
    use the remaining GPUs (1-7).
    """
    
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        max_tokens: int = 512,
        temperature: float = 1.0,
        num_candidates: int = 7,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 16384,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ):
        """Initialize the vLLM optimizer.
        
        Args:
            model: HuggingFace model path (default: openai/gpt-oss-120b)
            max_tokens: Max tokens for generation
            temperature: Sampling temperature
            num_candidates: Number of candidate prompts to generate per step
            gpu_id: GPU ID to use for the optimizer (default: 0)
            gpu_memory_utilization: Fraction of GPU memory for vLLM (0.95 for large models)
            max_model_len: Maximum context length. Reduced from model default (131K) 
                          to save KV cache memory. 16K is plenty for OPRO meta prompts.
            quantization: Quantization method (fp8, awq, gptq, None). 
                         For gpt-oss, use None as it has built-in MXFP4 quantization.
            tensor_parallel_size: Number of GPUs for tensor parallelism (1 for single GPU)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_candidates = num_candidates
        self.gpu_id = gpu_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.tensor_parallel_size = tensor_parallel_size
        
        self.llm = None
        self.sampling_params = None
        self._initialized = False
        
        # Logging for debugging - stores details of the last generation
        self.last_generation_log = []  # List of {raw_output, parsed_instruction, filtered_reason}
    
    def initialize(self):
        """Initialize the vLLM model.
        
        Must be called before generating candidates. This sets CUDA_VISIBLE_DEVICES
        and loads the model.
        """
        if self._initialized:
            return
        
        # Set CUDA_VISIBLE_DEVICES before importing vLLM
        # For tensor_parallel_size > 1, we'd need multiple GPUs
        if self.tensor_parallel_size == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            visible_gpus = [self.gpu_id]
        else:
            # Use gpu_id as the starting GPU for tensor parallelism
            visible_gpus = list(range(self.gpu_id, self.gpu_id + self.tensor_parallel_size))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_gpus))
        
        print(f"[VLLMOptimizer] Initializing on GPU(s): {visible_gpus}")
        print(f"[VLLMOptimizer] Loading model: {self.model}")
        
        # Now import vLLM after CUDA_VISIBLE_DEVICES is set
        from vllm import LLM, SamplingParams
        
        # Configure vLLM engine
        engine_kwargs = {
            "model": self.model,
            "dtype": "bfloat16",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "trust_remote_code": True,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        
        print(f"[VLLMOptimizer] GPU memory utilization: {self.gpu_memory_utilization}")
        print(f"[VLLMOptimizer] Max model length: {self.max_model_len}")
        
        # Add quantization if specified (gpt-oss has built-in MXFP4, so None is typical)
        if self.quantization:
            engine_kwargs["quantization"] = self.quantization
            print(f"[VLLMOptimizer] Using {self.quantization} quantization")
        else:
            print(f"[VLLMOptimizer] Using model's native quantization (MXFP4 for gpt-oss)")
        
        self.llm = LLM(**engine_kwargs)
        
        # Default sampling params (will be overridden per-call)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        self._initialized = True
        print(f"[VLLMOptimizer] Model loaded successfully")
    
    def generate_candidates(
        self,
        meta_prompt: str,
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """Generate candidate prompts using the local vLLM model.
        
        Uses batched inference to generate multiple candidates efficiently.
        
        Args:
            meta_prompt: The OPRO meta-prompt with instructions and scores
            num_candidates: Override default number of candidates
            temperature: Override default temperature
            
        Returns:
            List of generated candidate prompts (raw outputs)
        """
        if not self._initialized:
            self.initialize()
        
        from vllm import SamplingParams
        
        n = num_candidates or self.num_candidates
        temp = temperature if temperature is not None else self.temperature
        
        # Create sampling params with updated temperature
        # NOTE: Don't use stop sequences here - gpt-oss is a reasoning model that
        # "thinks out loud" before giving the answer. Stop sequences would trigger
        # when the model describes the format (e.g., "end with </INS>") before
        # it actually outputs the instruction. The parser handles extraction.
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=self.max_tokens,
        )
        
        # Generate all candidates in a single batched call
        # We create n copies of the same prompt to generate n different outputs
        prompts = [meta_prompt] * n
        
        print(f"[VLLMOptimizer] Generating {n} candidates...")
        start_time = time.time()
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        elapsed = time.time() - start_time
        print(f"[VLLMOptimizer] Generated {len(outputs)} candidates in {elapsed:.2f}s")
        
        # Extract text from outputs
        candidates = [output.outputs[0].text.strip() for output in outputs]
        
        return candidates
    
    def parse_instruction(
        self,
        raw_output: str,
        instruction_pos: str = "A_begin",
    ) -> str:
        """Parse the instruction from optimizer output.
        
        Extracts the instruction from XML-style tags <INS>...</INS> or
        <Start>...</Start> depending on instruction position.
        
        Args:
            raw_output: Raw output from the optimizer
            instruction_pos: Position type to determine parsing
            
        Returns:
            Extracted instruction string, or empty string if invalid
        """
        import re
        
        if instruction_pos == "A_begin":
            start_tag = "<Start>"
            end_tag = "</Start>"
        else:
            start_tag = "<INS>"
            end_tag = "</INS>"
        
        # Use regex to find ALL complete tag pairs, then pick the best one
        # Reasoning models often mention the tags in their thinking before the actual instruction
        # e.g., "should start with <INS> and end with </INS>" appears before "<INS>actual instruction</INS>"
        pattern = re.escape(start_tag) + r'\s*(.*?)\s*' + re.escape(end_tag)
        matches = re.findall(pattern, raw_output, re.DOTALL)
        
        if matches:
            # Filter out short/garbage matches (like "and end with" from reasoning)
            valid_matches = [m.strip() for m in matches if len(m.strip()) > 20]
            
            if valid_matches:
                # Take the LAST valid match (usually the actual instruction after reasoning)
                instruction = valid_matches[-1]
            else:
                # Fall back to the last match even if short
                instruction = matches[-1].strip()
        else:
            # Fallback: try to find content after start_tag until end_tag or end of reasonable content
            if start_tag in raw_output:
                start_idx = raw_output.index(start_tag) + len(start_tag)
                # Look for end_tag after start_tag
                remaining = raw_output[start_idx:]
                if end_tag in remaining:
                    end_idx = remaining.index(end_tag)
                    instruction = remaining[:end_idx].strip()
                else:
                    # No end tag - take first line/sentence as instruction
                    instruction = remaining.split('\n')[0].strip()
                    # Limit length if no end tag found
                    if len(instruction) > 300:
                        instruction = ""
            else:
                instruction = ""
        
        # Validate the instruction is reasonable (not garbage)
        if instruction:
            instruction_lower = instruction.lower()
            
            # Check for garbage indicators
            garbage_indicators = [
                len(instruction) > 500,  # Too long
                instruction.count('```') > 0,  # Contains code blocks
                instruction.count('{') > 3,  # Too many braces (likely code/JSON)
                instruction.count('|') > 3,  # Likely table/garbage
                any(ord(c) > 0x4000 for c in instruction[:100]),  # Non-Latin heavy
                instruction.count('\n') > 5,  # Too many newlines
                'def ' in instruction or 'class ' in instruction,  # Python code
                '```' in instruction,
                # Placeholder/template patterns
                '[instruction' in instruction_lower,
                '[insert' in instruction_lower,
                '[your' in instruction_lower,
                '{instruction' in instruction_lower,
                # Meta-commentary leaked from optimizer reasoning
                'listed earlier' in instruction_lower,
                'higher score' in instruction_lower,
                'previous ones' in instruction_lower,
                'better instruction' in instruction_lower,
                'scoring is not' in instruction_lower,
                'we have to generate' in instruction_lower,
                'should start with' in instruction_lower,
                'should end with' in instruction_lower,
                'must contain' in instruction_lower and 'tag' in instruction_lower,
                # Parsing fragments (starts/ends with quotes or fragments)
                instruction.startswith('"') and instruction.endswith('"') and len(instruction) < 30,
                instruction.startswith("'") and instruction.endswith("'") and len(instruction) < 30,
                'and end with' in instruction_lower and len(instruction) < 40,
                # Nonsense fragments
                instruction_lower in ['tags and closing', 'tags', 'closing tags', 'opening tags'],
                # Just punctuation or very short garbage
                len(instruction.strip('."\'!? ')) < 10,
            ]
            if any(garbage_indicators):
                _debug_print(f"[VLLMOptimizer] Filtering garbage instruction: '{instruction[:80]}...'")
                instruction = ""
        
        return instruction
    
    def generate_and_parse_candidates(
        self,
        meta_prompt: str,
        instruction_pos: str = "A_begin",
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> list[str]:
        """Generate and parse candidate prompts, retrying until enough valid ones.
        
        Keeps generating until we have `num_candidates` valid instructions,
        or we've tried `max_retries * num_candidates` total generations.
        
        Args:
            meta_prompt: The OPRO meta-prompt
            instruction_pos: Position type for parsing
            num_candidates: Number of valid candidates to aim for
            temperature: Sampling temperature
            max_retries: Maximum retry rounds (total attempts = max_retries * num_candidates)
            
        Returns:
            List of parsed instruction strings (may be fewer than num_candidates if limit hit)
        """
        n = num_candidates or self.num_candidates
        max_total_attempts = max_retries * n
        
        all_instructions = []
        seen_instructions = set()  # Deduplicate
        total_generated = 0
        retry_round = 0
        
        _debug_print(f"\n{'='*60}")
        _debug_print(f"[VLLMOptimizer] Target: {n} valid candidates, max {max_total_attempts} attempts")
        _debug_print(f"[VLLMOptimizer DEBUG] instruction_pos: {instruction_pos}")
        if instruction_pos == "A_begin":
            _debug_print(f"[VLLMOptimizer DEBUG] Looking for tags: <Start>...</Start>")
        else:
            _debug_print(f"[VLLMOptimizer DEBUG] Looking for tags: <INS>...</INS>")
        _debug_print(f"{'='*60}")
        
        # Clear previous generation log
        self.last_generation_log = []
        
        while len(all_instructions) < n and total_generated < max_total_attempts:
            retry_round += 1
            needed = n - len(all_instructions)
            batch_size = min(needed + 2, n)  # Generate a few extra to account for failures
            
            _debug_print(f"\n[VLLMOptimizer] Round {retry_round}: Generating {batch_size} candidates (have {len(all_instructions)}/{n})...")
            
            raw_outputs = self.generate_candidates(
                meta_prompt=meta_prompt,
                num_candidates=batch_size,
                temperature=temperature,
            )
            total_generated += len(raw_outputs)
            
            for i, raw in enumerate(raw_outputs):
                if len(all_instructions) >= n:
                    break
                    
                _debug_print(f"\n{'─'*60}")
                _debug_print(f"[VLLMOptimizer DEBUG] Candidate {total_generated - len(raw_outputs) + i + 1} RAW OUTPUT:")
                _debug_print(f"{'─'*60}")
                _debug_print(raw[:2000] if len(raw) > 2000 else raw)
                if len(raw) > 2000:
                    _debug_print(f"... [truncated, total length: {len(raw)}]")
                _debug_print(f"{'─'*60}")
                
                log_entry = {
                    "raw_output": raw[:4000] if len(raw) > 4000 else raw,  # Truncate for storage
                    "raw_output_length": len(raw),
                    "parsed_instruction": None,
                    "status": None,
                    "filter_reason": None,
                }
                
                try:
                    instruction = self.parse_instruction(raw, instruction_pos)
                    log_entry["parsed_instruction"] = instruction
                    _debug_print(f"[VLLMOptimizer DEBUG] PARSED INSTRUCTION:")
                    _debug_print(f"  '{instruction}'")
                    
                    if not instruction:
                        log_entry["status"] = "filtered"
                        log_entry["filter_reason"] = "empty_or_garbage"
                        _debug_print(f"[VLLMOptimizer DEBUG] ✗ Empty instruction, skipped")
                    elif instruction in seen_instructions:
                        log_entry["status"] = "filtered"
                        log_entry["filter_reason"] = "duplicate"
                        _debug_print(f"[VLLMOptimizer DEBUG] ✗ Duplicate instruction, skipped")
                    elif len(instruction) < 15:
                        log_entry["status"] = "filtered"
                        log_entry["filter_reason"] = f"too_short_{len(instruction)}_chars"
                        _debug_print(f"[VLLMOptimizer DEBUG] ✗ Too short ({len(instruction)} chars), skipped")
                    else:
                        log_entry["status"] = "accepted"
                        all_instructions.append(instruction)
                        seen_instructions.add(instruction)
                        _debug_print(f"[VLLMOptimizer DEBUG] ✓ Added ({len(all_instructions)}/{n})")
                except Exception as e:
                    log_entry["status"] = "error"
                    log_entry["filter_reason"] = str(e)
                    _debug_print(f"[VLLMOptimizer DEBUG] ✗ Failed to parse: {e}")
                
                self.last_generation_log.append(log_entry)
        
        _debug_print(f"\n{'='*60}")
        _debug_print(f"[VLLMOptimizer] Final: {len(all_instructions)} valid candidates from {total_generated} total generations")
        _debug_print(f"{'='*60}\n")
        
        return all_instructions
    
    def shutdown(self):
        """Cleanup the vLLM model and free GPU memory."""
        if self.llm is not None:
            # vLLM doesn't have an explicit shutdown, but we can delete the reference
            del self.llm
            self.llm = None
            self._initialized = False
            
            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            print("[VLLMOptimizer] Model unloaded")


def create_vllm_optimizer(
    model: str = "openai/gpt-oss-120b",
    **kwargs,
) -> VLLMOptimizer:
    """Factory function to create a vLLM optimizer.
    
    Args:
        model: Model to use
        **kwargs: Additional arguments for VLLMOptimizer
        
    Returns:
        Configured VLLMOptimizer instance
    """
    return VLLMOptimizer(
        model=model,
        **kwargs,
    )


# For backward compatibility with code that imports from openai_optimizer
LocalOptimizer = VLLMOptimizer

