"""LLM optimizer for generating candidate prompts in OPRO.

This module provides the optimizer component that uses an LLM (via OpenAI API)
to generate new candidate prompts based on previous instructions and scores.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)


class LLMOptimizer:
    """LLM-based optimizer for generating candidate prompts.
    
    Uses OpenAI API (or compatible endpoints) to generate new candidate prompts
    based on the OPRO meta-prompt containing previous instructions and scores.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 512,
        temperature: float = 1.0,
        num_candidates: int = 8,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the LLM optimizer.
        
        Args:
            model: Model to use (default: gpt-4o)
            max_tokens: Max tokens for generation
            temperature: Sampling temperature
            num_candidates: Number of candidate prompts to generate per step
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional base URL for OpenAI-compatible APIs
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_candidates = num_candidates
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
    
    def generate_candidates(
        self,
        meta_prompt: str,
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> list[str]:
        """Generate candidate prompts using the OpenAI API.
        
        Makes parallel API calls to generate multiple candidates efficiently.
        
        Args:
            meta_prompt: The OPRO meta-prompt with instructions and scores
            num_candidates: Override default number of candidates
            temperature: Override default temperature
            max_retries: Max retry attempts on failure
            retry_delay: Seconds to wait between retries
            
        Returns:
            List of generated candidate prompts
        """
        n = num_candidates or self.num_candidates
        temp = temperature if temperature is not None else self.temperature
        
        candidates = []
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(self._generate_single, meta_prompt, temp, max_retries, retry_delay)
                for _ in range(n)
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        candidates.append(result)
                except Exception as e:
                    print(f"[Optimizer] Single generation failed: {e}")
        
        return candidates
    
    def _generate_single(
        self,
        meta_prompt: str,
        temperature: float,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> Optional[str]:
        """Generate a single candidate prompt."""
        messages = [{"role": "user", "content": meta_prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle rate limits with exponential backoff
                if "rate" in error_str or "429" in error_str:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[Optimizer] Rate limit hit. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                # Handle content filter errors
                if "content" in error_str and ("filter" in error_str or "policy" in error_str):
                    print(f"[Optimizer] Content filter triggered: {e}")
                    return None
                
                # Other errors - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[Optimizer] Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Optimizer] Maximum retries reached. Last error: {e}")
                    return None
        
        return None
    
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
            Extracted instruction string
        """
        if instruction_pos == "A_begin":
            start_tag = "<Start>"
            end_tag = "</Start>"
        else:
            start_tag = "<INS>"
            end_tag = "</INS>"
        
        if start_tag not in raw_output:
            start_idx = 0
        else:
            start_idx = raw_output.index(start_tag) + len(start_tag)
        
        if end_tag not in raw_output:
            end_idx = len(raw_output)
        else:
            end_idx = raw_output.index(end_tag)
        
        instruction = raw_output[start_idx:end_idx].strip()
        return instruction
    
    def generate_and_parse_candidates(
        self,
        meta_prompt: str,
        instruction_pos: str = "A_begin",
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """Generate and parse candidate prompts in one call.
        
        Combines generation and parsing for convenience.
        
        Args:
            meta_prompt: The OPRO meta-prompt
            instruction_pos: Position type for parsing
            num_candidates: Number of candidates to generate
            temperature: Sampling temperature
            
        Returns:
            List of parsed instruction strings
        """
        raw_outputs = self.generate_candidates(
            meta_prompt=meta_prompt,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        
        instructions = []
        for raw in raw_outputs:
            try:
                instruction = self.parse_instruction(raw, instruction_pos)
                if instruction:
                    instructions.append(instruction)
            except Exception as e:
                print(f"[Optimizer] Failed to parse instruction: {e}")
        
        return instructions


# Alias for backward compatibility
OpenAIOptimizer = LLMOptimizer


def create_optimizer(
    model: str = "gpt-4o",
    **kwargs,
) -> LLMOptimizer:
    """Factory function to create an LLM optimizer.
    
    Args:
        model: Model to use
        **kwargs: Additional arguments for LLMOptimizer
        
    Returns:
        Configured LLMOptimizer instance
    """
    return LLMOptimizer(
        model=model,
        **kwargs,
    )
