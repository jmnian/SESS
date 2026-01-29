"""Parallel processing modules for multi-GPU OPRO."""

from opro.parallel.worker import ScorerWorker, start_worker
from opro.parallel.controller import OPROController
from opro.parallel.openai_optimizer import LLMOptimizer, OpenAIOptimizer, create_optimizer

__all__ = [
    "ScorerWorker",
    "start_worker",
    "OPROController",
    "LLMOptimizer",
    "OpenAIOptimizer",  # Alias for backward compatibility
    "create_optimizer",
]

