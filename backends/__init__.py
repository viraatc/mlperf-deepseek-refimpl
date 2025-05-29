"""
Modular backend system for MLPerf DeepSeek reference implementation.

Supports TensorRT-LLM, SGLang, vLLM, and PyTorch backends with shared API arguments
but independent execution implementations.
"""

from .base_backend import BaseBackend

__all__ = ['BaseBackend'] 