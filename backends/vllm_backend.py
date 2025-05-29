import asyncio
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed
from vllm import LLM, SamplingParams

from .base_backend import BaseBackend
from .utils import get_cache_directory, setup_huggingface_cache


# Hardcoded vLLM configuration (adapted from TensorRT-LLM config)
VLLM_CONFIG = {
    "model": "deepseek-ai/DeepSeek-R1",
    "tokenizer": "deepseek-ai/DeepSeek-R1",
    "tensor_parallel_size": 8,
    "max_num_seqs": 256,
    "gpu_memory_utilization": 0.95,
    "trust_remote_code": True,
    "dtype": "auto",
    "max_input_len": 3136 + 4,
    "max_output_len": 32 * 1024,
    "max_model_len": 32 * 1024 + 3136 + 4,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "enforce_eager": False,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}


class VLLMBackend(BaseBackend):
    """vLLM backend with optimized batch async generation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize vLLM backend with hardcoded configuration."""
        super().__init__()  # Initialize base class
        # Use only hardcoded config, ignore any passed config
        self.config = VLLM_CONFIG.copy()

        # Set model and tokenizer names
        self.model_name = self.config['model']
        self.tokenizer_name = self.config['tokenizer']
        self.backend_name = 'vllm'

        self.llm = None
        self.is_initialized = False
        self.sampling_params = None
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables and cache directories."""
        # Use the utility function to get cache directory
        cache_base = get_cache_directory()

        # Use models subdirectory to match user's example paths
        self.cache_dir = cache_base.parent / 'models'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config['model_cache_dir'] = str(self.cache_dir)

        # Set up HuggingFace cache environment variables
        setup_huggingface_cache()

        # Set PyTorch threading to match tensor_parallel_size
        os.environ['OMP_NUM_THREADS'] = str(self.config['tensor_parallel_size'])

        # Set vLLM specific environment variables
        # Use FlashInfer for better performance
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

        # Disable v1 engine to avoid socket communication issues
        os.environ['VLLM_USE_V1'] = '1'

        # Disable progress bars to avoid clutter in MLPerf runs
        os.environ['VLLM_DISABLE_TQDM'] = '1'
        os.environ['DISABLE_TQDM'] = '1'

        # Optimize CUDA settings
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        os.environ['NCCL_TREE_THRESHOLD'] = '0'

        # Additional vLLM stability settings
        # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        # os.environ['VLLM_TRACE_FUNCTION'] = '0'
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        # Disable CUDA graphs explicitly
        # os.environ['VLLM_DISABLE_CUDA_GRAPH'] = '1'

        # Set seeds for reproducibility
        seed = self.config['seed']
        self._set_all_seeds(seed)

    def _set_all_seeds(self, seed: int = 42) -> None:
        """Set seeds for all random number generators for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        set_seed(seed)

    def _ensure_model_cached(self) -> Path:
        """Ensure model is available locally and return path."""
        model_name = self.config['model']

        # Create safe directory name from model path
        model_dir_name = model_name.replace("/", "_")
        checkpoint_path = self.cache_dir / model_dir_name

        if not checkpoint_path.exists():
            print(f"Model not found at {checkpoint_path}")
            print(f"Downloading {model_name} from HuggingFace...")

            # Create download command following user's exact steps
            cmd = [
                "huggingface-cli", "download",
                model_name,
                "--local-dir", str(checkpoint_path)
            ]

            # Set environment variable for faster downloads
            env = os.environ.copy()
            env['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

            try:
                # Run download command
                print(
                    f"Running command: HF_HUB_ENABLE_HF_TRANSFER=1 {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"Model downloaded successfully to {checkpoint_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading model: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                raise RuntimeError(f"Failed to download model: {e}")
        else:
            print(f"Using cached model at {checkpoint_path}")

        return checkpoint_path

    def initialize(self) -> None:
        """Initialize the vLLM backend."""
        # Ensure model is cached locally
        checkpoint_path = self._ensure_model_cached()

        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            n=1,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            max_tokens=self.config['max_output_len'],
            seed=self.config['seed'],
        )

        # Create kwargs dict for LLM initialization
        llm_kwargs = {
            'model': self.model_name,
            'tokenizer': self.tokenizer_name,
            'tensor_parallel_size': self.config['tensor_parallel_size'],
            'max_model_len': self.config['max_model_len'],
            'max_num_seqs': self.config['max_num_seqs'],
            'gpu_memory_utilization': self.config['gpu_memory_utilization'],
            'trust_remote_code': self.config['trust_remote_code'],
            'dtype': self.config['dtype'],
            'seed': self.config['seed'],
            'enforce_eager': self.config['enforce_eager'],
            'enable_prefix_caching': self.config['enable_prefix_caching'],
        }
        print(f"Initializing vLLM with config: {llm_kwargs}")

        # Initialize LLM
        try:
            self.llm = LLM(**llm_kwargs)
            self.is_initialized = True
            print("vLLM backend initialized successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM: {e}")

    def generate(self, tokenized_prompts: List[List[int]], prompt_strings: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses synchronously."""
        if not self.is_initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        if prompt_strings is None:
            raise ValueError("vLLM backend requires prompt_strings parameter")

        # Use provided prompt strings directly
        string_prompts = prompt_strings

        # Generate responses
        outputs = self.llm.generate(string_prompts, self.sampling_params)

        # Process results
        results = []
        for output in outputs:
            # Get generated tokens only
            token_ids = output.outputs[0].token_ids

            # Return only tokens
            result = {
                'tokens': token_ids,
            }
            results.append(result)

        return results

    def generate_async(self, tokenized_prompts: List[List[int]], prompt_strings: Optional[List[str]] = None, **kwargs) -> List[asyncio.Future]:
        """Generate responses asynchronously, returning futures immediately."""
        if not self.is_initialized or self.llm is None:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        if prompt_strings is None:
            raise ValueError("vLLM backend requires prompt_strings parameter")

        # Use provided prompt strings directly
        string_prompts = prompt_strings

        # Create futures for all prompts
        loop = asyncio.get_event_loop()
        futures = []

        # Process in batches for better throughput
        batch_size = min(self.config['max_num_seqs'], len(string_prompts))

        for i in range(0, len(string_prompts), batch_size):
            batch_prompts = string_prompts[i:i + batch_size]

            # Create future for this batch
            batch_future = loop.run_in_executor(
                None,
                self.llm.generate,
                batch_prompts,
                self.sampling_params
            )

            # For each prompt in the batch, create a future that extracts its result
            for j in range(len(batch_prompts)):
                prompt_future = loop.create_future()

                def make_callback(prompt_idx, result_future):
                    def callback(future):
                        try:
                            batch_outputs = future.result()

                            # Check if batch_outputs is valid and has enough items
                            if not batch_outputs or prompt_idx >= len(batch_outputs):
                                raise ValueError(f"Invalid batch output: expected at least {prompt_idx + 1} items, got {len(batch_outputs) if batch_outputs else 0}")

                            output = batch_outputs[prompt_idx]

                            # Check if output has valid completion
                            if not output.outputs:
                                raise ValueError(f"No outputs generated for prompt {prompt_idx}")

                            completion = output.outputs[0]

                            result = {
                                'tokens': completion.token_ids,
                            }
                            result_future.set_result(result)
                        except Exception as e:
                            print(f"Error in batch callback for prompt {prompt_idx}: {e}")
                            result_future.set_exception(e)
                    return callback

                batch_future.add_done_callback(make_callback(j, prompt_future))
                futures.append(prompt_future)

        return futures

    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        print("Shutting down vLLM backend...")

        if self.llm is not None:
            try:
                # vLLM doesn't have an explicit shutdown method
                # Rely on garbage collection
                del self.llm
                self.llm = None

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print("vLLM engine released and CUDA cache cleared")
            except Exception as e:
                print(f"Warning: Error during vLLM shutdown: {e}")

        self.sampling_params = None
        self.is_initialized = False

        print("vLLM backend shutdown complete")
