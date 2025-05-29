#!/usr/bin/env python3
"""MLPerf inference benchmark runner for distributed PyTorch backend using MPI/torchrun.

This script runs MLPerf inference benchmarks using the LoadGen library
with PyTorch backend in a distributed setting.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import builtins
import time

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlperf_loadgen as lg
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from backends.pytorch_backend import PyTorchBackend
from mlperf import OfflineSUT, ServerSUT, BaseSUT
from utils import load_dataset, validate_dataset, save_results

# Hardcoded tokenization configuration to match other runners
TOKENIZATION_CONFIG = {
    "tokenizer_name": "deepseek-ai/DeepSeek-R1", 
    "max_input_length": 32*1024, 
}


# Configure logging - only for rank 0
def setup_logging(rank: int):
    """Setup logging based on rank."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Disable logging for non-rank 0 processes
        logging.disable(logging.CRITICAL)
    
    return logging.getLogger(__name__)


class DistributedOfflineSUT(BaseSUT):
    """Distributed Offline SUT implementation for PyTorch backend.
    
    Only rank 0 interacts with LoadGen, but all ranks participate in inference.
    """
    
    def __init__(self, 
                 backend: 'PyTorchBackend',
                 dataset: List[List[int]],
                 prompt_strings: Optional[List[str]] = None,
                 name: str = "DistributedOfflineSUT",
                 rank: int = 0,
                 world_size: int = 1):
        """Initialize the distributed offline SUT.
        
        Args:
            backend: Backend instance to use for inference
            dataset: List of tokenized prompts
            prompt_strings: Optional list of prompt strings
            name: Name of the SUT
            rank: Process rank
            world_size: Total number of processes
        """
        super().__init__(name)
        self.backend = backend
        self.dataset = dataset
        self.prompt_strings = prompt_strings
        self.rank = rank
        self.world_size = world_size
        
        # Results storage (only rank 0)
        if self.rank == 0:
            self.results = {}
            self.index_to_id = {}
        
        # Flag to signal other ranks to exit
        self._should_exit = False
        
    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries for processing.
        
        Only called on rank 0 by LoadGen.
        
        Args:
            query_samples: List of MLPerf LoadGen query samples
        """
        if self.rank != 0:
            return
            
        logger = logging.getLogger(__name__)
        logger.info(f"Issuing {len(query_samples)} queries")
        
        # Process queries in batches
        batch_size = self.backend.config['batch_size']
        
        for i in range(0, len(query_samples), batch_size):
            batch_samples = query_samples[i:i+batch_size]
            
            # Prepare batch tokens
            batch_tokens = []
            batch_ids = []
            
            for sample in batch_samples:
                # Track index to ID mapping
                self.index_to_id[sample.index] = sample.id
                
                # Get tokens for this sample
                tokens = self.dataset[sample.index]
                batch_tokens.append(tokens)
                batch_ids.append(sample.id)
            
            # Signal other ranks to participate in generation
            if self.world_size > 1:
                signal = ["generate"]
                dist.broadcast_object_list(signal, src=0)
            
            # Generate using distributed backend
            # This will broadcast to all ranks internally
            generated_tokens = self.backend.generate_batch_distributed(batch_tokens)
            
            # Process results and send to LoadGen
            for j, (sample_id, tokens) in enumerate(zip(batch_ids, generated_tokens)):
                # Convert tokens to bytes for LoadGen
                token_array = np.array(tokens, dtype=np.int32)
                n_tokens = len(tokens)
                
                # Create LoadGen response
                response = lg.QuerySampleResponse(
                    sample_id,
                    token_array.ctypes.data,
                    token_array.nbytes,
                    n_tokens,
                )
                
                # Store result
                self.results[sample_id] = {
                    'tokens': tokens,
                }
                
                # Send response to LoadGen
                lg.QuerySamplesComplete([response])
            
            # Send idle signal to other ranks after batch completes
            if self.world_size > 1:
                idle_signal = [None]
                dist.broadcast_object_list(idle_signal, src=0)
    
    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Nothing to flush in this implementation
        pass
    
    def start(self) -> lg.ConstructSUT:
        """Start the SUT."""
        # Signal that we're starting
        if self.rank == 0:
            logger = logging.getLogger(__name__)
            logger.info("Starting Distributed Offline SUT")
        
        return super().start()
    
    def stop(self) -> None:
        """Stop the SUT."""
        # Signal other ranks to exit is now handled in the main loop
        # after LoadGen test completes
        
        # Clear results
        if self.rank == 0:
            self.results.clear()
            self.index_to_id.clear()
        
        super().stop()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results in order of dataset indices.
        
        Returns:
            List of result dictionaries with output_text, output_tok, and output_tok_len
        """
        if self.rank != 0:
            return []
            
        # Create a list to hold results in dataset order
        ordered_results = []
        
        # Get tokenizer for decoding
        tokenizer = self.backend.tokenizer
        
        # Process results in order of dataset indices
        for i in range(len(self.dataset)):
            # Get the sample ID for this index
            sample_id = self.index_to_id.get(i)
            
            if sample_id is not None and sample_id in self.results:
                result = self.results[sample_id]
                if 'tokens' in result:
                    output_text = ''
                    if tokenizer:
                        try:
                            output_text = tokenizer.decode(result['tokens'], skip_special_tokens=True)
                        except:
                            pass
                    
                    ordered_results.append({
                        'output_text': output_text,
                        'output_tok': result['tokens'],
                        'output_tok_len': len(result['tokens'])
                    })
                else:
                    # Result exists but no tokens
                    ordered_results.append({
                        'output_text': '',
                        'output_tok': [],
                        'output_tok_len': 0
                    })
            else:
                # No result for this index
                ordered_results.append({
                    'output_text': '',
                    'output_tok': [],
                    'output_tok_len': 0
                })
        
        return ordered_results


class DistributedQuerySampleLibrary:
    """MLPerf QuerySampleLibrary implementation for distributed execution."""

    def __init__(self, dataset: List[List[int]], dataset_strings: List[str], rank: int):
        """Initialize QSL with dataset.

        Args:
            dataset: List of tokenized prompts
            dataset_strings: List of original prompt strings
            rank: Process rank
        """
        self.dataset = dataset
        self.dataset_strings = dataset_strings
        self.count = len(dataset)
        self.perf_count = self.count  # Use all samples for performance
        self.loaded_samples = set()
        self.rank = rank

        # Only rank 0 creates the actual QSL object
        if self.rank == 0:
            self.qsl = lg.ConstructQSL(
                self.count,
                self.perf_count,
                lambda x: None,
                lambda x: None
            )
        else:
            self.qsl = None

    def __del__(self):
        """Cleanup."""
        if self.rank == 0 and hasattr(self, 'qsl') and self.qsl:
            lg.DestroyQSL(self.qsl)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for distributed MLPerf runner."""
    parser = argparse.ArgumentParser(
        description="Run MLPerf inference benchmarks with distributed PyTorch backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument("--input-file", type=str,
                       default="data/final_output.pkl",
                       help="Input pickle file with prompts")

    # MLPerf configuration
    parser.add_argument("--mlperf-conf", type=str, default="mlperf/mlperf.conf",
                       help="Path to MLPerf configuration file")

    parser.add_argument("--user-conf", type=str, default="mlperf/user.conf",
                       help="Path to user configuration file")

    parser.add_argument("--mode", type=str, default="offline",
                       choices=["offline", "server"],
                       help="MLPerf scenario mode (only offline supported for distributed)")

    parser.add_argument("--accuracy", action="store_true",
                       help="Run accuracy mode instead of performance")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="mlperf_results",
                       help="Directory for MLPerf output logs")

    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for detailed logs")
    
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output pickle file path (auto-generated if not specified)")

    # Tokenizer configuration
    parser.add_argument("--tokenizer", type=str,
                       default="deepseek-ai/DeepSeek-R1",
                       help="Tokenizer model name (for parsing log outputs)")
    
    parser.add_argument("--no-chat-template", action="store_false", dest="use_chat_template",
                       help="Disable chat template for tokenization (enabled by default)")

    return parser


def tokenize_prompts(prompts: List[str],
                    tokenizer_name: str,
                    max_length: int,
                    rank: int,
                    logger,
                    use_chat_template: bool = True) -> tuple[List[List[int]], List[str]]:
    """Tokenize prompts using the specified tokenizer.

    Args:
        prompts: List of prompt strings
        tokenizer_name: Name of the tokenizer model
        max_length: Maximum sequence length
        rank: Process rank
        logger: Logger instance
        use_chat_template: Whether to use chat template if available

    Returns:
        Tuple of (tokenized_prompts, processed_strings)
    """
    if rank == 0:
        logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenized = []
    processed_strings = []

    for prompt in prompts:
        # Check if tokenizer has chat template and we want to use it
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                max_length=max_length,
                truncation=True
            )
            processed_string = tokenizer.decode(tokens, skip_special_tokens=False)
        else:
            tokens = tokenizer.encode(
                prompt,
                truncation=True,
                max_length=max_length
            )
            processed_string = prompt

        tokenized.append(tokens)
        processed_strings.append(processed_string)

    return tokenized, processed_strings


def configure_loadgen(scenario: str,
                     accuracy_mode: bool,
                     mlperf_conf: Optional[str] = None,
                     user_conf: Optional[str] = None,
                     log_dir: Optional[str] = None,
                     model_name: str = "deepseek-r1") -> lg.TestSettings:
    """Configure LoadGen test settings.

    Args:
        scenario: MLPerf scenario ("offline" or "server")
        accuracy_mode: Whether to run in accuracy mode
        mlperf_conf: Path to MLPerf config file
        user_conf: Path to user config file
        log_dir: Directory for logs
        model_name: Model name for configuration (default: deepseek-r1)

    Returns:
        LoadGen TestSettings
    """
    settings = lg.TestSettings()

    # Set scenario
    if scenario.lower() == "offline":
        settings.scenario = lg.TestScenario.Offline
    elif scenario.lower() == "server":
        settings.scenario = lg.TestScenario.Server
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Set mode
    if accuracy_mode:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    # Load configurations if files exist
    if mlperf_conf and Path(mlperf_conf).exists():
        settings.FromConfig(mlperf_conf, model_name, scenario)
    if user_conf and Path(user_conf).exists():
        settings.FromConfig(user_conf, model_name, scenario)

    return settings


def run_loadgen_test(sut: DistributedOfflineSUT,
                    qsl: DistributedQuerySampleLibrary,
                    settings: lg.TestSettings,
                    log_settings: lg.LogSettings,
                    rank: int,
                    logger) -> None:
    """Run LoadGen test (only on rank 0).

    Args:
        sut: System Under Test instance
        qsl: Query Sample Library
        settings: Test settings
        log_settings: Log settings
        rank: Process rank
        logger: Logger instance
    """
    if rank == 0:
        # Start the test
        logger.info("Starting LoadGen test")
        lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings)
        logger.info("LoadGen test completed")


def main():
    """Main function."""
    _print = builtins.print  # Capture the original built-in print

    # Get distributed environment info
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Override print for non-rank 0 processes
    if rank != 0:
        print = lambda *_, **__: None

    # Setup logging
    logger = setup_logging(rank)

    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate mode for distributed
    if args.mode != "offline":
        if rank == 0:
            logger.error("Only offline mode is supported for distributed execution")
        sys.exit(1)

    # Create output directories (only rank 0)
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.log_dir:
            log_dir = Path(args.log_dir)
        else:
            log_dir = output_dir / args.mode / ("accuracy" if args.accuracy else "performance")
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("MLPerf Inference Benchmark Runner (Distributed PyTorch)")
        logger.info("=" * 80)
        logger.info(f"Backend: pytorch (distributed)")
        logger.info(f"World size: {world_size}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Accuracy: {args.accuracy}")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
    else:
        log_dir = None

    try:
        # Initialize PyTorch backend
        backend = PyTorchBackend()
        backend.initialize()

        # Only rank 0 handles dataset loading
        prompts = []
        tokenized_prompts = []
        processed_strings = []
        df = None

        if rank == 0:
            # Load dataset
            df = load_dataset(args.input_file)
            validate_dataset(df)
            prompts = df['prompt'].tolist()

            logger.info(f"Loaded {len(prompts)} prompts from dataset")

            # Tokenize prompts
            logger.info("Tokenizing prompts...")
            tokenized_prompts, processed_strings = tokenize_prompts(
                prompts,
                args.tokenizer,
                TOKENIZATION_CONFIG["max_input_length"],
                rank,
                logger,
                args.use_chat_template
            )
            logger.info(f"Tokenized {len(tokenized_prompts)} prompts")

        # Create SUT
        sut = DistributedOfflineSUT(
            backend=backend,
            dataset=tokenized_prompts if rank == 0 else [],
            prompt_strings=processed_strings if rank == 0 else [],
            name=f"pytorch_distributed_offline_sut",
            rank=rank,
            world_size=world_size
        )

        # Create QSL (only rank 0 needs the actual QSL)
        qsl = DistributedQuerySampleLibrary(
            tokenized_prompts if rank == 0 else [],
            processed_strings if rank == 0 else [],
            rank
        )

        # Only rank 0 configures and runs LoadGen
        if rank == 0:
            # Configure LoadGen
            settings = configure_loadgen(
                scenario=args.mode,
                accuracy_mode=args.accuracy,
                mlperf_conf=args.mlperf_conf,
                user_conf=args.user_conf,
                log_dir=str(log_dir)
            )

            # Update settings with dataset info
            settings.min_query_count = len(tokenized_prompts)
            settings.max_query_count = len(tokenized_prompts)
            settings.use_token_latencies = True

            # Configure logging
            log_settings = lg.LogSettings()
            log_settings.log_output.outdir = str(log_dir)
            log_settings.log_output.copy_summary_to_stdout = True
            log_settings.enable_trace = False

        # Start the SUT
        sut.start()

        try:
            if rank == 0:
                # Run test (only rank 0)
                logger.info("Running test...")
                run_loadgen_test(sut, qsl, settings, log_settings, rank, logger)
                logger.info("Completed test...")
                
                # Send exit signal to other ranks
                if world_size > 1:
                    exit_signal = [True]
                    dist.broadcast_object_list(exit_signal, src=0)
            else:
                # Non-rank 0 processes participate in distributed generation
                # They wait for signals from rank 0 and participate in generate_batch_distributed
                while True:
                    # First, check if we should exit
                    # We use a separate broadcast to signal exit
                    exit_check = [None]
                    dist.broadcast_object_list(exit_check, src=0)
                    
                    if exit_check[0] is True:
                        # Exit signal received
                        break
                    elif exit_check[0] == "generate":
                        # Signal to participate in generation
                        # The actual batch tokens will be broadcast inside generate_batch_distributed
                        backend.generate_batch_distributed(None)
                    # If exit_check[0] is None, continue waiting
        finally:
            # Stop the SUT
            sut.stop()

        if rank == 0:
            logger.info(f"Results saved to: {log_dir}")

            # Print summary
            summary_file = log_dir / "mlperf_log_summary.txt"
            if summary_file.exists():
                logger.info("\nTest Summary:")
                logger.info("-" * 40)
                with open(summary_file, 'r') as f:
                    _print(f.read())
            
            # Save results to pickle file (always, regardless of mode)
            if df is not None:
                logger.info("Processing results for output file...")
                
                # Load tokenizer for decoding
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
                
                # Create output dataframe similar to run_eval.py
                df_output = pd.DataFrame()
                
                # Copy required columns from input
                df_output['prompt'] = df['prompt']
                if 'ground_truth' in df.columns:
                    df_output['ground_truth'] = df['ground_truth']
                if 'question' in df.columns:
                    df_output['question'] = df['question']
                if 'dataset' in df.columns:
                    df_output['dataset'] = df['dataset']
                
                # Add tokenized prompts and their lengths
                df_output['prompt_tok'] = tokenized_prompts
                df_output['prompt_tok_len'] = [len(tokens) for tokens in tokenized_prompts]
                
                # Get results from SUT (if available)
                if hasattr(sut, 'get_results'):
                    results = sut.get_results()
                    
                    # Process results
                    output_texts = []
                    output_toks = []
                    output_tok_lens = []
                    
                    for result in results:
                        # If we have tokens but no text, decode them
                        if result['output_tok'] and not result['output_text']:
                            try:
                                result['output_text'] = tokenizer.decode(result['output_tok'], skip_special_tokens=True)
                            except:
                                pass
                        
                        output_texts.append(result.get('output_text', ''))
                        output_toks.append(result.get('output_tok', []))
                        output_tok_lens.append(result.get('output_tok_len', 0))
                    
                    df_output['output_text'] = output_texts
                    df_output['output_tok'] = output_toks
                    df_output['output_tok_len'] = output_tok_lens
                else:
                    # If results not available from SUT, create empty columns
                    logger.warning("Results not available from SUT, creating empty output columns")
                    df_output['output_text'] = [''] * len(df_output)
                    df_output['output_tok'] = [[]] * len(df_output)
                    df_output['output_tok_len'] = [0] * len(df_output)
                
                # Add backend info
                df_output['backend'] = 'pytorch'
                
                # Determine output file path
                if args.output_file:
                    output_file = args.output_file
                else:
                    # Create output file path in the data directory
                    mode_str = "accuracy" if args.accuracy else "performance"
                    output_file = f"data/pytorch_distributed_mlperf_{args.mode}_{mode_str}_output.pkl"
                
                # Save results
                saved_file = save_results(df_output, output_file, add_timestamp=True)
                logger.info(f"Results saved to: {saved_file}")
                logger.info(f"Output columns: {list(df_output.columns)}")

    except KeyboardInterrupt:
        if rank == 0:
            logger.info("Test interrupted by user")
        backend.shutdown()
        if world_size > 1:
            dist.destroy_process_group()
        sys.exit(1)
    except Exception as e:
        if rank == 0:
            logger.error(f"Test failed: {e}", exc_info=True)
        backend.shutdown()
        if world_size > 1:
            dist.destroy_process_group()
        sys.exit(1)

    # Clean up
    backend.shutdown()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 