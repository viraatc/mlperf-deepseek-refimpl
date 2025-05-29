#!/usr/bin/env python3
"""MLPerf inference benchmark runner using LoadGen.

This script runs MLPerf inference benchmarks using the LoadGen library
with our modular backend system.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlperf_loadgen as lg
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from backends import BaseBackend
from mlperf import OfflineSUT, ServerSUT, BaseSUT
from utils import load_dataset, validate_dataset, save_results

# Hardcoded tokenization configuration to match other runners
TOKENIZATION_CONFIG = {
    "tokenizer_name": "deepseek-ai/DeepSeek-R1",
    "max_input_length": 32*1024,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuerySampleLibrary:
    """MLPerf QuerySampleLibrary implementation."""

    def __init__(self, dataset: List[List[int]], dataset_strings: List[str]):
        """Initialize QSL with dataset.

        Args:
            dataset: List of tokenized prompts
            dataset_strings: List of original prompt strings
        """
        self.dataset = dataset
        self.dataset_strings = dataset_strings
        self.count = len(dataset)
        self.perf_count = self.count  # Use all samples for performance
        self.loaded_samples = set()

        # Create the actual QSL object using LoadGen's constructor
        self.qsl = lg.ConstructQSL(
            self.count,
            self.perf_count,
            lambda x: None,
            lambda x: None
        )

        logger.info(f"Created QSL with {self.count} samples")

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'qsl') and self.qsl:
            lg.DestroyQSL(self.qsl)
        logger.info("QSL destroyed")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for MLPerf runner."""
    parser = argparse.ArgumentParser(
        description="Run MLPerf inference benchmarks with modular backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Backend selection
    parser.add_argument("--backend", type=str, required=True,
                       choices=["trtllm", "vllm", "sglang", "pytorch"],
                       help="Backend to use for inference (pytorch requires single GPU)")

    # Scenario selection
    parser.add_argument("--mode", type=str, default="offline",
                       choices=["offline", "server"],
                       help="MLPerf scenario mode")

    # Dataset arguments
    parser.add_argument("--input-file", type=str,
                       default="data/final_output.pkl",
                       help="Input pickle file with prompts")

    # MLPerf configuration
    parser.add_argument("--mlperf-conf", type=str, default="mlperf/mlperf.conf",
                       help="Path to MLPerf configuration file")

    parser.add_argument("--user-conf", type=str, default="mlperf/user.conf",
                       help="Path to user configuration file")

    parser.add_argument("--scenario", type=str, default=None,
                       choices=["Offline", "Server"],
                       help="MLPerf scenario (overrides --mode)")

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
                       default=TOKENIZATION_CONFIG["tokenizer_name"],
                       help="Tokenizer model name")

    parser.add_argument("--no-chat-template", action="store_false", dest="use_chat_template",
                       help="Disable chat template for tokenization (enabled by default)")

    parser.add_argument("--max-input-length", type=int, default=TOKENIZATION_CONFIG["max_input_length"],
                       help="Maximum input length for tokenization")

    return parser


def get_backend_instance(backend_name: str, args: argparse.Namespace) -> BaseBackend:
    """Get backend instance by name.

    Args:
        backend_name: Name of the backend
        args: Command line arguments

    Returns:
        Backend instance
    """
    backend_map = {
        'trtllm': 'backends.trtllm_backend.TRTLLMBackend',
        'vllm': 'backends.vllm_backend.VLLMBackend',
        'sglang': 'backends.sglang_backend.SGLangBackend',
        'pytorch': 'backends.pytorch_backend.PyTorchBackend'
    }

    if backend_name not in backend_map:
        raise ValueError(f"Backend '{backend_name}' not supported")

    module_path, class_name = backend_map[backend_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    backend_class = getattr(module, class_name)

    # Create backend instance
    return backend_class()


def tokenize_prompts(prompts: List[str],
                    tokenizer_name: str,
                    max_length: int,
                    use_chat_template: bool = True) -> tuple[List[List[int]], List[str]]:
    """Tokenize prompts using the specified tokenizer.

    Args:
        prompts: List of prompt strings
        tokenizer_name: Name of the tokenizer model
        max_length: Maximum sequence length
        use_chat_template: Whether to use chat template if available

    Returns:
        Tuple of (tokenized_prompts, processed_strings)
    """
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


def run_loadgen_test(sut: Union[OfflineSUT, ServerSUT],
                    qsl: QuerySampleLibrary,
                    settings: lg.TestSettings,
                    log_settings: lg.LogSettings) -> None:
    """Run LoadGen test.

    Args:
        sut: System Under Test instance
        qsl: Query Sample Library
        settings: Test settings
        log_settings: Log settings
    """
    # Start the test
    logger.info("Starting LoadGen test")
    lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings)
    logger.info("LoadGen test completed")


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle scenario override
    if args.scenario:
        args.mode = args.scenario.lower()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = output_dir / args.mode / ("accuracy" if args.accuracy else "performance")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MLPerf Inference Benchmark Runner")
    logger.info("=" * 80)
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Accuracy: {args.accuracy}")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    try:
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
            args.max_input_length,
            args.use_chat_template
        )
        logger.info(f"Tokenized {len(tokenized_prompts)} prompts")

        # Create backend
        logger.info(f"Initializing {args.backend} backend...")
        backend = get_backend_instance(args.backend, args)

        # Use backend context manager to ensure initialization and cleanup
        with backend:
            # Create SUT
            if args.mode == "offline":
                sut = OfflineSUT(
                    backend=backend,
                    dataset=tokenized_prompts,
                    prompt_strings=processed_strings,
                    name=f"{args.backend}_offline_sut"
                )
            else:  # server
                sut = ServerSUT(
                    backend=backend,
                    dataset=tokenized_prompts,
                    prompt_strings=processed_strings,
                    name=f"{args.backend}_server_sut"
                )

            # Create QSL
            qsl = QuerySampleLibrary(tokenized_prompts, processed_strings)

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
            settings.server_coalesce_queries = True

            # Configure logging
            log_settings = lg.LogSettings()
            log_settings.log_output.outdir = str(log_dir)
            log_settings.log_output.copy_summary_to_stdout = True
            log_settings.enable_trace = False

            # Start the SUT
            sut.start()

            try:
                # Run test
                logger.info("Running  test...")
                run_loadgen_test(sut, qsl, settings, log_settings)
                logger.info("Completed test...")
                
                # Get results BEFORE stopping the SUT
                if hasattr(sut, 'get_results'):
                    sut_results = sut.get_results()
                else:
                    sut_results = None
            finally:
                # Stop the SUT
                sut.stop()

            logger.info(f"Results saved to: {log_dir}")
            
            # Save results to pickle file (always, regardless of mode)
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
            if sut_results:
                # Process results
                output_texts = []
                output_toks = []
                output_tok_lens = []
                
                for result in sut_results:
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
            df_output['backend'] = args.backend
            
            # Determine output file path
            if args.output_file:
                output_file = args.output_file
            else:
                # Create output file path in the log directory
                mode_str = "accuracy" if args.accuracy else "performance"
                output_file = str(log_dir / f"{args.backend}_mlperf_{args.mode}_{mode_str}_output.pkl")
            
            # Save results
            saved_file = save_results(df_output, output_file, add_timestamp=True)
            logger.info(f"Results saved to: {saved_file}")
            logger.info(f"Output columns: {list(df_output.columns)}")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()