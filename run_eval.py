#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

from backends import BaseBackend
from utils import (
    load_dataset, save_results, validate_dataset
)


# Hardcoded tokenization configuration
TOKENIZATION_CONFIG = {
    "tokenizer_name": "deepseek-ai/DeepSeek-R1",
    "max_input_length": 32*1024, 
}


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with shared arguments only."""
    parser = argparse.ArgumentParser(
        description="Modular backend evaluation system for MLPerf DeepSeek reference implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Only essential arguments
    parser.add_argument("--backend", type=str, required=True,
                       choices=["trtllm", "vllm", "sglang"],
                       help="Backend to use for inference (use run_eval_mpi.py for pytorch)")
    
    parser.add_argument("--input-file", type=str,
                       default="data/final_output.pkl",
                       help="Input pickle file path")
    
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output pickle file path (auto-generated if not specified)")
    
    parser.add_argument("--async", action="store_true",
                       help="Use async generation instead of synchronous")
    
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to process from dataset (process all if not specified)")
    
    parser.add_argument("--skip-samples", type=int, default=0,
                       help="Number of samples to skip from the beginning of the dataset")
    
    parser.add_argument("--no-chat-template", action="store_false", dest="use_chat_template",
                       help="Disable chat template for tokenization (enabled by default)")
    
    return parser


def get_backend_class(backend_name: str) -> BaseBackend:
    """Get backend class by name."""
    backend_map = {
        'trtllm': 'backends.trtllm_backend.TRTLLMBackend',
        'vllm': 'backends.vllm_backend.VLLMBackend', 
        'sglang': 'backends.sglang_backend.SGLangBackend',
    }
    
    if backend_name not in backend_map:
        raise ValueError(f"Backend '{backend_name}' not supported in run_eval.py. "
                        f"Use run_eval_mpi.py for pytorch backend. "
                        f"Available backends: {', '.join(backend_map.keys())}")
                        
    module_path, class_name = backend_map[backend_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def tokenize_prompts(prompts: List[str], use_chat_template: bool = True) -> tuple[List[List[int]], List[str]]:
    """Tokenize prompts using the standard tokenizer.

    Args:
        prompts: List of prompt strings
        use_chat_template: Whether to use chat template if available

    Returns:
        tuple: (tokenized_prompts, processed_strings) where processed_strings
               are the prompts after tokenization and decoding (for debugging)
    """
    print(f"Loading tokenizer: {TOKENIZATION_CONFIG['tokenizer_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZATION_CONFIG['tokenizer_name'])
    
    # Tokenize all prompts
    tokenized = []
    processed_strings = []
    
    for prompt in prompts:
        # Check if tokenizer has apply_chat_template (for chat models) and we want to use it
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                max_length=TOKENIZATION_CONFIG['max_input_length'],
                truncation=True
            )
            # Decode to see what the actual prompt looks like
            processed_string = tokenizer.decode(tokens, skip_special_tokens=False)
        else:
            # Standard tokenization
            tokens = tokenizer.encode(
                prompt,
                truncation=True,
                max_length=TOKENIZATION_CONFIG['max_input_length']
            )
            processed_string = prompt
        
        tokenized.append(tokens)
        processed_strings.append(processed_string)
    
    return tokenized, processed_strings


async def run_async_inference(backend: BaseBackend, 
                            tokenized_prompts: List[List[int]],
                            prompt_strings: List[str]) -> List[Dict[str, Any]]:
    """Run async inference with proper error handling and progress bar that updates as tasks complete."""
    try:
        # Get futures from backend
        futures = backend.generate_async(tokenized_prompts, prompt_strings=prompt_strings)
        
        # Create a list to store results in order
        results = [None] * len(futures)
        
        # Create enumerated futures with their original indices for tracking
        indexed_futures = [(i, future) for i, future in enumerate(futures)]
        
        # Track completion for debugging
        completed_indices = set()
        
        # Process tasks with progress bar that updates as tasks complete
        with async_tqdm(total=len(futures), desc="Async inference", unit="prompt") as pbar:
            # Use asyncio.wait with FIRST_COMPLETED to handle out-of-order completion
            pending = {future for _, future in indexed_futures}
            
            while pending:
                # Wait for at least one future to complete
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                
                # Process all completed futures in this batch
                for completed_future in done:
                    # Find the original index for this completed future
                    original_idx = None
                    for idx, future in indexed_futures:
                        if future is completed_future:
                            original_idx = idx
                            break
                    
                    if original_idx is None:
                        print(f"\nWarning: Could not find original index for completed future")
                        continue
                    
                    # Check for duplicate completion
                    if original_idx in completed_indices:
                        print(f"\nWarning: Prompt {original_idx} completed multiple times!")
                        continue
                    
                    try:
                        # Get the result from the completed future
                        result = await completed_future
                        
                        # Store the result in the correct position
                        results[original_idx] = result
                        completed_indices.add(original_idx)
                        
                    except Exception as e:
                        print(f"\nError processing prompt {original_idx}: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exception(type(e), e, e.__traceback__)
                        
                        completed_indices.add(original_idx)
                        results[original_idx] = {'tokens': []}
                    
                    # Update progress bar after each completion
                    pbar.update(1)
        
        # Verify all results are populated
        if len(completed_indices) != len(futures):
            print(f"\nWarning: Completed {len(completed_indices)} != {len(futures)} total")
        
        for i, result in enumerate(results):
            if result is None:
                print(f"\nWarning: Missing result for prompt {i}, using empty result")
                results[i] = {'tokens': []}
        
        print(f"\nCompleted all {len(completed_indices)} prompts successfully")
        
        return results
    except Exception as e:
        print(f"Error during async inference: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_sync_inference(backend: BaseBackend, 
                      tokenized_prompts: List[List[int]],
                      prompt_strings: List[str]) -> List[Dict[str, Any]]:
    """Run sync inference with proper error handling."""
    try:
        results = backend.generate(tokenized_prompts, prompt_strings=prompt_strings)
        return results
    except Exception as e:
        print(f"Error during sync inference: {e}")
        raise


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = f"data/{args.backend}_output.pkl"
    
    # Get async flag using getattr since 'async' is a reserved keyword
    use_async = getattr(args, 'async', False)
    
    print("=" * 80)
    print("Modular Backend Evaluation System")
    print("=" * 80)
    print(f"Backend: {args.backend}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Mode: {'Async' if use_async else 'Sync'}")
    if args.num_samples:
        print(f"Sample limit: {args.num_samples}")
    print("=" * 80)
    
    try:
        # Load and validate dataset
        df = load_dataset(args.input_file, args.num_samples, args.skip_samples)
        validate_dataset(df)
        
        prompts = df['prompt'].tolist()
        
        # Tokenize prompts before initializing backend
        print("Tokenizing prompts...")
        tokenized_prompts, processed_strings = tokenize_prompts(prompts, args.use_chat_template)
        print(f"Tokenized {len(tokenized_prompts)} prompts")
        print(f"Tokenizer Max length: {TOKENIZATION_CONFIG['max_input_length']}")
        
        # Load tokenizer for decoding (if needed)
        print(f"Loading tokenizer for decoding: {TOKENIZATION_CONFIG['tokenizer_name']}...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZATION_CONFIG['tokenizer_name'])
        
        # Initialize backend
        backend_class = get_backend_class(args.backend)
        
        print(f"\nInitializing {args.backend.upper()} backend...")
        
        with backend_class() as backend:
            # Create new output dataframe with only required columns
            df_output = pd.DataFrame()
            
            # Copy only the required columns from input
            df_output['prompt'] = df['prompt']
            if 'ground_truth' in df.columns:
                df_output['ground_truth'] = df['ground_truth']
            if 'question' in df.columns:
                df_output['question'] = df['question']
            if 'dataset' in df.columns:
                df_output['dataset'] = df['dataset']
            
            # Run inference with pre-tokenized prompts
            if use_async:
                print("Running async inference...")
                raw_results = asyncio.run(run_async_inference(backend, tokenized_prompts, processed_strings))
            else:
                print("Running sync inference...")
                raw_results = run_sync_inference(backend, tokenized_prompts, processed_strings)
            
            # Process raw results into standardized format
            print("Processing results...")
            standardized_results = []
            for raw_result in tqdm(raw_results, desc="Processing results", unit="result"):
                # Decode tokens to get text
                text = tokenizer.decode(raw_result['tokens'], skip_special_tokens=True)
                
                standardized = {
                    'output_text': text,
                    'output_tok': raw_result['tokens'],
                    'output_tok_len': len(raw_result['tokens']),
                    'backend': args.backend,
                }
                standardized_results.append(standardized)
            
            # Add tokenized prompts and their lengths
            df_output['prompt_tok'] = tokenized_prompts
            df_output['prompt_tok_len'] = [len(tokens) for tokens in tokenized_prompts]
            
            # Add generated columns
            df_output['output_text'] = [r['output_text'] for r in standardized_results]
            df_output['output_tok'] = [r['output_tok'] for r in standardized_results]
            df_output['output_tok_len'] = [r['output_tok_len'] for r in standardized_results]
            df_output['backend'] = [r['backend'] for r in standardized_results]
            
            # Save results
            output_file = save_results(df_output, args.output_file, add_timestamp=True)
            
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved to: {output_file}")
            print(f"Output columns: {list(df_output.columns)}")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 