#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from typing import Optional
import builtins

import torch
import torch.distributed as dist

# Import PyTorchBackend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backends.pytorch_backend import PyTorchBackend
from utils import save_results

# Hardcoded tokenization configuration to match other runners
TOKENIZATION_CONFIG = {
    "max_input_length": 32*1024, 
}

def main(
    input_pickle_path: str,
    output_pickle_path: str,
    num_samples: Optional[int] = None,
    skip_samples: int = 0,
    use_chat_template: bool = True,
) -> None:
    """
    Main function to load the model, process prompts from a DataFrame, and save results.
    """
    _print = builtins.print  # Capture the original built-in print

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Override print for non-rank 0 processes
    if rank != 0:
        print = lambda *_, **__: None

    # Initialize PyTorch backend
    backend = PyTorchBackend()
    backend.initialize()
    
    # Get tokenizer from backend
    tokenizer = backend.tokenizer

    # Only rank 0 handles data
    prompts_text_list = []
    df_for_results = None

    if rank == 0:
        # Load DataFrame
        _print(f"Loading input DataFrame from {input_pickle_path}...")
        try:
            df_for_results = pd.read_pickle(input_pickle_path)
            _print(f"Loaded DataFrame with {len(df_for_results)} rows and columns: {df_for_results.columns.tolist()}")
            
            # Apply skip_samples if specified
            if skip_samples > 0:
                if skip_samples >= len(df_for_results):
                    _print(f"Error: skip_samples ({skip_samples}) is greater than or equal to total samples ({len(df_for_results)})")
                    backend.shutdown()
                    if world_size > 1:
                        dist.destroy_process_group()
                    return
                _print(f"Skipping first {skip_samples} samples")
                df_for_results = df_for_results.iloc[skip_samples:].copy()
                # Reset index to ensure sequential indices starting from 0
                df_for_results = df_for_results.reset_index(drop=True)
            
            # Apply num_samples limit if specified
            if num_samples is not None and num_samples < len(df_for_results):
                _print(f"Limiting to first {num_samples} samples (out of {len(df_for_results)} total after skipping)")
                df_for_results = df_for_results.head(num_samples).copy()
                # Reset index to ensure sequential indices starting from 0
                df_for_results = df_for_results.reset_index(drop=True)
                
        except Exception as e:
            _print(f"Error loading input pickle file: {e}")
            backend.shutdown()
            if world_size > 1:
                dist.destroy_process_group()
            return

        if 'prompt' not in df_for_results.columns:
            _print("Error: 'prompt' column not found in the input DataFrame.")
            backend.shutdown()
            if world_size > 1:
                dist.destroy_process_group()
            return

        prompts_text_list = df_for_results['prompt'].tolist()
        _print(f"Extracted {len(prompts_text_list)} prompts from 'prompt' column.")

        # Pre-initialize output columns
        df_for_results['output_text'] = ""
        df_for_results['output_tok'] = None
        df_for_results['output_tok'] = df_for_results['output_tok'].astype('object')
        df_for_results['output_tok_len'] = 0
        df_for_results['prompt_tok'] = None
        df_for_results['prompt_tok'] = df_for_results['prompt_tok'].astype('object')
        df_for_results['prompt_tok_len'] = 0
        df_for_results['backend'] = 'pytorch'

    # Broadcast the number of prompts to all ranks
    if world_size > 1:
        if rank == 0:
            num_prompts_tensor = torch.tensor(len(prompts_text_list), dtype=torch.long, device="cuda")
        else:
            num_prompts_tensor = torch.empty(1, dtype=torch.long, device="cuda")
        dist.broadcast(num_prompts_tensor, src=0)
        num_total_prompts = num_prompts_tensor.item()
    else:
        num_total_prompts = len(prompts_text_list)

    batch_size = backend.config['batch_size']

    # Process prompts in batches
    for i in range(0, num_total_prompts, batch_size):
        current_batch_num = (i // batch_size) + 1
        current_batch_prompt_texts = None
        current_batch_prompt_tokens = None

        if rank == 0:
            current_batch_prompt_texts = prompts_text_list[i:i+batch_size]
            # Tokenize on rank 0
            current_batch_prompt_tokens = []
            for p_text in current_batch_prompt_texts:
                if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
                    tokens = tokenizer.apply_chat_template(
                        [{"role": "user", "content": p_text}], 
                        add_generation_prompt=True,
                        max_length=TOKENIZATION_CONFIG['max_input_length'],
                        truncation=True
                    )
                else:
                    tokens = tokenizer.encode(
                        p_text,
                        truncation=True,
                        max_length=TOKENIZATION_CONFIG['max_input_length']
                    )
                current_batch_prompt_tokens.append(tokens)
            
            _print(f"Processing batch {current_batch_num}, size {len(current_batch_prompt_tokens)}")

        # All ranks call generate_batch_distributed
        generated_tokens_for_batch = backend.generate_batch_distributed(
            current_batch_prompt_tokens if rank == 0 else None
        )

        if rank == 0:
            # Decode tokens to text
            decoded_texts_for_batch = tokenizer.batch_decode(
                generated_tokens_for_batch, skip_special_tokens=True
            )

            # Update DataFrame for the current batch
            start_index_in_df = i
            num_items_in_batch_output = len(decoded_texts_for_batch)

            for batch_idx in range(num_items_in_batch_output):
                original_df_idx = start_index_in_df + batch_idx
                if original_df_idx < len(df_for_results):
                    # Use at for assignments with list values
                    df_for_results.at[original_df_idx, 'prompt_tok'] = current_batch_prompt_tokens[batch_idx]
                    df_for_results.at[original_df_idx, 'prompt_tok_len'] = len(current_batch_prompt_tokens[batch_idx])
                    df_for_results.at[original_df_idx, 'output_text'] = decoded_texts_for_batch[batch_idx]
                    df_for_results.at[original_df_idx, 'output_tok'] = generated_tokens_for_batch[batch_idx]
                    df_for_results.at[original_df_idx, 'output_tok_len'] = len(generated_tokens_for_batch[batch_idx])

            _print(f"Batch {current_batch_num} completed.")

    if rank == 0 and df_for_results is not None:
        _print(f"All batches processed. Saving results to {output_pickle_path}")
        
        # Keep only required columns in the same order as run_eval.py
        output_columns = ['prompt', 'ground_truth', 'question', 'dataset', 'prompt_tok', 'prompt_tok_len', 
                         'output_text', 'output_tok', 'output_tok_len', 'backend']
        # Filter to only columns that exist
        output_columns = [col for col in output_columns if col in df_for_results.columns]
        df_output = df_for_results[output_columns]
        
        try:
            saved_file = save_results(df_output, output_pickle_path, add_timestamp=True)
            _print(f"Successfully saved results to {saved_file}")
        except Exception as e:
            _print(f"Error saving output pickle file: {e}")

    # Clean up
    backend.shutdown()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run distributed inference with PyTorch backend using MPI/torchrun"
    )
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input pickle file")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to the output pickle file (auto-generated if not specified)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process from dataset (process all if not specified)")
    parser.add_argument("--skip-samples", type=int, default=0,
                        help="Number of samples to skip from the beginning of the dataset")
    parser.add_argument("--no-chat-template", action="store_false", dest="use_chat_template",
                        help="Disable chat template for tokenization (enabled by default)")

    args = parser.parse_args()

    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = "data/pytorch_output.pkl"

    main(
        args.input_file,
        args.output_file,
        args.num_samples,
        args.skip_samples,
        args.use_chat_template
    ) 