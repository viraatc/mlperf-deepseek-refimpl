"""
Data utilities for loading datasets and saving results.

Provides common functionality for data handling across all backends.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


def load_dataset(file_path: str, num_samples: Optional[int] = None, skip_samples: int = 0) -> pd.DataFrame:
    """
    Load dataset from pickle file.
    
    Args:
        file_path: Path to the pickle file
        num_samples: Optional limit on number of samples to load
        skip_samples: Number of samples to skip from the beginning
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file can't be loaded
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found!")
    
    print(f"Loading dataset from {file_path}...")
    
    try:
        with open(file_path, "rb") as f:
            df = pd.read_pickle(f)
    except Exception as e:
        raise Exception(f"Failed to load dataset from {file_path}: {e}")
    
    print(f"Loaded {len(df)} samples")
    
    # Skip samples if specified
    if skip_samples > 0:
        if skip_samples >= len(df):
            raise ValueError(f"skip_samples ({skip_samples}) is greater than or equal to total samples ({len(df)})")
        original_length = len(df)
        df = df.iloc[skip_samples:].reset_index(drop=True)
        print(f"Skipped first {skip_samples} samples (from {original_length} total)")
    
    # Limit number of samples if specified
    if num_samples is not None:
        original_length = len(df)
        df = df.head(num_samples)
        print(f"Limited to {len(df)} samples (from {original_length} total after skipping)")
    
    return df


def save_results(df: pd.DataFrame, 
                output_file: str, 
                add_timestamp: bool = True) -> str:
    """
    Save results DataFrame to pickle file.
    
    Args:
        df: DataFrame to save
        output_file: Output file path
        add_timestamp: Whether to add timestamp to filename
        
    Returns:
        Actual output file path used
    """
    # Add timestamp to filename if requested
    if add_timestamp:
        timestamp_suffix = time.strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(output_file)
        output_file = f"{base_name}_{timestamp_suffix}{ext}"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving results to {output_file}...")
    
    # Reset index before saving
    df_to_save = df.reset_index(drop=True)
    
    try:
        with open(output_file, "wb") as f:
            pickle.dump(df_to_save, f)
        print(f"Save completed: {len(df_to_save)} samples saved to {output_file}")
    except Exception as e:
        raise Exception(f"Failed to save results to {output_file}: {e}")
    
    return output_file


def prepare_output_dataframe(input_df: pd.DataFrame, 
                           backend_name: str) -> pd.DataFrame:
    """
    Prepare output DataFrame by cleaning up old columns.
    
    Args:
        input_df: Input DataFrame
        backend_name: Name of the backend being used
        
    Returns:
        Cleaned DataFrame ready for new results
    """
    df_output = input_df.copy()
    
    # Define columns to drop (old model outputs and unwanted columns)
    columns_to_drop = [
        # specify columns to drop here
    ]
    
    # Also drop any existing backend-specific columns
    backend_columns = [col for col in df_output.columns if col.startswith(f'{backend_name}_')]
    columns_to_drop.extend(backend_columns)
    
    # Drop columns that exist
    df_output = df_output.drop(
        columns=[col for col in columns_to_drop if col in df_output.columns]
    )
    
    return df_output


def add_standardized_columns(df: pd.DataFrame, 
                           results: List[Dict[str, Any]],
                           tokenized_prompts: List[List[int]]) -> pd.DataFrame:
    """
    Add standardized output columns to DataFrame.
    
    Args:
        df: Input DataFrame
        results: List of result dictionaries from backend
        tokenized_prompts: List of tokenized input prompts
        
    Returns:
        DataFrame with added standardized columns
    """
    # Add tokenized prompts
    df['prompt_tokens'] = tokenized_prompts
    
    # Add results columns
    df['raw_output'] = [r['raw_output'] for r in results]
    df['output_tokens'] = [r['output_tokens'] for r in results]
    df['output_token_len'] = [r['output_token_len'] for r in results]
    df['backend_used'] = [r['backend_used'] for r in results]
    
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that the dataset has required columns.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ['prompt']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    # Check for empty prompts
    empty_prompts = df['prompt'].isna().sum()
    if empty_prompts > 0:
        print(f"Warning: Found {empty_prompts} empty prompts in dataset")
    
    print(f"Dataset validation passed: {len(df)} samples with required columns") 