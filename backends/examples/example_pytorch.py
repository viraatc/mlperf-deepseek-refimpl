#!/usr/bin/env python3
"""
Example script demonstrating PyTorch backend usage.

This script shows how to use the modular backend system with PyTorch
for both synchronous and asynchronous inference.
"""

import asyncio
import sys
import time
from pathlib import Path

# Disable tokenizers parallelism to avoid forking issues
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backends import PyTorchBackend

# Hardcoded tokenization configuration to match main runners
TOKENIZATION_CONFIG = {
    "tokenizer_name": "deepseek-ai/DeepSeek-R1",
    "max_input_length": 32*1024,  # max length for deepseek-r1
}


def tokenize_prompts(prompts, tokenizer_name=TOKENIZATION_CONFIG["tokenizer_name"]):
    """Tokenize prompts using the DeepSeek tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized = []
    
    for prompt in prompts:
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                add_generation_prompt=True,
                truncation=True,
                max_length=TOKENIZATION_CONFIG["max_input_length"]
            )
        else:
            tokens = tokenizer.encode(
                prompt, 
                truncation=True, 
                max_length=TOKENIZATION_CONFIG["max_input_length"]
            )
            
        tokenized.append(tokens)
    
    return tokenized, tokenizer


def example_sync_inference():
    """Example of synchronous inference with PyTorch backend."""
    print("=" * 60)
    print("PyTorch Synchronous Inference Example")
    print("=" * 60)
    
    # Sample prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
    ]
    
    # Tokenize prompts
    print("Tokenizing prompts...")
    tokenized_prompts, tokenizer = tokenize_prompts(prompts)
    
    # Initialize and use backend
    with PyTorchBackend() as backend:
        print(f"Backend initialized")
        
        start_time = time.time()
        results = backend.generate(tokenized_prompts)
        total_time = time.time() - start_time
        
        print(f"\nGenerated {len(results)} responses in {total_time:.2f}s")
        
        for i, result in enumerate(results):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            # Decode the generated tokens
            response_text = tokenizer.decode(result['tokens'], skip_special_tokens=True)
            print(f"Response: {response_text[:200]}...")
            print(f"Tokens generated: {len(result['tokens'])}")


async def example_async_inference():
    """Example of asynchronous inference with PyTorch backend."""
    print("\n" + "=" * 60)
    print("PyTorch Asynchronous Inference Example")
    print("=" * 60)
    
    # Sample prompts
    prompts = [
        "Describe the process of photosynthesis.",
        "What are the benefits of renewable energy?",
        "How does machine learning work?",
        "Explain the theory of relativity.",
        "What is the importance of biodiversity?",
    ]
    
    # Tokenize prompts
    print("Tokenizing prompts...")
    tokenized_prompts, tokenizer = tokenize_prompts(prompts)
    
    # Initialize and use backend
    with PyTorchBackend() as backend:
        print(f"Backend initialized")
        
        start_time = time.time()
        # Get futures from backend
        futures = backend.generate_async(tokenized_prompts)
        
        # Create mapping for out-of-order completion tracking
        future_to_index = {future: i for i, future in enumerate(futures)}
        results = [None] * len(futures)
        
        # Verify mapping consistency
        assert len(future_to_index) == len(futures), f"Future mapping mismatch: {len(future_to_index)} != {len(futures)}"
        
        # Track completion for debugging
        completed_indices = set()
        
        print("Processing async futures as they complete...")
        # Use asyncio.as_completed to process futures as they complete (out-of-order)
        for completed_future in asyncio.as_completed(futures):
            try:
                # Get the result from the completed future
                result = await completed_future
                
                # Find which index this future corresponds to
                assert completed_future in future_to_index, "Completed future not found in mapping!"
                idx = future_to_index[completed_future]
                
                # Check for duplicate completion
                assert idx not in completed_indices, f"Prompt {idx} completed multiple times!"
                completed_indices.add(idx)
                
                # Store the result in the correct position
                results[idx] = result
                
                print(f"  Completed prompt {idx + 1}/{len(futures)}")
                
            except Exception as e:
                # Find which index this future corresponds to
                assert completed_future in future_to_index, "Failed future not found in mapping!"
                idx = future_to_index[completed_future]
                
                print(f"Error processing prompt {idx}: {type(e).__name__}: {e}")
                
                completed_indices.add(idx)
                results[idx] = {'tokens': []}
        
        # Verify all results are populated
        assert len(completed_indices) == len(futures), f"Completed {len(completed_indices)} != {len(futures)} total"
        for i, result in enumerate(results):
            assert result is not None, f"Missing result for prompt {i}"
        
        total_time = time.time() - start_time
        
        print(f"\nGenerated {len(results)} responses in {total_time:.2f}s")
        
        for i, result in enumerate(results):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            # Decode the generated tokens
            response_text = tokenizer.decode(result['tokens'], skip_special_tokens=True)
            print(f"Response: {response_text[:200]}...")
            print(f"Tokens generated: {len(result['tokens'])}")


def main():
    """Run both sync and async examples."""
    try:
        # Run synchronous example
        example_sync_inference()
        
        # Run asynchronous example
        asyncio.run(example_async_inference())
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main() 