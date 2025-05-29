#!/usr/bin/env python3
"""Offline SUT implementation for MLPerf inference benchmarks."""

import asyncio
import logging
import threading
from typing import List, Dict, Any, Optional
import numpy as np
import mlperf_loadgen as lg
import pdb

from .base_sut import BaseSUT
from backends import BaseBackend


logger = logging.getLogger(__name__)


class OfflineSUT(BaseSUT):
    """Offline scenario SUT implementation.
    
    This SUT uses async inference APIs for concurrent processing.
    All queries are processed asynchronously without explicit batching.
    """
    
    def __init__(self, 
                 backend: BaseBackend,
                 dataset: List[List[int]],
                 prompt_strings: Optional[List[str]] = None,
                 name: str = "OfflineSUT"):
        """Initialize the offline SUT.
        
        Args:
            backend: Backend instance to use for inference
            dataset: List of tokenized prompts
            prompt_strings: Optional list of prompt strings (for backends that need them)
            name: Name of the SUT
        """
        super().__init__(name)
        self.backend = backend
        self.dataset = dataset
        self.prompt_strings = prompt_strings
        
        # Async event loop and thread
        self.loop = None
        self.loop_thread = None
        self.stop_event = threading.Event()
        
        # Track pending futures
        self.pending_futures = []
        self.futures_lock = threading.Lock()
        
        # Results storage
        self.results = {}
        self.results_lock = threading.Lock()
        
        # Track index to sample ID mapping
        self.index_to_id = {}
        
    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries asynchronously.
        
        Args:
            query_samples: List of MLPerf LoadGen query samples
        """
        logger.info(f"Issuing {len(query_samples)} queries")
        
        # Check prompt lengths
        for sample in query_samples:
            prompt_length = len(self.dataset[sample.index])
            # FIXME: This is a hardcoded limit that should be made configurable
            if prompt_length > 3135:
                logger.error(f"Prompt length {prompt_length} exceeds maximum allowed length of 3135 tokens for sample index {sample.index}")
                logger.error(f"Entering debugger. You can inspect:")
                logger.error(f"  - sample.index: {sample.index}")
                logger.error(f"  - prompt_length: {prompt_length}")
                logger.error(f"  - self.dataset[sample.index][:50]: {self.dataset[sample.index][:50]}...")
                logger.error(f"  - len(self.dataset): {len(self.dataset)}")
                pdb.set_trace()  # Drop into debugger here
            assert prompt_length <= 3135, f"Prompt length {prompt_length} exceeds maximum allowed length of 3135 tokens for sample index {sample.index}"
        
        # Schedule queries in the async loop
        for sample in query_samples:
            logger.info(f"Scheduling query - ID: {sample.id}, Index: {sample.index}")
            # Track index to ID mapping
            self.index_to_id[sample.index] = sample.id
            future = asyncio.run_coroutine_threadsafe(
                self._process_query_async(sample),
                self.loop
            )
            
            with self.futures_lock:
                self.pending_futures.append(future)
            
    def flush_queries(self) -> None:
        """Wait for all queries to complete."""
        # Wait for all pending futures to complete
        with self.futures_lock:
            futures_to_wait = list(self.pending_futures)
            
        for future in futures_to_wait:
            try:
                future.result()  # Wait for completion
            except Exception as e:
                logger.error(f"Error waiting for query completion: {e}")
                
        # Clear pending futures
        with self.futures_lock:
            self.pending_futures.clear()
            
    async def _process_query_async(self, sample: lg.QuerySample):
        """Process a single query asynchronously.
        
        Args:
            sample: Query sample to process
        """
        try:
            # Get prompt for this sample
            index = sample.index
            tokenized_prompt = [self.dataset[index]]
            
            # Get prompt string if available
            prompt_string = None
            if self.prompt_strings:
                prompt_string = [self.prompt_strings[index]]
            
            # Generate response asynchronously
            futures = self.backend.generate_async(
                tokenized_prompt,
                prompt_strings=prompt_string
            )
            
            # Create mapping for out-of-order completion tracking
            future_to_index = {future: i for i, future in enumerate(futures)}
            results = [None] * len(futures)
            
            # Verify mapping consistency
            assert len(future_to_index) == len(futures), f"Future mapping mismatch: {len(future_to_index)} != {len(futures)}"
            
            # Track completion
            completed_indices = set()
            
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
                    
                except Exception as e:
                    # Find which index this future corresponds to
                    assert completed_future in future_to_index, "Failed future not found in mapping!"
                    idx = future_to_index[completed_future]
                    
                    logger.error(f"Error processing future for prompt {idx}: {type(e).__name__}: {e}")
                    
                    completed_indices.add(idx)
                    results[idx] = {'tokens': []}
            
            # Verify all results are populated
            assert len(completed_indices) == len(futures), f"Completed {len(completed_indices)} != {len(futures)} total"
            for i, result in enumerate(results):
                assert result is not None, f"Missing result for prompt {i}"
            
            # Get the result (should be only one since we passed one prompt)
            result = results[0]
            
            # Convert tokens to bytes for LoadGen
            token_array = np.array(result['tokens'], dtype=np.int32)
            n_tokens = len(result['tokens'])
            
            # Create LoadGen response
            response = lg.QuerySampleResponse(
                sample.id,
                token_array.ctypes.data,
                token_array.nbytes,
                n_tokens,
            )
            
            # Store result for later access if needed
            with self.results_lock:
                self.results[sample.id] = result
                
            # Send response to LoadGen
            lg.QuerySamplesComplete([response])
            
        except Exception as e:
            logger.error(f"Error processing query {sample.id}: {e}")
            # Send empty response for failed query
            empty_array = np.array([], dtype=np.int32)
            response = lg.QuerySampleResponse(
                sample.id, 
                empty_array.ctypes.data if empty_array.size > 0 else 0,
                empty_array.nbytes
            )
            lg.QuerySamplesComplete([response])
            
    def _run_event_loop(self):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def start(self) -> lg.ConstructSUT:
        """Start the SUT and async event loop."""
        # Create and start event loop in separate thread
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop)
        self.loop_thread.start()
        
        # Call parent start
        return super().start()
        
    def stop(self) -> None:
        """Stop the SUT and clean up."""
        # Stop the event loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            
        # Wait for thread to finish
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join()
            
        # Close the loop
        if self.loop:
            self.loop.close()
            self.loop = None
            
        # Clear results
        self.results.clear()
        self.index_to_id.clear()
        
        # Call parent stop
        super().stop()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results in order of dataset indices.
        
        Returns:
            List of result dictionaries with output_text, output_tok, and output_tok_len
        """
        # Create a list to hold results in dataset order
        ordered_results = []
        
        # Get tokenizer for decoding (if available from backend)
        tokenizer = getattr(self.backend, 'tokenizer', None)
        
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