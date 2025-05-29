#!/usr/bin/env python3
"""Server SUT implementation for MLPerf inference benchmarks."""

import asyncio
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import mlperf_loadgen as lg
from dataclasses import dataclass
from datetime import datetime

from .base_sut import BaseSUT
from backends import BaseBackend


logger = logging.getLogger(__name__)


@dataclass
class QueryInfo:
    """Information about a query."""
    query_id: int
    index: int
    issued_time: float
    future: Optional[asyncio.Future] = None
    result: Optional[Dict[str, Any]] = None
    completed: bool = False


class ServerSUT(BaseSUT):
    """Server scenario SUT implementation.
    
    This SUT uses async inference APIs to handle queries with latency constraints.
    Each query is processed individually to minimize latency.
    """
    
    def __init__(self,
                 backend: BaseBackend,
                 dataset: List[List[int]],
                 prompt_strings: Optional[List[str]] = None,
                 name: str = "ServerSUT"):
        """Initialize the server SUT.
        
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
        
        # Prompt-output async results table
        self.query_table: Dict[int, QueryInfo] = {}
        self.query_table_lock = threading.Lock()
        
    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries by scheduling them for async processing.
        
        This method returns immediately after enqueuing the queries.
        
        Args:
            query_samples: List of MLPerf LoadGen query samples
        """
        if self.loop is None:
            logger.error("Event loop not initialized")
            return
            
        # Check prompt lengths
        for sample in query_samples:
            prompt_length = len(self.dataset[sample.index])
            # FIXME: This is a hardcoded limit that should be made configurable
            assert prompt_length <= 3135, f"Prompt length {prompt_length} exceeds maximum allowed length of 3135 tokens for sample index {sample.index}"
            
        # Enqueue all queries immediately
        for sample in query_samples:
            # Create query info
            query_info = QueryInfo(
                query_id=sample.id,
                index=sample.index,
                issued_time=time.time()
            )
            
            # Add to table
            with self.query_table_lock:
                self.query_table[sample.id] = query_info
            
            # Schedule async processing
            future = asyncio.run_coroutine_threadsafe(
                self._process_query_async(query_info),
                self.loop
            )
            query_info.future = future
            
    def flush_queries(self) -> None:
        """Wait for all queries to complete."""
        if self.loop is None:
            return
            
        # Collect all pending futures
        pending_futures = []
        with self.query_table_lock:
            for query_info in self.query_table.values():
                if query_info.future and not query_info.completed:
                    pending_futures.append(query_info.future)
        
        # Wait for all futures to complete
        for future in pending_futures:
            try:
                future.result()  # This blocks until the future completes
            except Exception as e:
                logger.error(f"Error waiting for query completion: {e}")
                
        # Clear the query table
        with self.query_table_lock:
            self.query_table.clear()
            
    async def _process_query_async(self, query_info: QueryInfo):
        """Process a single query asynchronously.
        
        Args:
            query_info: Query information
        """
        # Get prompt for this query
        tokenized_prompt = [self.dataset[query_info.index]]
        prompt_string = None
        if self.prompt_strings:
            prompt_string = [self.prompt_strings[query_info.index]]
            
        try:
            # Record start time for first token latency
            start_time = time.time()
            
            # Get futures from backend
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
            
            # Store result in query info
            query_info.result = result
            
            # Process result
            if result and 'tokens' in result:
                # Convert tokens to bytes for LoadGen
                token_array = np.array(result['tokens'], dtype=np.int32)
                n_tokens = len(result['tokens'])
                
                # Create LoadGen response
                response = lg.QuerySampleResponse(
                    query_info.query_id,
                    token_array.ctypes.data,
                    token_array.nbytes,
                    n_tokens,
                )
                
                # Send response to LoadGen
                lg.QuerySamplesComplete([response])
                
                # Log latency metrics if available
                if 'first_token_time' in result:
                    first_token_latency = result['first_token_time'] - start_time
                    logger.debug(f"Query {query_info.query_id}: First token latency: {first_token_latency:.3f}s")
                    
            else:
                # Send empty response
                empty_array = np.array([], dtype=np.int32)
                response = lg.QuerySampleResponse(
                    query_info.query_id, 
                    empty_array.ctypes.data if empty_array.size > 0 else 0,
                    empty_array.nbytes
                )
                lg.QuerySamplesComplete([response])
                
        except Exception as e:
            logger.error(f"Error processing query {query_info.query_id}: {e}")
            # Send empty response for failed query
            empty_array = np.array([], dtype=np.int32)
            response = lg.QuerySampleResponse(
                query_info.query_id, 
                empty_array.ctypes.data if empty_array.size > 0 else 0,
                empty_array.nbytes
            )
            lg.QuerySamplesComplete([response])
            
        finally:
            # Mark query as completed
            query_info.completed = True
            
    def _run_event_loop(self):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        
        # Run until stopped
        self.loop.run_until_complete(self._run_until_stopped())
        
    async def _run_until_stopped(self):
        """Run until stop event is set."""
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)
            
    def start(self) -> lg.ConstructSUT:
        """Start the SUT and async event loop."""
        # Create new event loop
        self.loop = asyncio.new_event_loop()
        self.stop_event.clear()
        
        # Start event loop thread
        self.loop_thread = threading.Thread(target=self._run_event_loop)
        self.loop_thread.start()
        
        # Call parent start
        return super().start()
        
    def stop(self) -> None:
        """Stop the SUT and clean up."""
        # Signal stop
        self.stop_event.set()
        
        # Stop event loop
        if self.loop and self.loop_thread:
            # Wait for thread to finish
            self.loop_thread.join(timeout=5.0)
            
            # Close loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop.close()
            self.loop = None
            
        # Clear query table
        with self.query_table_lock:
            self.query_table.clear()
            
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
        
        # Create index to query info mapping
        index_to_query = {}
        with self.query_table_lock:
            for query_id, query_info in self.query_table.items():
                index_to_query[query_info.index] = query_info
        
        # Process results in order of dataset indices
        for i in range(len(self.dataset)):
            query_info = index_to_query.get(i)
            
            if query_info and query_info.result and 'tokens' in query_info.result:
                result = query_info.result
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
                # No result for this index
                ordered_results.append({
                    'output_text': '',
                    'output_tok': [],
                    'output_tok_len': 0
                })
        
        return ordered_results 