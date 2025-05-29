"""
Utility functions for MLPerf backends.
"""

import os
from pathlib import Path


def get_cache_directory() -> Path:
    """
    Determine the cache directory with the following preference order:
    1. /raid/data/$USER/.cache (if available)
    2. /home/$USER/.cache (fallback)
    
    Returns:
        Path: The determined cache directory path
    """
    # Get the current user
    user = os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))
    
    # First preference: /raid/data/$USER/.cache
    raid_cache = Path(f'/raid/data/{user}/.cache')
    
    # Check if /raid/data exists and is accessible
    if raid_cache.parent.parent.exists() and os.access(raid_cache.parent.parent, os.W_OK):
        # Create the cache directory if it doesn't exist
        raid_cache.mkdir(parents=True, exist_ok=True)
        return raid_cache
    
    # Fallback: /home/$USER/.cache
    home_cache = Path(f'/home/{user}/.cache')
    
    # If /home/$USER doesn't exist, use $HOME environment variable
    if not home_cache.parent.exists():
        home = os.environ.get('HOME', '.')
        home_cache = Path(home) / '.cache'
    
    # Create the cache directory if it doesn't exist
    home_cache.mkdir(parents=True, exist_ok=True)
    
    return home_cache


def setup_huggingface_cache() -> Path:
    """
    Set up HuggingFace cache environment variables using the preferred cache directory.
    
    Returns:
        Path: The cache directory being used
    """
    cache_dir = get_cache_directory()
    
    # Set HuggingFace cache environment variables
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['HF_HUB_CACHE'] = str(cache_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
    
    return cache_dir 