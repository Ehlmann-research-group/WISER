"""
Singleton multiprocessing context for WISER.

This module provides a consistent 'spawn' context for all multiprocessing
operations across the application. Using 'spawn' ensures cross-platform
compatibility and avoids issues with GUI frameworks in child processes.

Usage:
    Instead of:
        import multiprocessing as mp
        pool = mp.Pool()
    
    Use:
        from wiser.utils.multiprocessing_context import CTX
        pool = CTX.Pool()
"""

import multiprocessing as mp

# Create singleton spawn context
CTX = mp.get_context('spawn')
