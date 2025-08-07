import time

from typing import TYPE_CHECKING, Dict, List, Tuple
from concurrent.futures import as_completed, ProcessPoolExecutor

def submit_tasks_mimic(process_pool_executor: ProcessPoolExecutor, func, unique_files: List[str]) -> None:
    for path in unique_files:
        future_to_path = {
            process_pool_executor.submit(func, path): path
            for path in unique_files
            }
    for future in as_completed(future_to_path):
        path = future_to_path[future]
        try:
            future.result()
            print(f"Processed {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

def process_mimic(path):
    sleep_time = 10
    print(f"path: {path}")
    print(f"About to sleep for {sleep_time} seconds!")
    time.sleep(sleep_time)
    print(f"Sleeping finished!")