from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

def monitor_threads(interval=1.0):
    """
    Periodically print a list of all active threads.
    Args:
    - interval (float): The time interval (in seconds) to wait between each print.
    """
    while True:
        # Get a list of all currently running threads
        active_threads = threading.enumerate()

        # Print out the name and status of each thread
        print(f"\n[Thread Monitor] Active Threads ({len(active_threads)} total):")
        for thread in active_threads:
            print(f"  - Name: {thread.name}, Is Alive: {thread.is_alive()}")

        # Wait for the specified interval before printing again
        time.sleep(interval)

# Start the monitoring function in a separate daemon thread
monitor_thread = threading.Thread(target=monitor_threads, name="Thread-Monitor", daemon=True)
monitor_thread.start()