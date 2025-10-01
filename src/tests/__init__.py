import multiprocessing as mp

print(f"!@!#!# mp_start method: {mp.get_start_method()}")
mp.set_start_method('spawn')