import multiprocessing as mp

try:
    print(f"!@#@# trying to set mp to spawn!!: before: {mp.get_start_method()}")
    mp.set_start_method("spawn")
    print(f"!@#@# successfuly set mp to spawn!! after: {mp.get_start_method()}")
except RuntimeError:
    # context already set (pytest worker may reuse interpreter)
    print(f"!@#@# mp already set!!: {mp.get_start_method()}")
    pass