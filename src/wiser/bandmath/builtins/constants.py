import os

MAX_RAM_BYTES = 4000000000
SCALAR_BYTES = 4
TEMP_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
RATIO_OF_MEM_TO_USE = 0.25

NUM_READERS = 4
NUM_PROCESSORS = 4
NUM_WRITERS = 1

LHS_KEY = 'lhs'
RHS_KEY = 'rhs'

DEFAULT_IGNORE_VALUE = -9999