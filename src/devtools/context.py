# Per the recommendations at this page, this helper module updates the Python
# path to include the sources to be tested.
#
# https://docs.python-guide.org/writing/structure/

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))