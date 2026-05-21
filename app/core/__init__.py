"""
Core runtime configuration.
"""

import os
import platform


if platform.system() == "Darwin":
    # Current macOS torch and faiss wheels can each load an OpenMP runtime.
    # Without this guard, the process aborts when both are used during search.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
