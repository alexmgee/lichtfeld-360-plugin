"""Root conftest — DLL directory setup for Windows test runs.

The plugin's __init__.py adds lib/ to the DLL search path (for python3.dll
and _C.pyd needed by OpenCV and SAM). Pytest doesn't go through __init__.py,
so we replicate the setup here.
"""
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    _lib_dir = Path(__file__).resolve().parent / "lib"
    if _lib_dir.is_dir():
        os.add_dll_directory(str(_lib_dir))
