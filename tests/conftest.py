# -*- coding: utf-8 -*-
"""Pytest configuration for xdispersion tests.

Ensures the ``xdispersion`` package is importable regardless of the
directory from which pytest is invoked.
"""
import os
import sys

# Insert the project root (parent of tests/) at the front of sys.path
# so that ``import xdispersion`` works when running pytest directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
