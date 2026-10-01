#!/usr/bin/env python3

"""
Backward-compatible CLI entrypoint.

The canonical implementation lives in:
    src/possession/controller_validation.py
"""

from possession.controller_validation import main


if __name__ == "__main__":
    main()