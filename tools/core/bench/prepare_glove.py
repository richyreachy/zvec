#!/usr/bin/env python3
# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0.
"""Compatibility entry point for exporting GloVe-100 HDF5 inputs."""

from __future__ import annotations

if __package__:
    from .prepare_hdf5 import export
    from .prepare_hdf5 import main as _main
else:
    from prepare_hdf5 import export
    from prepare_hdf5 import main as _main

__all__ = ["export", "main"]


def main():
    _main(expected_dimension=100)


if __name__ == "__main__":
    main()
