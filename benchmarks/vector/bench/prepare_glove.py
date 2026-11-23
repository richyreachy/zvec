#!/usr/bin/env python3
# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
