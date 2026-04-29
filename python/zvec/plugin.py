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
"""Optional DiskAnn plugin loader.

The DiskAnn index is shipped as a separate shared library
(``libzvec_diskann_plugin.so``) so that the core ``zvec`` package does not
hard-depend on libaio. Import this module and call
:func:`load_diskann_plugin` — typically after :func:`is_libaio_available`
returns ``True`` — before creating a DiskAnn index.

Example:
    >>> import zvec
    >>> if zvec.is_libaio_available():
    ...     status = zvec.load_diskann_plugin()
    ...     assert status == zvec.DiskAnnPluginStatus.OK
"""

from __future__ import annotations

from typing import Optional

from _zvec import DiskAnnPluginStatus as _DiskAnnPluginStatus
from _zvec import is_diskann_plugin_loaded as _is_loaded
from _zvec import is_libaio_available as _is_libaio
from _zvec import load_diskann_plugin as _load
from _zvec import unload_diskann_plugin as _unload

__all__ = [
    "DiskAnnPluginStatus",
    "is_diskann_plugin_loaded",
    "is_libaio_available",
    "load_diskann_plugin",
    "unload_diskann_plugin",
]

# Re-export the enum under a Pythonic name.
DiskAnnPluginStatus = _DiskAnnPluginStatus


def is_libaio_available() -> bool:
    """Return ``True`` if ``libaio`` is present on the host and exposes the
    symbols required by the DiskAnn plugin.

    The probe is non-destructive and safe to call multiple times.
    """
    return bool(_is_libaio())


def load_diskann_plugin(path: Optional[str] = None) -> DiskAnnPluginStatus:
    """Load the DiskAnn plugin shared library.

    Args:
        path: Optional explicit path to ``libzvec_diskann_plugin.so``.
            When ``None`` or empty the plugin is searched next to the running
            executable first, then on the platform default dynamic-linker path.

    Returns:
        A :class:`DiskAnnPluginStatus` value. ``OK`` is also returned when the
        plugin is already loaded (idempotent).
    """
    return DiskAnnPluginStatus(_load(path or ""))


def is_diskann_plugin_loaded() -> bool:
    """Return ``True`` if the DiskAnn plugin is currently loaded."""
    return bool(_is_loaded())


def unload_diskann_plugin() -> bool:
    """Unload the DiskAnn plugin.

    Caller must guarantee that no DiskAnn objects are still alive and no
    background threads are executing DiskAnn code before calling this.

    Returns:
        ``True`` if a live handle was released, ``False`` otherwise.
    """
    return bool(_unload())
