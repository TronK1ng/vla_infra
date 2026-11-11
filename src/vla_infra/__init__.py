"""VLA infrastructure for the Flexiv Rizon 4s station.

Milestone M1 (this tree): contracts, fake devices, synchronisation, a dummy
OpenPI-shaped policy loop and the supervisor/HOLD state machine.  No module in
this package talks to real hardware unless a real backend is explicitly built
and enabled in a later milestone.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
