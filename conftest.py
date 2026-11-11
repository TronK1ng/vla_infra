"""Repository-level pytest bootstrap.

The package is laid out under ``src/``.  The project is normally installed with
``pip install -e .``; the ``sys.path`` insertion below keeps ``python -m pytest``
working in the offline station image where an editable install has not been
performed.  It is intentionally tiny and has no side effects beyond import
path setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
