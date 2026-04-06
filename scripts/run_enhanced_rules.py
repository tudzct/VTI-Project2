#!/usr/bin/env python3
"""Placeholder entrypoint for the enhancement stage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.enhanced_rules import enhancement_requires_external_elements


def main() -> int:
    raise NotImplementedError(enhancement_requires_external_elements())


if __name__ == "__main__":
    raise SystemExit(main())
