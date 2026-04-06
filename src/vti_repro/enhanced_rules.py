"""Scaffolding for the paper's enhancement stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnhancedConfig:
    positive_ratio_threshold: float = 1.0
    negative_ratio_threshold: float = 1.0


def enhancement_requires_external_elements() -> str:
    return (
        "The enhancement stage needs code elements extracted from Joern/CPG "
        "such as calls, assignments, control structures, and return statements."
    )
