from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TargetData:
    contours: dict
    metadata: dict = field(default_factory=dict)
    primary_contour_key: str = "primary"

    @property
    def primary_contour(self):
        return self.contours[self.primary_contour_key]


@dataclass
class ParsedResult:
    contours: dict
    features: dict
    artifacts: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    primary_contour_key: str = "primary"

    @property
    def primary_contour(self):
        return self.contours[self.primary_contour_key]


@dataclass
class ObjectiveResult:
    name: str
    value: float
    metadata: dict = field(default_factory=dict)
