from __future__ import annotations

from typing import Protocol

import numpy as np


class MethodRunner(Protocol):
    def __call__(self, args, rng: np.random.Generator) -> list[dict]:
        ...
