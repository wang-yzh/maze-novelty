from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents import QAgent


@dataclass
class ArtifactSummary:
    method: str
    source_env: str
    seed: int
    population_size: int
    hall_of_fame_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PretrainArtifact:
    method: str
    source_env: str
    seed: int
    population: list[QAgent]
    hall_of_fame: list[QAgent]
    metadata: dict[str, Any] = field(default_factory=dict)

    def best_agent(self) -> QAgent:
        if self.hall_of_fame:
            return self.hall_of_fame[0].clone()
        if self.population:
            return self.population[0].clone()
        raise ValueError("pretrain artifact has no agents")

    def summary(self) -> ArtifactSummary:
        return ArtifactSummary(
            method=self.method,
            source_env=self.source_env,
            seed=self.seed,
            population_size=len(self.population),
            hall_of_fame_size=len(self.hall_of_fame),
            metadata=dict(self.metadata),
        )
