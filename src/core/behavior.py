from __future__ import annotations

from dataclasses import dataclass, field

from minigrid_encoders import StateSignature


@dataclass(frozen=True)
class BehaviorPrior:
    kind: str
    initiation: StateSignature
    action_trace: tuple[int, ...]
    termination: StateSignature | None
    score: float
    source: str = "unknown"
    support: int = 1
    state_trace: tuple[int, ...] = ()

    def matches(self, signature: StateSignature) -> bool:
        return self.initiation.matches(signature)


@dataclass
class BehaviorLibrary:
    max_items: int = 64
    priors: list[BehaviorPrior] = field(default_factory=list)

    def add(self, prior: BehaviorPrior) -> None:
        key = (prior.kind, prior.initiation, prior.action_trace, prior.termination)
        for idx, existing in enumerate(self.priors):
            existing_key = (existing.kind, existing.initiation, existing.action_trace, existing.termination)
            if existing_key != key:
                continue
            support = existing.support + prior.support
            score = (
                existing.score * existing.support + prior.score * prior.support
            ) / max(1, support)
            representative = prior if prior.score >= existing.score else existing
            self.priors[idx] = BehaviorPrior(
                kind=existing.kind,
                initiation=existing.initiation,
                action_trace=existing.action_trace,
                termination=existing.termination,
                score=score,
                source=existing.source,
                support=support,
                state_trace=representative.state_trace,
            )
            self._trim()
            return

        self.priors.append(prior)
        self._trim()

    def best_match(self, signature: StateSignature) -> BehaviorPrior | None:
        matches = [prior for prior in self.priors if prior.matches(signature)]
        if not matches:
            return None
        return max(matches, key=lambda prior: (prior.score, prior.support, -len(prior.action_trace)))

    def _trim(self) -> None:
        self.priors.sort(key=lambda prior: (prior.score, prior.support, -len(prior.action_trace)), reverse=True)
        if len(self.priors) > self.max_items:
            del self.priors[self.max_items :]

    def __len__(self) -> int:
        return len(self.priors)
