from abc import ABC, abstractmethod

from hitplus.core.artifact import ArtifactStore
from hitplus.core.config import HitPlusConfig
from hitplus.core.types import StepResult, StepStatus


class PipelineStep(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def input_artifact_keys(self) -> list[str]: ...

    @abstractmethod
    def output_artifact_keys(self) -> list[str]: ...

    @abstractmethod
    def run(self, config: HitPlusConfig, store: ArtifactStore) -> StepResult: ...


class Pipeline:
    def __init__(self, steps: list[PipelineStep]) -> None:
        self._steps = steps
        self._validate_dag()

    def _validate_dag(self) -> None:
        """Ensure every step's inputs are provided by prior steps' outputs."""
        available: set[str] = set()
        for step in self._steps:
            missing = set(step.input_artifact_keys()) - available
            if missing:
                raise ValueError(
                    f"Step '{step.name}' requires artifacts {missing} "
                    f"not produced by prior steps"
                )
            available.update(step.output_artifact_keys())

    def run(
        self,
        config: HitPlusConfig,
        store: ArtifactStore,
        *,
        until: str | None = None,
    ) -> list[StepResult]:
        results: list[StepResult] = []
        for step in self._steps:
            result = step.run(config, store)
            results.append(result)
            if result.status == StepStatus.FAILED:
                break
            if until and step.name == until:
                break
        return results
