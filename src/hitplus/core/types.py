from dataclasses import dataclass
from enum import Enum


class StepStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class SplitMode(Enum):
    DEV = "dev"
    FULL = "full"


@dataclass
class StepResult:
    status: StepStatus
    message: str = ""
    duration_seconds: float = 0.0
