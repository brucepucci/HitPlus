from pathlib import Path


class ArtifactStore:
    def __init__(self, base_dir: Path = Path("artifacts")) -> None:
        self._base_dir = base_dir

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def path_for(self, model_name: str, category: str, filename: str) -> Path:
        raise NotImplementedError

    def put(self, model_name: str, category: str, filename: str, data: object) -> Path:
        raise NotImplementedError

    def get(self, model_name: str, category: str, filename: str) -> object:
        raise NotImplementedError

    def exists(self, model_name: str, category: str, filename: str) -> bool:
        raise NotImplementedError

    def is_fresh(
        self,
        model_name: str,
        category: str,
        filename: str,
        *,
        relative_to: list[Path],
    ) -> bool:
        raise NotImplementedError
