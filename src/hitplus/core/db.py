from hitplus.core.config import HitPlusConfig


class DatabaseConnection:
    def __init__(self, config: HitPlusConfig) -> None:
        self._config = config
        self._connection = None

    def __enter__(self) -> "DatabaseConnection":
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        raise NotImplementedError

    def query(self, sql: str) -> object:
        """Execute SQL and return a polars DataFrame."""
        raise NotImplementedError
