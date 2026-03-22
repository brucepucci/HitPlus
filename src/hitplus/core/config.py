from pathlib import Path

from pydantic import BaseModel

from hitplus.core.types import SplitMode


class HitPlusConfig(BaseModel):
    model_name: str = ""
    dev_mode: bool = True
    force: bool = False
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    db_name: str = "mlb_stats.db"

    @property
    def split_mode(self) -> SplitMode:
        return SplitMode.DEV if self.dev_mode else SplitMode.FULL

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name
