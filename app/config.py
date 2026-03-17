from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    app_title: str
    app_host: str
    app_port: int
    data_dir: Path
    db_path: Path
    upload_dir: Path
    manifest_dir: Path
    result_dir: Path
    export_dir: Path
    log_dir: Path
    openai_api_key: str | None
    openai_model: str
    auto_approve_threshold: float
    rscript_bin: str
    runner_script: Path
    max_workers: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")

        data_dir = DATA_DIR
        return cls(
            app_title=os.getenv("APP_TITLE", "KM SurvdigitizeR Batch Runner"),
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=int(os.getenv("APP_PORT", "8000")),
            data_dir=data_dir,
            db_path=data_dir / "km_survdigitizer.sqlite3",
            upload_dir=data_dir / "uploads",
            manifest_dir=data_dir / "manifests",
            result_dir=data_dir / "results",
            export_dir=data_dir / "exports",
            log_dir=data_dir / "logs",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            auto_approve_threshold=float(os.getenv("AUTO_APPROVE_THRESHOLD", "0.82")),
            rscript_bin=os.getenv("RSCRIPT_BIN", "Rscript"),
            runner_script=BASE_DIR / "scripts" / "run_survdigitizer.R",
            max_workers=int(os.getenv("MAX_WORKERS", "2")),
        )

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.upload_dir,
            self.manifest_dir,
            self.result_dir,
            self.export_dir,
            self.log_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()
settings.ensure_directories()
