from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RunStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def ensure_run_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, run_id: str, name: str, value: Any) -> Path:
        path = self.ensure_run_dir(run_id) / name
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        text = json.dumps(value, ensure_ascii=True, indent=2, default=str)
        path.write_text(text + "\n", encoding="utf-8")
        return path

    def write_text(self, run_id: str, name: str, text: str) -> Path:
        path = self.ensure_run_dir(run_id) / name
        path.write_text(text, encoding="utf-8")
        return path

    def save_checkpoint(self, run_id: str, state: dict[str, Any]) -> Path:
        return self.write_json(run_id, "checkpoint.json", state)

    def load_checkpoint(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "checkpoint.json"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found for run_id={run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def clear_checkpoints(self) -> int:
        count = 0
        if not self.base_dir.exists():
            return 0
        for checkpoint in self.base_dir.glob("*/checkpoint.json"):
            checkpoint.unlink()
            count += 1
        return count

    def remove_run(self, run_id: str) -> None:
        path = self.run_dir(run_id)
        if path.exists():
            shutil.rmtree(path)
