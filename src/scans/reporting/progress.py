from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def append_status(status_path: str | Path | None, event: str, **payload: Any) -> None:
    if status_path is None:
        return
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
