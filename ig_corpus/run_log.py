from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, TextIO


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, *, limit: int) -> str:
    s = str(text or "")
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


class RunLogger:
    """
    Tiny JSONL logger for long-running corpus builds.

    Each log line is a single JSON object, making it easy to parse for audits.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        overwrite: bool = True,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._overwrite = bool(overwrite)
        self._run_id = (run_id or "").strip() or None
        self._session_id = (session_id or "").strip() or uuid.uuid4().hex
        self._fp: TextIO | None = None
        self._lock = Lock()
        self._opened = False

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        overwrite: bool = True,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> "RunLogger":
        logger = cls(path, overwrite=overwrite, run_id=run_id, session_id=session_id)
        logger._ensure_open()
        return logger

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.flush()
                finally:
                    self._fp.close()
                self._fp = None

    def __enter__(self) -> "RunLogger":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def set_run_id(self, run_id: str) -> None:
        rid = (run_id or "").strip()
        if not rid:
            return
        self._run_id = rid

    def info(self, event: str, *, url: str | None = None, **data: Any) -> None:
        self.log("INFO", event, url=url, **data)

    def warning(self, event: str, *, url: str | None = None, **data: Any) -> None:
        self.log("WARN", event, url=url, **data)

    def error(self, event: str, *, url: str | None = None, **data: Any) -> None:
        self.log("ERROR", event, url=url, **data)

    def exception(
        self,
        event: str,
        *,
        exc: BaseException,
        url: str | None = None,
        **data: Any,
    ) -> None:
        err = {
            "type": type(exc).__name__,
            "message": _truncate(str(exc), limit=2000),
            "traceback": _truncate(
                "".join(
                    traceback.format_exception(
                        type(exc), exc, exc.__traceback__
                    )
                ),
                limit=12000,
            ),
        }
        self.log("ERROR", event, url=url, error=err, **data)

    def log(self, level: str, event: str, *, url: str | None = None, **data: Any) -> None:
        lvl = (level or "").strip().upper() or "INFO"
        ev = (event or "").strip() or "event"

        record: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "level": lvl,
            "event": ev,
            "session_id": self._session_id,
        }

        if self._run_id:
            record["run_id"] = self._run_id

        u = (url or "").strip()
        if u:
            record["url"] = u

        if data:
            record["data"] = data

        self._write(record)

    def _ensure_open(self) -> None:
        if self._fp is not None:
            return

        with self._lock:
            if self._fp is not None:
                return

            self._path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if self._overwrite and not self._opened else "a"

            self._fp = self._path.open(mode, encoding="utf-8", newline="\n")
            self._opened = True

    def _write(self, record: dict[str, Any]) -> None:
        self._ensure_open()

        payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

        with self._lock:
            if self._fp is None:
                return
            self._fp.write(payload + "\n")
            self._fp.flush()
