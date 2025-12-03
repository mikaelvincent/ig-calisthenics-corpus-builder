from __future__ import annotations

from .cli import main


def _entrypoint() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
