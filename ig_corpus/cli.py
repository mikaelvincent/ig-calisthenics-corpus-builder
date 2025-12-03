from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import load_config, resolve_runtime_secrets
from .dry_run import run_dry_run
from .errors import ApifyError, ConfigError, LLMError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ig_corpus")

    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser(
        "dry-run",
        help="Verify scraping + LLM labeling on a small sample.",
    )
    dry.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    dry.set_defaults(_handler=_cmd_dry_run)

    run = subparsers.add_parser(
        "run",
        help="Run the full corpus build loop (not implemented yet).",
    )
    run.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    run.add_argument(
        "--out",
        required=True,
        help="Output directory for exports.",
    )
    run.set_defaults(_handler=_cmd_run)

    return parser


def _eprint(message: str) -> None:
    print(message, file=sys.stderr)


def _cmd_dry_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    secrets = resolve_runtime_secrets(cfg)

    result = run_dry_run(cfg, secrets)

    print(f"scraped_count={result.scraped_count}")
    print(f"processed_count={result.processed_count}")
    print(f"eligible_count={result.eligible_count}")
    print(f"query_term={result.query_term}")
    print("example_decision=")
    print(json.dumps(result.example_decision, indent=2, ensure_ascii=False, sort_keys=True))

    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _eprint("The 'run' command is not implemented yet.")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        handler = getattr(args, "_handler")
        return int(handler(args))
    except ConfigError as e:
        _eprint(str(e))
        return 2
    except (ApifyError, LLMError) as e:
        _eprint(str(e))
        return 3
    except KeyboardInterrupt:
        _eprint("Interrupted")
        return 130
    except Exception as e:
        _eprint(f"Unexpected error: {e}")
        return 1
