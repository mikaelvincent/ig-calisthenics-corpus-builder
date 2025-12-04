from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import load_config, resolve_runtime_secrets
from .dry_run import run_dry_run
from .errors import ApifyError, ConfigError, ExportError, LLMError, StorageError
from .export_excel import export_corpus_workbook
from .export_pdf import export_codebook_pdf
from .loop import run_feedback_loop
from .run_log import RunLogger
from .storage import SQLiteStateStore


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
    dry.add_argument(
        "--offline",
        action="store_true",
        help="Run without network calls using a small stub dataset.",
    )
    dry.set_defaults(_handler=_cmd_dry_run)

    run = subparsers.add_parser(
        "run",
        help="Run the corpus build feedback loop until the pool target is met.",
    )
    run.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    run.add_argument(
        "--out",
        required=True,
        help="Output directory for state and logs.",
    )
    run.set_defaults(_handler=_cmd_run)

    return parser


def _eprint(message: str) -> None:
    print(message, file=sys.stderr)


def _cmd_dry_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    secrets = resolve_runtime_secrets(cfg)

    if bool(getattr(args, "offline", False)):
        from .offline import OfflineInstagramHashtagScraper, OfflinePostClassifier

        result = run_dry_run(
            cfg,
            secrets,
            scraper=OfflineInstagramHashtagScraper(),
            classifier=OfflinePostClassifier(),
        )
    else:
        result = run_dry_run(cfg, secrets)

    print(f"scraped_count={result.scraped_count}")
    print(f"processed_count={result.processed_count}")
    print(f"eligible_count={result.eligible_count}")
    print(f"query_term={result.query_term}")
    print("example_decision=")
    print(json.dumps(result.example_decision, indent=2, ensure_ascii=False, sort_keys=True))

    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    with RunLogger.open(log_path, overwrite=True) as log:
        log.info(
            "run_command_started",
            config_path=str(args.config),
            out_dir=str(out_dir),
        )

        try:
            cfg = load_config(args.config)
            secrets = resolve_runtime_secrets(cfg)

            log.info(
                "config_loaded",
                config_path=str(args.config),
                apify_token_env=cfg.apify.token_env,
                openai_api_key_env=cfg.openai.api_key_env,
            )

            db_path = out_dir / "state.sqlite"
            xlsx_path = out_dir / "corpus.xlsx"
            pdf_path = out_dir / "codebook.pdf"

            with SQLiteStateStore.open(db_path) as store:
                result = run_feedback_loop(cfg, secrets, store=store, logger=log)

                log.info(
                    "feedback_loop_completed",
                    status=result.status,
                    run_id=result.run_id,
                    iterations=result.iterations,
                    raw_posts=result.raw_posts,
                    decisions=result.decisions,
                    eligible=result.eligible,
                )

                log.info("export_excel_started", path=str(xlsx_path))
                export_corpus_workbook(cfg, store, xlsx_path, run_id=result.run_id)
                log.info("export_excel_completed", path=str(xlsx_path))

                log.info("export_pdf_started", path=str(pdf_path))
                export_codebook_pdf(cfg, store, pdf_path, run_id=result.run_id)
                log.info("export_pdf_completed", path=str(pdf_path))

            print(f"status={result.status}")
            print(f"run_id={result.run_id}")
            print(f"iterations={result.iterations}")
            print(f"raw_posts={result.raw_posts}")
            print(f"decisions={result.decisions}")
            print(f"eligible={result.eligible}")
            print(f"corpus_xlsx={xlsx_path}")
            print(f"codebook_pdf={pdf_path}")
            print(f"run_log={log_path}")

            return 0 if result.status == "completed_pool" else 4
        except Exception as e:
            log.exception("run_command_failed", exc=e)
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        handler = getattr(args, "_handler")
        return int(handler(args))
    except ConfigError as e:
        _eprint(str(e))
        return 2
    except (ApifyError, LLMError, StorageError, ExportError) as e:
        _eprint(str(e))
        return 3
    except KeyboardInterrupt:
        _eprint("Interrupted")
        return 130
    except Exception as e:
        _eprint(f"Unexpected error: {e}")
        return 1
