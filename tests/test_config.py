from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ig_corpus.config import load_config, resolve_runtime_secrets
from ig_corpus.errors import ConfigError


_VALID_YAML = """\
targets:
  final_n: 5
  pool_n: 6
  sampling_seed: 1

apify:
  token_env: APIFY_TOKEN
  primary_actor: apify/instagram-hashtag-scraper
  fallback_actor: apify/instagram-scraper
  results_type: posts
  results_limit_per_query: 10
  keyword_search: true
  run_batch_queries: 1

openai:
  api_key_env: OPENAI_API_KEY
  model_primary: gpt-5-nano
  model_escalation: gpt-5-mini
  escalation_confidence_threshold: 0.70
  max_output_tokens: 16000

filters:
  min_caption_chars: 40
  max_posts_per_user: 0
  allow_reels: true
  reject_if_sponsored_true: false

loop:
  max_iterations: 1
  stagnation_window: 1
  stagnation_min_new_eligible: 0
  max_raw_items: 10
  backoff_seconds: 0

querying:
  seed_terms:
    - calisthenics
  expansion:
    enabled: true
    max_new_terms_per_iter: 1
    min_hashtag_freq_in_eligible: 1
    blocklist_terms: []
"""


class TestConfig(unittest.TestCase):
    def test_load_config_ok(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.yaml"
            path.write_text(_VALID_YAML, encoding="utf-8")

            cfg = load_config(path)
            self.assertEqual(cfg.targets.final_n, 5)
            self.assertEqual(cfg.targets.pool_n, 6)
            self.assertEqual(cfg.apify.results_type, "posts")

    def test_load_config_rejects_bad_pool(self) -> None:
        bad_yaml = _VALID_YAML.replace("pool_n: 6", "pool_n: 4")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.yaml"
            path.write_text(bad_yaml, encoding="utf-8")

            with self.assertRaises(ConfigError):
                load_config(path)

    def test_resolve_runtime_secrets_requires_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.yaml"
            path.write_text(_VALID_YAML, encoding="utf-8")
            cfg = load_config(path)

            with self.assertRaises(ConfigError):
                resolve_runtime_secrets(cfg, environ={})

            secrets = resolve_runtime_secrets(
                cfg, environ={"APIFY_TOKEN": "a", "OPENAI_API_KEY": "b"}
            )
            self.assertEqual(secrets.apify_token, "a")
            self.assertEqual(secrets.openai_api_key, "b")


if __name__ == "__main__":
    unittest.main()
