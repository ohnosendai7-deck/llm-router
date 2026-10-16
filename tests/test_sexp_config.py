"""Unit tests for the S-expression and JSON configuration parser."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest

from llm_router.config import (
    SCAFFOLD_CONFIG,
    ensure_scaffold_config,
    load_config,
    parse_config_sexp,
)


class TestSexpConfig(unittest.TestCase):
    def test_scaffold_config_parses_as_disabled(self) -> None:
        cfg = parse_config_sexp(SCAFFOLD_CONFIG)
        self.assertTrue(cfg.disabled)
        self.assertEqual(cfg.port, 8402)
        self.assertEqual(len(cfg.providers), 2)
        kimi = cfg.get_provider("kimi")
        self.assertIsNotNone(kimi)
        assert kimi is not None
        self.assertEqual(kimi.upstream, "api.moonshot.ai")
        anth = cfg.get_provider("anthropic")
        self.assertIsNotNone(anth)
        assert anth is not None
        self.assertTrue(anth.mock)
        self.assertEqual(anth.protocol, "anthropic")

    def test_managed_providers_and_complex_overrides(self) -> None:
        sexp = """
        ;; Production-style config with overrides, vectors, alists, booleans, null, and fanout
        (api-proxy
         (disabled #f)
         (port 18402)
         (providers
          (openrouter (upstream "openrouter.ai")
                      (path "/api")
                      (key "sk-or-real")
                      (models
                       (argon (overrides (model . "google/gemini-4-argon")
                                         (include_reasoning . #t)
                                         (stop . #("END" "STOP"))
                                         (safety . 'null)
                                         (reasoning . ((effort . "high")
                                                       (max_tokens . 4096)))))))
          (kimi-coding (upstream "api.kimi.com")
                       (path "/coding")
                       (key "sk-kimi-real")
                       (models
                        (fast (overrides (model . "kimi-k2-turbo")))
                        (max  (overrides (model . "k3") (thinking . "max")))
                        (glm  (overrides (model . "glm-5"))
                              (upstream "api.z.ai")
                              (path "/anthropic")
                              (key "sk-glm-real"))))
          (kimi/max (upstream "api.moonshot.ai")
                    (key "sk-slash"))))
        """
        cfg = parse_config_sexp(sexp)
        self.assertFalse(cfg.disabled)
        self.assertEqual(cfg.port, 18402)
        self.assertFalse(cfg.has_live_anthropic_provider())

        or_prov = cfg.get_provider("openrouter")
        assert or_prov is not None
        self.assertEqual(or_prov.path, "/api")
        argon = or_prov.find_model("argon")
        assert argon is not None
        self.assertEqual(argon.target_model, "google/gemini-4-argon")
        self.assertEqual(argon.overrides["include_reasoning"], True)
        self.assertEqual(argon.overrides["stop"], ["END", "STOP"])
        self.assertIsNone(argon.overrides["safety"])
        self.assertEqual(argon.overrides["reasoning"], {"effort": "high", "max_tokens": 4096})

        kc = cfg.get_provider("kimi-coding")
        assert kc is not None
        glm = kc.find_model("glm")
        assert glm is not None
        self.assertEqual(glm.upstream, "api.z.ai")
        self.assertEqual(glm.path, "/anthropic")
        self.assertEqual(glm.key, "sk-glm-real")

        slash = cfg.get_provider("kimi/max")
        self.assertIsNotNone(slash)

    def test_validation_errors(self) -> None:
        # Missing overrides in model
        with self.assertRaises(ValueError):
            parse_config_sexp(
                "(api-proxy (disabled #f) (providers (p (upstream \"h\") (key \"k\") (models (m (path \"/x\"))))))"
            )
        # Model upstream without key
        with self.assertRaises(ValueError):
            parse_config_sexp(
                "(api-proxy (disabled #f) (providers (p (upstream \"h\") (key \"k\") (models (m (overrides (model . \"x\")) (upstream \"h2\"))))))"
            )
        # Enabled with no providers
        with self.assertRaises(ValueError):
            parse_config_sexp("(api-proxy (disabled #f) (providers))")

    def test_ensure_scaffold_config_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "sub", "config.scm")
            created = ensure_scaffold_config(cfg_path)
            self.assertTrue(created)
            self.assertFalse(ensure_scaffold_config(cfg_path))
            mode = stat.S_IMODE(os.stat(cfg_path).st_mode)
            self.assertEqual(mode, 0o600)
            loaded = load_config(cfg_path)
            self.assertTrue(loaded.disabled)


if __name__ == "__main__":
    unittest.main()
