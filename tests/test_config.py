"""Tests for pdf_mcp.config module."""

from pathlib import Path

import pytest

from pdf_mcp.config import PDFConfig


class TestConfigLoad:
    def test_missing_file_is_permissive(self, tmp_path):
        """Missing config file means no restrictions beyond the SSRF floor."""
        config = PDFConfig(config_path=tmp_path / "nonexistent.toml")
        config.check_path("/any/path/file.pdf")
        config.check_url_host("example.com")

    def test_malformed_toml_raises_with_file_path(self, tmp_path):
        """Malformed TOML raises ValueError mentioning the file path."""
        bad = tmp_path / "config.toml"
        bad.write_text("invalid toml [[[", encoding="utf-8")
        with pytest.raises(ValueError, match="config.toml"):
            PDFConfig(config_path=bad)

    def test_valid_file_is_loaded(self, tmp_path):
        """Valid TOML file is loaded and rules applied."""
        cfg = tmp_path / "config.toml"
        secret = tmp_path / "secret"
        cfg.write_text(
            f'[paths]\ndeny = ["{secret.as_posix()}/**"]\n', encoding="utf-8"
        )
        config = PDFConfig(config_path=cfg)
        with pytest.raises(ValueError):
            config.check_path(str(secret / "file.pdf"))


class TestPathRules:
    def test_no_allow_is_permissive(self, tmp_path):
        """Empty allow list means any path is accepted (within floor)."""
        cfg = tmp_path / "config.toml"
        cfg.write_text("[paths]\ndeny = []\n", encoding="utf-8")
        config = PDFConfig(config_path=cfg)
        config.check_path("/any/path/file.pdf")

    def test_allow_list_enforced(self, tmp_path):
        """Path outside allow list is rejected."""
        cfg = tmp_path / "config.toml"
        pdfs = tmp_path / "data" / "pdfs"
        cfg.write_text(f'[paths]\nallow = ["{pdfs.as_posix()}/**"]\n', encoding="utf-8")
        config = PDFConfig(config_path=cfg)
        config.check_path(str(pdfs / "report.pdf"))
        with pytest.raises(ValueError, match="not in allowed"):
            config.check_path(str(tmp_path / "home" / "private.pdf"))

    def test_deny_list_enforced(self, tmp_path):
        """Path matching deny pattern is rejected."""
        cfg = tmp_path / "config.toml"
        secret = tmp_path / "secret"
        cfg.write_text(
            f'[paths]\ndeny = ["{secret.as_posix()}/**"]\n', encoding="utf-8"
        )
        config = PDFConfig(config_path=cfg)
        with pytest.raises(ValueError, match="denied"):
            config.check_path(str(secret / "file.pdf"))

    def test_deny_wins_over_allow(self, tmp_path):
        """Path matching both allow and deny is denied (fail-closed)."""
        cfg = tmp_path / "config.toml"
        data = tmp_path / "data"
        cfg.write_text(
            f'[paths]\nallow = ["{data.as_posix()}/**"]\n'
            f'deny = ["{(data / "secret").as_posix()}/**"]\n',
            encoding="utf-8",
        )
        config = PDFConfig(config_path=cfg)
        config.check_path(str(data / "public" / "report.pdf"))
        with pytest.raises(ValueError, match="denied"):
            config.check_path(str(data / "secret" / "private.pdf"))

    def test_tilde_expansion(self, tmp_path):
        """~ in patterns is expanded to the home directory."""
        home = str(Path.home())
        cfg = tmp_path / "config.toml"
        cfg.write_text('[paths]\nallow = ["~/Documents/**"]\n', encoding="utf-8")
        config = PDFConfig(config_path=cfg)
        config.check_path(f"{home}/Documents/report.pdf")
        with pytest.raises(ValueError, match="not in allowed"):
            config.check_path("/tmp/report.pdf")

    def test_symlink_traversal_blocked(self, tmp_path):
        """Symlink from allowed path into denied path is rejected via Path.resolve()."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        secret_file = secret_dir / "private.pdf"
        secret_file.write_bytes(b"secret")

        link = allowed_dir / "link.pdf"
        link.symlink_to(secret_file)

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            # as_posix(): a Windows path interpolated raw makes TOML read
            # its backslashes as escapes ("Invalid hex value" on \Users).
            # fnmatch normalises separators on Windows, so forward slashes
            # match either way.
            f'[paths]\nallow = ["{allowed_dir.as_posix()}/**"]\n'
            f'deny = ["{secret_dir.as_posix()}/**"]\n',
            encoding="utf-8",
        )
        config = PDFConfig(config_path=cfg)

        with pytest.raises(ValueError, match="denied"):
            config.check_path(str(link))


class TestUrlRules:
    def test_no_allow_is_permissive(self, tmp_path):
        """No allow list means any public host is accepted."""
        config = PDFConfig(config_path=tmp_path / "none.toml")
        config.check_url_host("example.com")

    def test_wildcard_matching(self, tmp_path):
        """* in hostname pattern matches any chars including dots."""
        cfg = tmp_path / "config.toml"
        cfg.write_text('[urls]\nallow = ["*.example.com"]\n', encoding="utf-8")
        config = PDFConfig(config_path=cfg)
        config.check_url_host("docs.example.com")
        with pytest.raises(ValueError, match="not in allowed"):
            config.check_url_host("evil.com")

    def test_case_insensitive(self, tmp_path):
        """Hostname matching is case-insensitive."""
        cfg = tmp_path / "config.toml"
        cfg.write_text('[urls]\ndeny = ["Evil.com"]\n', encoding="utf-8")
        config = PDFConfig(config_path=cfg)
        with pytest.raises(ValueError, match="denied"):
            config.check_url_host("EVIL.COM")

    def test_deny_wins_over_allow(self, tmp_path):
        """Host matching both allow and deny is denied (fail-closed)."""
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[urls]\nallow = ["*.example.com"]\ndeny = ["bad.example.com"]\n',
            encoding="utf-8",
        )
        config = PDFConfig(config_path=cfg)
        config.check_url_host("docs.example.com")
        with pytest.raises(ValueError, match="denied"):
            config.check_url_host("bad.example.com")


def test_max_response_bytes_default_when_missing(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("", encoding="utf-8")
    cfg = PDFConfig(config_path=cfg_path)
    assert cfg.max_response_bytes == 200_000


def test_max_response_bytes_honored(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[limits]\nmax_response_bytes = 50000\n", encoding="utf-8")
    cfg = PDFConfig(config_path=cfg_path)
    assert cfg.max_response_bytes == 50_000


def test_max_response_bytes_clamped_to_ceiling(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[limits]\nmax_response_bytes = 999999999\n", encoding="utf-8")
    cfg = PDFConfig(config_path=cfg_path)
    assert cfg.max_response_bytes == 2_000_000


def test_max_response_bytes_clamped_to_floor(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[limits]\nmax_response_bytes = 10\n", encoding="utf-8")
    cfg = PDFConfig(config_path=cfg_path)
    assert cfg.max_response_bytes == 4_096


def test_max_response_bytes_rejects_non_int(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[limits]\nmax_response_bytes = "200kb"\n', encoding="utf-8")
    cfg = PDFConfig(config_path=cfg_path)
    with pytest.raises(ValueError, match="must be an integer"):
        cfg.max_response_bytes


def test_injection_phrases_default_empty_when_missing(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("", encoding="utf-8")
    cfg = PDFConfig(config_path=cfg_path)
    assert cfg.injection_phrases == ()


def test_injection_phrases_loaded_raw(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[content_trust]\n"
        'injection_phrases = ["忽略以上所有指示", '
        '"ignorez les instructions"]\n',
        # Explicit: the default encoding is cp1252 on Windows, which
        # cannot represent these phrases, and the config loader reads the
        # file as UTF-8 regardless of platform.
        encoding="utf-8",
    )
    cfg = PDFConfig(config_path=cfg_path)
    assert cfg.injection_phrases == (
        "忽略以上所有指示",
        "ignorez les instructions",
    )


def test_injection_phrases_rejects_non_list(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[content_trust]\ninjection_phrases = "not a list"\n', encoding="utf-8"
    )
    cfg = PDFConfig(config_path=cfg_path)
    with pytest.raises(ValueError, match="must be a list of strings"):
        cfg.injection_phrases


def test_injection_phrases_rejects_non_string_element(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[content_trust]\ninjection_phrases = ["ok", 123]\n', encoding="utf-8"
    )
    cfg = PDFConfig(config_path=cfg_path)
    with pytest.raises(ValueError, match="must be a list of strings"):
        cfg.injection_phrases


class TestPathAllowlistIntrospection:
    def test_missing_file_has_no_allowlist(self, tmp_path):
        config = PDFConfig(config_path=tmp_path / "nonexistent.toml")
        assert config.has_path_allowlist is False

    def test_empty_allow_list_does_not_count(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("[paths]\nallow = []\n", encoding="utf-8")
        assert PDFConfig(config_path=cfg).has_path_allowlist is False

    def test_deny_only_does_not_count(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[paths]\ndeny = ["/secret/**"]\n', encoding="utf-8")
        assert PDFConfig(config_path=cfg).has_path_allowlist is False

    def test_non_empty_allow_list_counts(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[paths]\nallow = ["/data/pdfs/**"]\n', encoding="utf-8")
        assert PDFConfig(config_path=cfg).has_path_allowlist is True

    def test_config_path_is_exposed(self, tmp_path):
        cfg = tmp_path / "config.toml"
        assert PDFConfig(config_path=cfg).config_path == cfg


class TestEmbeddingBackend:
    def test_default_backend_is_fastembed(self, tmp_path):
        cfg = tmp_path / "config.toml"
        assert PDFConfig(config_path=cfg).embedding_backend == "fastembed"

    def test_explicit_fastembed_backend(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[embedding]\nbackend = "fastembed"\n', encoding="utf-8")
        assert PDFConfig(config_path=cfg).embedding_backend == "fastembed"

    def test_unknown_backend_raises(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[embedding]\nbackend = "ollama-native"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="'fastembed' or 'openai'"):
            PDFConfig(config_path=cfg).embedding_backend

    def test_remote_embedding_spec_is_none_for_fastembed(self, tmp_path):
        cfg = tmp_path / "config.toml"
        assert PDFConfig(config_path=cfg).remote_embedding_spec is None

    def test_openai_backend_requires_base_url(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\nmodel = "nomic-embed-text"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="base_url is required"):
            PDFConfig(config_path=cfg).remote_embedding_spec

    def test_openai_backend_rejects_non_http_base_url(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "ftp://localhost/v1"\n'
            'model = "nomic-embed-text"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must be an http"):
            PDFConfig(config_path=cfg).remote_embedding_spec

    def test_openai_backend_requires_model(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="model is required"):
            PDFConfig(config_path=cfg).remote_embedding_spec

    def test_openai_backend_minimal_config_uses_defaults(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "nomic-embed-text"\n',
            encoding="utf-8",
        )
        spec = PDFConfig(config_path=cfg).remote_embedding_spec
        assert spec is not None
        assert spec.base_url == "http://localhost:11434/v1"
        assert spec.model == "nomic-embed-text"
        assert spec.api_key is None
        assert spec.timeout == 60.0
        assert spec.batch_size == 32
        assert spec.max_concurrency == 4
        assert spec.dimensions is None
        assert spec.document_prefix == ""
        assert spec.query_prefix == ""

    def test_openai_backend_full_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "sk-test-123")
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[embedding]\n"
            'backend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "nomic-embed-text"\n'
            'api_key_env = "MY_API_KEY"\n'
            "timeout = 30\n"
            "batch_size = 16\n"
            "max_concurrency = 2\n"
            "dimensions = 768\n"
            'document_prefix = "search_document: "\n'
            'query_prefix = "search_query: "\n',
            encoding="utf-8",
        )
        spec = PDFConfig(config_path=cfg).remote_embedding_spec
        assert spec is not None
        assert spec.api_key == "sk-test-123"
        assert spec.timeout == 30.0
        assert spec.batch_size == 16
        assert spec.max_concurrency == 2
        assert spec.dimensions == 768
        assert spec.document_prefix == "search_document: "
        assert spec.query_prefix == "search_query: "

    def test_api_key_env_missing_from_environment_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NO_SUCH_KEY", raising=False)
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "nomic-embed-text"\n'
            'api_key_env = "NO_SUCH_KEY"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="NO_SUCH_KEY"):
            PDFConfig(config_path=cfg).remote_embedding_spec

    def test_api_key_never_appears_in_config_toml(self, tmp_path, monkeypatch):
        """The whole point of api_key_env: the secret itself is never
        written to config.toml, only the env var's name."""
        monkeypatch.setenv("MY_API_KEY", "sk-do-not-persist")
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "nomic-embed-text"\n'
            'api_key_env = "MY_API_KEY"\n',
            encoding="utf-8",
        )
        assert "sk-do-not-persist" not in cfg.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("timeout", "0"),
            ("timeout", "-1"),
            ("batch_size", "0"),
            ("max_concurrency", "0"),
            ("dimensions", "0"),
        ],
    )
    def test_numeric_fields_reject_out_of_range(self, tmp_path, field, bad_value):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "nomic-embed-text"\n'
            f"{field} = {bad_value}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            PDFConfig(config_path=cfg).remote_embedding_spec

    def test_embedding_model_unchanged_for_fastembed_default(self, tmp_path):
        """No cache invalidation for existing installs: this feature must
        not change the identity string for anyone not opting into it."""
        cfg = tmp_path / "config.toml"
        assert PDFConfig(config_path=cfg).embedding_model == "BAAI/bge-small-en-v1.5"

    def test_embedding_model_is_namespaced_for_openai_backend(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "nomic-embed-text"\n',
            encoding="utf-8",
        )
        model_name = PDFConfig(config_path=cfg).embedding_model
        assert model_name == "openai:localhost:11434/nomic-embed-text"

    def test_embedding_model_includes_prefix_hash_when_prefixes_set(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[embedding]\nbackend = "openai"\n'
            'base_url = "http://localhost:11434/v1"\n'
            'model = "nomic-embed-text"\n'
            'document_prefix = "search_document: "\n',
            encoding="utf-8",
        )
        model_name = PDFConfig(config_path=cfg).embedding_model
        assert model_name.startswith("openai:localhost:11434/nomic-embed-text@")
