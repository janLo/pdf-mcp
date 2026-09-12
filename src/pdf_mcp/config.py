"""
User-configurable access rules for pdf-mcp.

Loads ~/.config/pdf-mcp/config.toml (optional). Missing file = permissive.
Malformed file = ValueError at startup (never silently fall back to permissive).
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Any

from urllib.parse import urlsplit

from .embedder import DEFAULT_MODEL, is_bge_small_compatible
from .remote_embedder import RemoteSpec, _redact_base_url, identity_for
from .url_fetcher import URLFetcher

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "pdf-mcp" / "config.toml"

_DEFAULT_MAX_RESPONSE_BYTES = 200_000
_MAX_RESPONSE_BYTES_CEILING = 2_000_000
_MIN_RESPONSE_BYTES = 4_096

# [embedding] defaults for the "openai" backend. Mirrors RemoteSpec's own
# defaults so a config.toml that sets only backend/base_url/model still gets
# sane values.
_DEFAULT_REMOTE_TIMEOUT = 60.0
_DEFAULT_REMOTE_BATCH_SIZE = 32
_DEFAULT_REMOTE_MAX_CONCURRENCY = 4

# Cosine-similarity threshold below which pdf_search/pdf_corpus_search flag a
# semantic match `low_confidence`. Historically a single module constant in
# server.py (`_SEMANTIC_CONFIDENCE_THRESHOLD`); moved here so it can be
# overridden per-config and so `confidence_threshold` below can decide,
# per model, whether this default is even meaningful to apply. Tuned to
# BAAI/bge-small-en-v1.5's own cosine-similarity distribution -- see
# `confidence_threshold`'s docstring.
_DEFAULT_BGE_SMALL_CONFIDENCE_THRESHOLD = 0.5


class PDFConfig:
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = _DEFAULT_CONFIG_PATH
        self._config_path = config_path
        self._data = self._load(config_path)
        # Set by disable_remote_embedding_backend() when the startup safety
        # check (issue #42, remote_embedding_check) decides the configured
        # remote endpoint cannot be trusted. Once set, embedding_backend,
        # embedding_model, and remote_embedding_spec all report the local
        # fastembed default for the rest of this process -- see
        # disable_remote_embedding_backend's docstring.
        self._remote_backend_disabled = False

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "rb") as f:
                data: dict[str, Any] = tomllib.load(f)
                return data
        except Exception as e:
            raise ValueError(f"Failed to parse config file {path}: {e}") from e

    def check_path(self, path: str) -> None:
        """Enforce [paths] allow/deny rules. Raises ValueError if denied."""
        rules = self._data.get("paths", {})
        allow: list[str] = rules.get("allow", [])
        deny: list[str] = rules.get("deny", [])

        resolved = str(Path(path).expanduser().resolve())

        for pattern in deny:
            expanded = str(Path(pattern).expanduser())
            if fnmatch.fnmatch(resolved, expanded):
                raise ValueError(f"Path denied by config: {path}")

        if allow:
            for pattern in allow:
                expanded = str(Path(pattern).expanduser())
                if fnmatch.fnmatch(resolved, expanded):
                    return
            raise ValueError(f"Path not in allowed list: {path}")

    def disable_remote_embedding_backend(self) -> None:
        """Force the local fastembed default for the rest of this process.

        Called exactly once, by server.py's startup safety check
        (`remote_embedding_check`, issue #42), when a configured
        ``[embedding].backend = "openai"`` endpoint fails the cosine-parity
        check against stored fastembed reference vectors -- wrong model,
        wrong quantization, wrong pooling, or simply unreachable. After
        this call, `embedding_backend`, `embedding_model`, and
        `remote_embedding_spec` all behave exactly as if
        ``[embedding].backend`` had never been set to "openai", regardless
        of what config.toml says -- this is what makes the fallback safe:
        every call site that resolves an embedding identity from this
        object (search-capability probing, cache identity, the actual
        encode() dispatch) sees the same, consistent, local-only state, so
        nothing downstream can end up with `embedding_model` naming a
        remote identity that `embedder.configure_remote` was never told
        about. Idempotent.
        """
        self._remote_backend_disabled = True

    @property
    def embedding_backend(self) -> str:
        """``[embedding].backend``: "fastembed" (default) or "openai".

        "openai" means an OpenAI-compatible ``/v1/embeddings`` HTTP endpoint
        (ollama, lemonade, llama-server, vLLM, OpenAI, ...) -- see
        remote_embedding_spec's docstring for the model-choice caveat this
        entails (issue #46). Validated eagerly at load time -- this module
        never touches [embedding] lazily, matching the rest of this class.

        Reports "fastembed" unconditionally once
        `disable_remote_embedding_backend` has been called, regardless of
        what config.toml says -- see that method's docstring.
        """
        if self._remote_backend_disabled:
            return "fastembed"
        backend = self._data.get("embedding", {}).get("backend", "fastembed")
        if backend not in ("fastembed", "openai"):
            raise ValueError(
                f"[embedding].backend must be 'fastembed' or 'openai', got "
                f"{backend!r}"
            )
        return str(backend)

    @property
    def embedding_model(self) -> str:
        """
        The embedding identity string that keys the vector cache (cache.py's
        page_embeddings/doc_profiles ``model`` column) and is passed to
        embedder.encode/encode_query/check_available.

        fastembed backend (default): the bare model name, e.g.
        "BAAI/bge-small-en-v1.5" -- byte-identical to every existing install,
        so no cache is invalidated by this backend existing.

        openai backend: 'openai:<host>[:<port>]/<model>[@<prefix-hash>]' (see
        remote_embedder.identity_for). Host/port and the prefix hash are
        part of the identity deliberately: a different endpoint, or a
        different document_prefix/query_prefix (which changes what text
        actually gets embedded), may be a different vector space even
        under the "same" model name, so none of them may share a cache row.

        After `disable_remote_embedding_backend` has been called, always
        returns `DEFAULT_MODEL` -- NOT `[embedding].model` (that key, when
        present, names the *remote* model label, which is meaningless to
        fastembed's local catalog and would raise in `embedder.
        check_available` if used here).
        """
        if self._remote_backend_disabled:
            return DEFAULT_MODEL
        if self.embedding_backend == "fastembed":
            model: str = self._data.get("embedding", {}).get("model", DEFAULT_MODEL)
            return model
        spec = self.remote_embedding_spec
        assert spec is not None  # embedding_backend == "openai" guarantees this
        return identity_for(spec)

    @property
    def remote_embedding_spec(self) -> "RemoteSpec | None":
        """Parsed ``[embedding]`` config for the openai backend, or None
        when ``backend`` is "fastembed" (the default).

        `model` here genuinely selects what gets encoded and how (via
        `document_prefix`/`query_prefix` below), unlike the narrower first
        version of this feature (issue #42) where it was cache-naming only.
        This reopens the concern that version was scoped specifically to
        avoid: `_SEMANTIC_CONFIDENCE_THRESHOLD` in server.py is tuned to
        bge-small-en-v1.5's own cosine-similarity distribution, and this
        property does not itself validate that a configured model's
        distribution is compatible with that tuning. `confidence_threshold`
        below is the resolution issue #46 asked for: it degrades
        `low_confidence` to null (with a startup warning) rather than
        silently reusing bge-small's tuning for an incompatible model,
        unless a real, calibrated value is set explicitly (see
        `scripts/calibrate_confidence_threshold.py`).

        `document_prefix`/`query_prefix` default to "" (bge-small's own
        contract: no prefix), and `dimensions`, when set, is validated by
        remote_embedder.encode against the endpoint's actual response
        width.

        `api_key_env` names an environment variable; the key itself is
        never read from or written to config.toml.

        `base_url`'s hostname must resolve to a private/loopback address
        (checked via `url_fetcher.URLFetcher._is_blocked_ip`, reused the
        other way round) -- an accident-guard against sending page/query
        text to a public API by mistake, not a security boundary (won't
        catch a tunnel/VPN routing a private address to a public host). A
        hostname that fails to resolve at all is treated as "blocked" by
        `_is_blocked_ip` (fail-closed for its SSRF use), which here means
        this check passes rather than raises -- e.g. a compose service
        that hasn't started yet won't block config load.
        """
        if self.embedding_backend != "openai":
            return None
        section = self._data.get("embedding", {})

        base_url = section.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(
                "[embedding].base_url is required when "
                "[embedding].backend = 'openai'"
            )
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise ValueError(
                "[embedding].base_url must be an http(s) URL, got "
                f"{_redact_base_url(base_url)!r}"
            )

        hostname = urlsplit(base_url).hostname
        if hostname is None:
            raise ValueError(
                "[embedding].base_url must include a hostname, got "
                f"{_redact_base_url(base_url)!r}"
            )
        if not URLFetcher._is_blocked_ip(hostname):
            raise ValueError(
                f"[embedding].base_url {_redact_base_url(base_url)!r} "
                "resolves to a public address -- this guards against "
                "pointing pdf-mcp at a public API by accident, not a "
                "security boundary (see docs/configuration.md). Point it "
                "at a loopback, RFC 1918/link-local/CGNAT, or otherwise "
                "private-resolving address -- 'localhost', "
                "'host.docker.internal', a Docker/compose service name, a "
                "LAN hostname/IP, or a Tailscale address all resolve as "
                "private and are accepted."
            )

        model = section.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(
                "[embedding].model is required when [embedding].backend = 'openai'"
            )

        api_key = None
        api_key_env = section.get("api_key_env")
        if api_key_env is not None:
            if not isinstance(api_key_env, str) or not api_key_env:
                raise ValueError("[embedding].api_key_env must be a non-empty string")
            import os

            api_key = os.environ.get(api_key_env)
            if api_key is None:
                raise ValueError(
                    f"[embedding].api_key_env names {api_key_env!r}, but that "
                    "environment variable is not set"
                )

        timeout = section.get("timeout", _DEFAULT_REMOTE_TIMEOUT)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise ValueError(
                f"[embedding].timeout must be a positive number, got {timeout!r}"
            )
        if timeout <= 0:
            raise ValueError(
                f"[embedding].timeout must be a positive number, got {timeout!r}"
            )

        batch_size = section.get("batch_size", _DEFAULT_REMOTE_BATCH_SIZE)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError(
                f"[embedding].batch_size must be a positive integer, got "
                f"{batch_size!r}"
            )
        if batch_size <= 0:
            raise ValueError(
                f"[embedding].batch_size must be a positive integer, got "
                f"{batch_size!r}"
            )

        max_concurrency = section.get(
            "max_concurrency", _DEFAULT_REMOTE_MAX_CONCURRENCY
        )
        if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool):
            raise ValueError(
                "[embedding].max_concurrency must be a positive integer, got "
                f"{max_concurrency!r}"
            )
        if max_concurrency <= 0:
            raise ValueError(
                "[embedding].max_concurrency must be a positive integer, got "
                f"{max_concurrency!r}"
            )

        dimensions = section.get("dimensions")
        if dimensions is not None:
            if not isinstance(dimensions, int) or isinstance(dimensions, bool):
                raise ValueError(
                    f"[embedding].dimensions must be a positive integer, got "
                    f"{dimensions!r}"
                )
            if dimensions <= 0:
                raise ValueError(
                    f"[embedding].dimensions must be a positive integer, got "
                    f"{dimensions!r}"
                )

        document_prefix = section.get("document_prefix", "")
        if not isinstance(document_prefix, str):
            raise ValueError("[embedding].document_prefix must be a string")

        query_prefix = section.get("query_prefix", "")
        if not isinstance(query_prefix, str):
            raise ValueError("[embedding].query_prefix must be a string")

        return RemoteSpec(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=float(timeout),
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            dimensions=dimensions,
            document_prefix=document_prefix,
            query_prefix=query_prefix,
        )

    @property
    def remote_embedding_verify_startup(self) -> bool:
        """``[embedding].verify_startup``: default True.

        Gates the cosine-parity safety check (`remote_embedding_check`,
        issue #42) that runs once at startup when the "openai" backend is
        configured AND the configured model is bge-small-compatible (see
        `embedder.is_bge_small_compatible` and server.py's startup gating --
        the check has nothing but a bge-small-shaped reference to compare
        against, so it is meaningless, and would be spuriously failing, for
        a model that intentionally names something else): it embeds a
        handful of fixed reference sentences through the remote endpoint
        and compares them to stored local fastembed vectors, falling back
        to local fastembed if they don't match closely enough. True by
        default -- the check is what makes the remote backend trustworthy
        without pdf-mcp being able to see what model the endpoint actually
        serves. Set to false only for an endpoint whose identity is already
        verified out-of-band and where the extra startup round-trip is
        unwanted (e.g. a very slow cold start some servers have on the
        first request).

        Meaningless, and not read, for the default fastembed backend, or
        for an "openai" backend configured with a non-bge-small `model`
        (there is nothing bge-small-shaped to compare a different model's
        vectors against -- see `confidence_threshold` for that case
        instead).
        """
        value = self._data.get("embedding", {}).get("verify_startup", True)
        if not isinstance(value, bool):
            raise ValueError(
                f"[embedding].verify_startup must be true or false, got {value!r}"
            )
        return value

    @property
    def confidence_threshold(self) -> float | None:
        """``[embedding].confidence_threshold``: the cosine-similarity cutoff
        below which pdf_search/pdf_corpus_search flag a semantic (or hybrid)
        match `low_confidence`.

        This resolves what value server.py should actually use, given three
        cases:

        1. Explicitly set in config.toml -- used verbatim, for ANY backend
           or model, after validating it is a number in [-1.0, 1.0] (the
           valid range for cosine similarity on a normalized embedding).
           This is the only way to get `low_confidence` for a non-bge-small
           REMOTE model: run `scripts/calibrate_confidence_threshold.py`
           against your endpoint and paste its suggested value here.

        2. Not set, and the backend is "fastembed" (the default, local
           path) -- always falls back to
           `_DEFAULT_BGE_SMALL_CONFIDENCE_THRESHOLD` (0.5), REGARDLESS of
           which local model is configured. This preserves this
           codebase's pre-existing behavior for `docs/embedding-models.md`'s
           documented local-model alternatives (e.g.
           `snowflake/snowflake-arctic-embed-s`), which predate this
           property and were never gated on bge-small compatibility --
           issue #42/#46's concern is specifically an arbitrary *remote*
           model silently reusing bge-small's tuning with no way to
           verify it, not the already-supported local-model case. A local
           non-bge-small model keeps whatever accuracy `low_confidence`
           already had for it before this change; that is a pre-existing,
           separately-tracked concern, not a regression introduced here.

        3. Not set, and the backend is "openai" with a `model` that is NOT
           bge-small-compatible (`embedder.is_bge_small_compatible`) --
           returns None. server.py reads None as "we do not know what a
           meaningful cutoff is for this model's cosine distribution" and
           reports `low_confidence`/`all_results_low_confidence` as null
           plus `confidence_unavailable=True`, rather than silently
           reusing a threshold tuned for a different model's score
           distribution. This is the failure mode
           https://github.com/jztan/pdf-mcp/issues/42 flagged and
           https://github.com/jztan/pdf-mcp/issues/46 tracks fixing. (An
           "openai" backend with a bge-small-compatible `model` also
           returns 0.5 here, same as case 2 -- same vector space, same
           tuning, verified at startup by the safety check above.)

        Why case 3 degrades instead of raising ValueError at config-load
        time (unlike `base_url`/`model` above, which DO raise when
        missing): an unset threshold does not make the "openai" backend
        inoperable -- encoding, keyword search, and hybrid RRF ranking
        (see `_rrf_fuse` in server.py -- rank-based, not score-magnitude-
        based, so it is unaffected by this) all still work correctly.
        Only the derived `low_confidence` signal is compromised, which is
        exactly the situation `semantic_unavailable` and `text_coverage`
        already handle elsewhere in server.py: report the gap loudly on
        every affected response and in a one-time startup warning
        (server.py, right after `configure_remote`), rather than block a
        server whose search and extraction tools work fine otherwise.
        """
        section = self._data.get("embedding", {})
        raw = section.get("confidence_threshold")
        if raw is not None:
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise ValueError(
                    "[embedding].confidence_threshold must be a number "
                    f"between -1.0 and 1.0, got {raw!r}"
                )
            if not (-1.0 <= raw <= 1.0):
                raise ValueError(
                    "[embedding].confidence_threshold must be a number "
                    f"between -1.0 and 1.0, got {raw!r}"
                )
            return float(raw)

        if self.embedding_backend == "fastembed":
            # Any local model keeps the pre-existing 0.5 default -- see
            # case 2 in the docstring above. Only a remote, non-bge-small
            # model is genuinely unknown territory.
            return _DEFAULT_BGE_SMALL_CONFIDENCE_THRESHOLD

        spec = self.remote_embedding_spec
        assert spec is not None  # embedding_backend == "openai" guarantees this
        if is_bge_small_compatible(spec.model):
            return _DEFAULT_BGE_SMALL_CONFIDENCE_THRESHOLD
        return None

    @property
    def fts_language(self) -> str | None:
        """``[fts] language``: None (default, porter/English stemming) or
        "de" for the German-stemmed FTS mirror index (cache.py's
        pdf_search_fts_de / pdf_section_fts_de tables).

        This is a whole-cache setting, not per-document: it applies to every
        PDF the running server touches, the same way `embedding_model` does.
        A user who mostly reads German documents turns it on once; mixed
        English/German corpora are not distinguished (that would need
        per-document language detection, which this option deliberately
        does not attempt).
        """
        value = self._data.get("fts", {}).get("language")
        if value is None:
            return None
        if value == "de":
            return "de"
        raise ValueError(
            f"[fts] language must be 'de' (or omitted) in {self._config_path}, "
            f"got {value!r}"
        )

    @property
    def ocr_auto_install(self) -> bool | None:
        """``[ocr] auto_install``: None when absent. Malformed values fail loudly."""
        value = self._data.get("ocr", {}).get("auto_install")
        if value is None or isinstance(value, bool):
            return value
        raise ValueError(
            f"[ocr] auto_install must be true or false in {self._config_path}, "
            f"got {value!r}"
        )

    @property
    def update_check(self) -> bool | None:
        """``[updates] check``: None when absent. Malformed values fail loudly."""
        value = self._data.get("updates", {}).get("check")
        if value is None or isinstance(value, bool):
            return value
        raise ValueError(
            f"[updates] check must be true or false in {self._config_path}, "
            f"got {value!r}"
        )

    @property
    def config_path(self) -> Path:
        """Path this config was loaded from (may not exist)."""
        return self._config_path

    @property
    def has_path_allowlist(self) -> bool:
        """
        True when [paths] allow is a non-empty list.

        An absent or empty allow list disables only the allow gate in
        check_path; [paths] deny rules still apply unconditionally. A
        deny-only posture is the correct default for a local install and
        insufficient for a remote one, so main_http() refuses to start
        when this is False.
        """
        allow = self._data.get("paths", {}).get("allow", [])
        return isinstance(allow, list) and len(allow) > 0

    @property
    def path_allow_patterns(self) -> tuple[str, ...]:
        """
        Configured [paths] allow globs, verbatim and unexpanded.

        Reported by server_info so a caller can see which corpus the
        server is willing to open. Returned as-is (no ~ expansion, no
        glob resolution): check_path is the authority on what these
        mean, and a presentation layer that reinterpreted them could
        drift from it.
        """
        allow = self._data.get("paths", {}).get("allow", [])
        if not isinstance(allow, list):
            return ()
        return tuple(str(p) for p in allow)

    @property
    def path_deny_patterns(self) -> tuple[str, ...]:
        """
        Configured [paths] deny globs, verbatim and unexpanded.

        Deny applies unconditionally, including when there is no allow
        list, so this is meaningful in both access modes.
        """
        deny = self._data.get("paths", {}).get("deny", [])
        if not isinstance(deny, list):
            return ()
        return tuple(str(p) for p in deny)

    @property
    def max_response_bytes(self) -> int:
        """
        Maximum UTF-8 byte size of the **text content** returned by
        `pdf_read_all` (the `full_text` field) and section-granularity
        `pdf_search` (the sum of included section titles plus a per-entry
        overhead estimate). This bounds the content the cap was designed
        to bound — the field an LLM sees as untrusted PDF data.

        Note: this is NOT a wire-level envelope cap. The MCP TextContent
        block that crosses the transport also carries the other response
        fields (`truncated`, `next_page`, etc.) plus JSON framing
        overhead, typically adding ~300–500 bytes on top of this limit.
        Callers that need strict wire-size enforcement should pick a
        cap a few KB below their transport ceiling.

        Loaded from `[limits].max_response_bytes` in config.toml. Values
        above `_MAX_RESPONSE_BYTES_CEILING` are clamped down; values below
        `_MIN_RESPONSE_BYTES` are clamped up.
        """
        raw = self._data.get("limits", {}).get(
            "max_response_bytes", _DEFAULT_MAX_RESPONSE_BYTES
        )
        if not isinstance(raw, int):
            raise ValueError(
                f"[limits].max_response_bytes must be an integer, "
                f"got {type(raw).__name__}"
            )
        return max(_MIN_RESPONSE_BYTES, min(_MAX_RESPONSE_BYTES_CEILING, raw))

    @property
    def injection_phrases(self) -> tuple[str, ...]:
        """Extra hidden-text injection phrases from
        ``[content_trust].injection_phrases``. These EXTEND the built-in
        English defaults (never replace them) and enable non-English coverage.

        Returns the raw user strings; normalization happens at the matching
        site in ``content_trust`` (this module stays free of a content_trust
        import). Missing table/key -> empty tuple. A value that is not a list
        of strings raises ``ValueError`` — consistent with the
        never-silently-permissive config contract.
        """
        raw = self._data.get("content_trust", {}).get("injection_phrases", [])
        if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
            raise ValueError(
                "[content_trust].injection_phrases must be a list of strings"
            )
        return tuple(raw)

    def check_url_host(self, hostname: str) -> None:
        """Enforce [urls] allow/deny rules. Raises ValueError if denied."""
        rules = self._data.get("urls", {})
        allow: list[str] = rules.get("allow", [])
        deny: list[str] = rules.get("deny", [])

        host = hostname.lower()

        for pattern in deny:
            if fnmatch.fnmatch(host, pattern.lower()):
                raise ValueError(f"URL host denied by config: {hostname}")

        if allow:
            for pattern in allow:
                if fnmatch.fnmatch(host, pattern.lower()):
                    return
            raise ValueError(f"URL host not in allowed list: {hostname}")
