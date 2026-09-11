"""
User-configurable access rules for pdf-mcp.

Loads ~/.config/pdf-mcp/config.toml (optional). Missing file = permissive.
Malformed file = ValueError at startup (never silently fall back to permissive).
"""

from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .embedder import DEFAULT_MODEL
from .remote_embedder import RemoteSpec, identity_for

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
# sane behaviour.
_DEFAULT_REMOTE_TIMEOUT = 60.0
_DEFAULT_REMOTE_BATCH_SIZE = 32
_DEFAULT_REMOTE_MAX_CONCURRENCY = 4


class PDFConfig:
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = _DEFAULT_CONFIG_PATH
        self._config_path = config_path
        self._data = self._load(config_path)

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

    @property
    def embedding_backend(self) -> str:
        """``[embedding].backend``: "fastembed" (default) or "openai".

        Any other value raises ValueError at property-access time (not at
        load time -- this module never touches [embedding] eagerly, matching
        the lazy-property style of the rest of this class).
        """
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
        The embedding identity string that keys the vector cache
        (cache.py's page_embeddings/doc_profiles ``model`` column) and is
        passed to every embedder.py call.

        fastembed backend (default): the bare model name, e.g.
        'BAAI/bge-small-en-v1.5' -- byte-identical to every pre-existing
        install, so no cache is invalidated by this feature shipping.

        openai backend: 'openai:<host>[:<port>]/<model>[@<prefix-hash>]'.
        Host/port are part of the identity deliberately -- this repo's own
        benchmark_data/mlx_backend_results.md measured the SAME model
        weights diverging at cosine 0.894 across two backends (a pooling
        difference), so an endpoint change must be treated as a different
        vector space, not just a different label. The prefix hash is
        appended only when document_prefix/query_prefix are set, for the
        same reason: prepending "search_document: " changes what gets
        embedded, so editing that prefix must re-embed rather than mix
        prefixed and unprefixed vectors under one identity. The API key is
        never included here (it isn't part of what changes the vectors).
        """
        section = self._data.get("embedding", {})
        if self.embedding_backend == "fastembed":
            model: str = section.get("model", DEFAULT_MODEL)
            return model
        spec = self.remote_embedding_spec
        assert spec is not None  # embedding_backend == "openai" guarantees this
        return identity_for(spec)

    @property
    def remote_embedding_spec(self) -> "RemoteSpec | None":
        """Parsed ``[embedding]`` config for the openai backend, or None
        when ``backend`` is "fastembed" (the default).

        Raises ValueError for a missing/malformed base_url, a non-existent
        api_key_env, or an out-of-type numeric field -- matching this
        class's "malformed config = ValueError at property access" contract
        (never silently fall back to a guessed default for something the
        user explicitly configured wrong).
        """
        if self.embedding_backend != "openai":
            return None
        section = self._data.get("embedding", {})

        base_url = section.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                "[embedding].base_url is required when "
                "[embedding].backend = 'openai'"
            )
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"[embedding].base_url must be an http(s) URL, got {base_url!r}"
            )

        model = section.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "[embedding].model is required when " "[embedding].backend = 'openai'"
            )

        api_key: "str | None" = None
        api_key_env = section.get("api_key_env")
        if api_key_env is not None:
            if not isinstance(api_key_env, str) or not api_key_env.strip():
                raise ValueError("[embedding].api_key_env must be a non-empty string")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"[embedding].api_key_env names {api_key_env!r}, but that "
                    "environment variable is unset or empty"
                )

        timeout = section.get("timeout", _DEFAULT_REMOTE_TIMEOUT)
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError(
                f"[embedding].timeout must be a positive number, got {timeout!r}"
            )

        batch_size = section.get("batch_size", _DEFAULT_REMOTE_BATCH_SIZE)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError(
                f"[embedding].batch_size must be a positive integer, got {batch_size!r}"
            )

        max_concurrency = section.get(
            "max_concurrency", _DEFAULT_REMOTE_MAX_CONCURRENCY
        )
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency < 1
        ):
            raise ValueError(
                "[embedding].max_concurrency must be a positive integer, got "
                f"{max_concurrency!r}"
            )

        dimensions = section.get("dimensions")
        if dimensions is not None and (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions < 1
        ):
            raise ValueError(
                f"[embedding].dimensions must be a positive integer, got {dimensions!r}"
            )

        document_prefix = section.get("document_prefix", "")
        query_prefix = section.get("query_prefix", "")
        if not isinstance(document_prefix, str):
            raise ValueError("[embedding].document_prefix must be a string")
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
