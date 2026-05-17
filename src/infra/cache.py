"""Redis-backed cache layer with graceful fallback.

The CacheClient is designed so that the application remains fully functional
even when Redis is unreachable. All Redis errors are caught and logged; cache
operations degrade to no-ops rather than propagating exceptions.

Typical usage:

    cache = CacheClient()  # connect lazily via env REDIS_HOST/REDIS_PORT
    key = make_cache_key('search', query, top_k, use_reranker)
    cached = cache.get(key)
    if cached is None:
        result = do_expensive_work()
        cache.set(key, result, ttl=300)
        return result
    return cached
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from loguru import logger

try:  # redis is optional at import time; we still want module to load.
    import redis
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - redis should be installed
    redis = None  # type: ignore[assignment]

    class RedisError(Exception):  # type: ignore[no-redef]
        """Fallback RedisError when redis package is missing."""


_KEY_HASH_THRESHOLD = 200


def make_cache_key(*parts: Any) -> str:
    """Build a stable cache key from arbitrary args.

    Lists and dicts are JSON-serialized with sorted keys for stability.
    If the combined length of all parts exceeds 200 chars, the full key is
    hashed with sha256 and prefixed with a short tag so collisions are
    effectively impossible while keys remain Redis-friendly.

    Example:
        make_cache_key('search', 'low-sugar yogurt', 10, False, [])
        -> 'search:low-sugar yogurt:10:False:[]'
    """
    rendered: list[str] = []
    for p in parts:
        if isinstance(p, (list, dict, tuple, set)):
            try:
                # tuples / sets normalised to list for json
                if isinstance(p, (tuple, set)):
                    p = list(p)
                rendered.append(json.dumps(p, sort_keys=True, default=str))
            except (TypeError, ValueError):
                rendered.append(str(p))
        elif p is None:
            rendered.append("None")
        else:
            rendered.append(str(p))

    joined = ":".join(rendered)
    if len(joined) > _KEY_HASH_THRESHOLD:
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        # Keep a readable prefix (first part) so logs are diagnosable.
        first = rendered[0] if rendered else "key"
        return f"{first}:sha256:{digest}"
    return joined


class CacheClient:
    """Redis-backed cache with graceful fallback when Redis is unreachable.

    All errors are caught and logged; if Redis is down, get() returns None
    and set() is a no-op. The app must remain functional without Redis.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        default_ttl: int = 300,
        prefix: str = "grocery:",
    ) -> None:
        """Try to connect to Redis. If connection fails, mark unavailable but
        don't raise. Health is exposed via .is_available()."""
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        try:
            self.port = int(port if port is not None else os.environ.get("REDIS_PORT", 6379))
        except (TypeError, ValueError):
            self.port = 6379
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._available = False
        self._client: Any = None
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}

        if redis is None:
            logger.warning("redis package not installed; CacheClient running in disabled mode")
            return

        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
            # ping triggers an actual TCP connect so we know if Redis is up.
            self._client.ping()
            self._available = True
            logger.info(f"CacheClient connected to redis://{self.host}:{self.port}")
        except (RedisError, OSError, Exception) as exc:
            self._available = False
            self._client = None
            logger.warning(
                f"CacheClient: redis at {self.host}:{self.port} unreachable ({exc!r}); "
                "operating in fallback mode"
            )

    # -- helpers ---------------------------------------------------------

    def _prefixed(self, key: str) -> str:
        if key.startswith(self.prefix):
            return key
        return f"{self.prefix}{key}"

    # -- public API ------------------------------------------------------

    def get(self, key: str) -> dict | list | None:
        """Look up and JSON-deserialize. None if missing, expired, or Redis down."""
        if not self._available or self._client is None:
            self._stats["misses"] += 1
            return None
        try:
            raw = self._client.get(self._prefixed(key))
        except (RedisError, OSError) as exc:
            self._stats["errors"] += 1
            logger.warning(f"CacheClient.get({key}) error: {exc!r}")
            return None
        if raw is None:
            self._stats["misses"] += 1
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            self._stats["errors"] += 1
            logger.warning(f"CacheClient.get({key}) JSON decode error: {exc!r}")
            return None
        self._stats["hits"] += 1
        return value

    def set(self, key: str, value: dict | list, ttl: int | None = None) -> bool:
        """JSON-serialize and store. Returns True on success, False otherwise."""
        if not self._available or self._client is None:
            return False
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:
            self._stats["errors"] += 1
            logger.warning(f"CacheClient.set({key}) JSON encode error: {exc!r}")
            return False
        try:
            self._client.set(
                self._prefixed(key),
                payload,
                ex=ttl if ttl is not None else self.default_ttl,
            )
        except (RedisError, OSError) as exc:
            self._stats["errors"] += 1
            logger.warning(f"CacheClient.set({key}) error: {exc!r}")
            return False
        self._stats["sets"] += 1
        return True

    def delete(self, key: str) -> None:
        """Delete a key. Silent no-op when Redis is unreachable."""
        if not self._available or self._client is None:
            return
        try:
            self._client.delete(self._prefixed(key))
        except (RedisError, OSError) as exc:
            self._stats["errors"] += 1
            logger.warning(f"CacheClient.delete({key}) error: {exc!r}")

    def is_available(self) -> bool:
        """Returns True iff the client successfully connected at __init__."""
        return self._available

    def stats(self) -> dict:
        """Return {'hits': int, 'misses': int, 'sets': int, 'errors': int}."""
        return dict(self._stats)

    def close(self) -> None:
        """Close the underlying redis connection, if any."""
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception as exc:  # pragma: no cover - close should rarely fail
            logger.warning(f"CacheClient.close() error: {exc!r}")
        finally:
            self._client = None
            self._available = False
