"""Tests for the Redis cache layer (src.infra.cache).

Uses fakeredis so the suite runs without a real Redis server. All Redis
client construction inside CacheClient is patched to return a fakeredis
instance via monkeypatching `redis.Redis`. For the "unreachable Redis"
case we patch `redis.Redis` to raise ConnectionError on .ping().
"""

from __future__ import annotations

import time
from unittest.mock import patch

import fakeredis
import pytest
import redis as real_redis
from redis.exceptions import ConnectionError as RedisConnectionError

from src.infra.cache import CacheClient, make_cache_key


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def fake_redis_factory():
    """Return a factory that yields fakeredis StrictRedis instances.

    Patches `redis.Redis` so that CacheClient.__init__ binds to fakeredis
    instead of a real TCP connection.
    """
    # Single shared server so all clients created in a test see the same data.
    server = fakeredis.FakeServer()

    def _build(*args, **kwargs):
        # decode_responses must propagate for json.loads to receive str
        decode = kwargs.get("decode_responses", False)
        return fakeredis.FakeStrictRedis(server=server, decode_responses=decode)

    with patch.object(real_redis, "Redis", side_effect=_build):
        yield


@pytest.fixture
def cache(fake_redis_factory):
    """A connected CacheClient backed by fakeredis."""
    client = CacheClient(host="localhost", port=6379, prefix="test:")
    yield client
    client.close()


# ----------------------------------------------------------- make_cache_key


class TestMakeCacheKey:
    def test_basic_join(self):
        key = make_cache_key("search", "low-sugar yogurt", 10, False, [])
        assert key == "search:low-sugar yogurt:10:False:[]"

    def test_stable_across_calls(self):
        a = make_cache_key("a", {"x": 1, "y": 2}, [1, 2, 3])
        b = make_cache_key("a", {"y": 2, "x": 1}, [1, 2, 3])
        # dicts must serialize to the same key regardless of insertion order
        assert a == b

    def test_hashes_long_inputs(self):
        long_query = "x" * 250
        key = make_cache_key("search", long_query)
        # Should fall back to sha256 form, not contain the literal long string
        assert "sha256" in key
        assert len(key) < 300
        assert long_query not in key

    def test_short_inputs_not_hashed(self):
        key = make_cache_key("feed", "user_42", 5)
        assert "sha256" not in key
        assert key == "feed:user_42:5"

    def test_none_handled(self):
        key = make_cache_key("qa", None, 3)
        assert key == "qa:None:3"


# ----------------------------------------------------------------- CacheClient


class TestCacheClient:
    def test_is_available_when_connected(self, cache):
        assert cache.is_available() is True

    def test_get_returns_none_for_missing_key(self, cache):
        assert cache.get("does-not-exist") is None
        assert cache.stats()["misses"] == 1

    def test_set_and_get_dict(self, cache):
        value = {"name": "yogurt", "price": 3.99, "tags": ["dairy"]}
        assert cache.set("product:1", value) is True
        got = cache.get("product:1")
        assert got == value

    def test_set_and_get_list(self, cache):
        value = [{"id": 1}, {"id": 2}, {"id": 3}]
        assert cache.set("results", value) is True
        got = cache.get("results")
        assert got == value

    def test_delete(self, cache):
        cache.set("temp", {"x": 1})
        assert cache.get("temp") == {"x": 1}
        cache.delete("temp")
        assert cache.get("temp") is None

    def test_ttl_expiry(self, cache):
        cache.set("short-lived", {"data": "soon-gone"}, ttl=1)
        assert cache.get("short-lived") == {"data": "soon-gone"}
        time.sleep(1.2)
        assert cache.get("short-lived") is None

    def test_stats_counters(self, cache):
        # set increments sets
        cache.set("a", {"x": 1})
        cache.set("b", {"y": 2})
        # hits & misses
        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("c")  # miss
        s = cache.stats()
        assert s["sets"] == 2
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["errors"] == 0

    def test_stats_returns_copy(self, cache):
        s1 = cache.stats()
        s1["hits"] = 99999
        s2 = cache.stats()
        assert s2["hits"] == 0  # mutation of returned dict must not leak


# -------------------------------------------------- unreachable Redis fallback


class TestGracefulFallback:
    def test_unreachable_redis_marks_unavailable(self):
        """When ping raises ConnectionError, the client must not raise."""

        class _FailingClient:
            def __init__(self, *a, **kw):
                pass

            def ping(self):
                raise RedisConnectionError("nope")

            def close(self):
                pass

        with patch.object(real_redis, "Redis", _FailingClient):
            client = CacheClient(host="bogus", port=1)
            assert client.is_available() is False

    def test_get_returns_none_when_unavailable(self):
        class _FailingClient:
            def __init__(self, *a, **kw):
                pass

            def ping(self):
                raise RedisConnectionError("nope")

        with patch.object(real_redis, "Redis", _FailingClient):
            client = CacheClient(host="bogus", port=1)
            assert client.get("anything") is None

    def test_set_returns_false_when_unavailable(self):
        class _FailingClient:
            def __init__(self, *a, **kw):
                pass

            def ping(self):
                raise RedisConnectionError("nope")

        with patch.object(real_redis, "Redis", _FailingClient):
            client = CacheClient(host="bogus", port=1)
            assert client.set("k", {"v": 1}) is False

    def test_connection_error_during_get(self):
        """Even after successful connect, runtime ConnectionError must be caught."""

        class _SometimesFailing:
            def __init__(self, *a, **kw):
                self._set_ok = True

            def ping(self):
                return True

            def get(self, key):
                raise RedisConnectionError("connection lost")

            def set(self, key, value, ex=None):
                raise RedisConnectionError("connection lost")

            def delete(self, key):
                raise RedisConnectionError("connection lost")

            def close(self):
                pass

        with patch.object(real_redis, "Redis", _SometimesFailing):
            client = CacheClient(host="x", port=1)
            assert client.is_available() is True
            # Should not raise
            assert client.get("k") is None
            assert client.set("k", {"v": 1}) is False
            client.delete("k")  # should not raise
            assert client.stats()["errors"] >= 2
