"""Search analytics dashboard module.

Aggregate metrics over the behavior event log: top queries, conversion
funnel, hot products, daily counts, category breakdown, per-user
activity, and high-level search-quality signals.

All functions take a ``BehaviorLogger`` (or anything else exposing
``count_events`` / ``get_events`` plus a ``db_path`` / ``_conn``) and
return plain Python data structures suitable for JSON serialization.
"""
