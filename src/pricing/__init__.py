"""Pricing module: mock prices, price history, and promotions.

This package generates deterministic mock prices for products in the
catalog (which ships without price data), simulates 90-day price
histories suitable for charts, and evaluates cart-level promotions.

All randomness is seeded so the same inputs always yield the same
outputs across runs.
"""
