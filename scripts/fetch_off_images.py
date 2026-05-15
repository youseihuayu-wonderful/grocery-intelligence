"""Best-effort: fetch Open Food Facts image URLs for popular Instacart products.

This is a stretch goal. OFF matching by raw Instacart name is unreliable —
expect ~10-20% match rate at best. The system MUST work without these images,
so this script is wrapped in defensive error handling and never raises.

Run:
    cd /Users/shihuayu/grocery-intelligence
    source venv/bin/activate
    python scripts/fetch_off_images.py
"""

from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "product_catalog.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "product_images.parquet"

OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
SAMPLE_SIZE = 200
REQUEST_DELAY_S = 0.3
REQUEST_TIMEOUT_S = 10
USER_AGENT = "grocery-intelligence/0.1 (best-effort image enrichment)"


def _clean_query(name: str) -> str:
    """Trim the product name into something more OFF-search-friendly.

    Instacart names often start with "Organic" / "Bag of" / etc. OFF's index
    keys on the actual brand + product, so we strip those qualifiers.
    """
    cleaned = name.lower()
    # Drop the most common Instacart-isms; this is intentionally tiny so we
    # don't over-engineer for a 10-20% expected hit rate.
    for prefix in ("organic ", "bag of ", "large ", "small ", "fresh "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    # Collapse whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or name


def _query_off(session: requests.Session, name: str) -> str | None:
    """Hit OFF once for `name`. Returns image_small_url or None."""
    params = {
        "action": "process",
        "search_terms": _clean_query(name),
        "json": 1,
        "page_size": 1,
    }
    response = session.get(OFF_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    products = payload.get("products") or []
    if not products:
        return None
    image = products[0].get("image_small_url")
    return image or None


def main() -> int:
    if not CATALOG_PATH.exists():
        print(f"Catalog not found at {CATALOG_PATH}; nothing to do.", file=sys.stderr)
        return 0

    catalog = pd.read_parquet(CATALOG_PATH)
    if catalog.empty:
        print("Catalog is empty; exiting cleanly.")
        return 0

    # Take the top-N most ordered products. They're the ones most worth a
    # real image and are also most likely to have OFF matches.
    sample = catalog.nlargest(SAMPLE_SIZE, "order_count")[
        ["product_id", "product_name"]
    ]
    print(f"Trying OFF lookup for top {len(sample)} products by order_count...")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    matches: list[dict[str, object]] = []
    tried = 0
    network_failures = 0
    for row in sample.itertuples(index=False):
        tried += 1
        try:
            image_url = _query_off(session, str(row.product_name))
        except (requests.RequestException, ValueError) as exc:
            # ValueError catches malformed JSON. We swallow per-request errors
            # and keep going so a single 5xx doesn't blow up the whole batch.
            network_failures += 1
            # If half of the sample so far has died on the network, the API
            # is clearly down and we should bail rather than burn 200 retries.
            if tried >= 20 and network_failures / tried > 0.5:
                print(
                    f"Aborting early: {network_failures}/{tried} requests failed."
                    f" Last error: {exc!r}"
                )
                break
            time.sleep(REQUEST_DELAY_S)
            continue
        if image_url:
            matches.append(
                {
                    "product_id": int(row.product_id),
                    "image_url": image_url,
                }
            )
        time.sleep(REQUEST_DELAY_S)

    match_rate = (len(matches) / tried) if tried else 0.0
    print(f"Total tried:   {tried}")
    print(f"Total matched: {len(matches)}")
    print(f"Match rate:    {match_rate:.1%}")
    print(f"Failures:      {network_failures}")

    if matches:
        out = pd.DataFrame(matches)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUTPUT_PATH, index=False)
        print(f"Wrote {len(out)} image URLs to {OUTPUT_PATH}")
    else:
        # Don't write an empty parquet — downstream code can just check for
        # file existence to know whether image data is available at all.
        print("No matches; not writing parquet.")

    return 0


if __name__ == "__main__":
    # Outer safety net: this script must NEVER hard-fail. If OFF is down,
    # the rest of the system needs to keep working with emoji icons.
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 — intentional broad catch
        print("fetch_off_images.py failed; ignoring.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(0)
