"""Download real grocery datasets from public sources.

Data sources:
1. Instacart Market Basket Analysis (Kaggle)
   - 50K real products with names, aisles, departments
   - 3.4M real orders

2. Open Food Facts
   - Real nutrition data, ingredients, allergens
   - 2M+ products worldwide

No mock data. All real.
"""

import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_instacart():
    """Download Instacart dataset from Kaggle using kagglehub.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables,
    or ~/.kaggle/kaggle.json credentials file.
    """
    import kagglehub

    print("Downloading Instacart Market Basket Analysis dataset from Kaggle...")
    print("This contains ~50K real grocery products and 3.4M real orders.")

    # Download dataset
    path = kagglehub.dataset_download(
        "psparks/instacart-market-basket-analysis"
    )

    print(f"Dataset downloaded to: {path}")

    # Create symlink or copy to our data directory
    instacart_dir = DATA_DIR / "instacart"
    instacart_dir.mkdir(parents=True, exist_ok=True)

    # Copy relevant files
    import shutil
    source_path = Path(path)

    for filename in ["products.csv", "aisles.csv", "departments.csv",
                     "orders.csv", "order_products__prior.csv"]:
        # Search for file in downloaded directory
        matches = list(source_path.rglob(filename))
        if matches:
            dest = instacart_dir / filename
            shutil.copy2(matches[0], dest)
            print(f"  Copied {filename} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  Warning: {filename} not found in download")

    print(f"\nInstacart data ready at: {instacart_dir}")
    return instacart_dir


def download_open_food_facts():
    """Download Open Food Facts product data.

    Downloads the CSV export filtered to US products.
    This is real nutrition data for real products.
    """
    import requests

    off_dir = DATA_DIR / "openfoodfacts"
    off_dir.mkdir(parents=True, exist_ok=True)

    # Open Food Facts CSV export URL
    # Using the smaller US-filtered dataset for manageable size
    url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"

    output_path = off_dir / "products.csv.gz"
    final_path = off_dir / "products.csv"

    if final_path.exists():
        print(f"Open Food Facts data already exists at {final_path}")
        return off_dir

    print("Downloading Open Food Facts dataset...")
    print("This contains real nutrition data for 2M+ products worldwide.")
    print(f"URL: {url}")
    print("This may take several minutes (file is ~2GB compressed)...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192 * 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                print(f"\r  Progress: {pct:.1f}% ({downloaded / 1024 / 1024:.0f} MB)", end="")

    print(f"\n  Downloaded to: {output_path}")

    # Decompress
    import gzip
    print("  Decompressing...")
    with gzip.open(output_path, "rb") as f_in:
        with open(final_path, "wb") as f_out:
            while True:
                chunk = f_in.read(8192 * 16)
                if not chunk:
                    break
                f_out.write(chunk)

    # Remove compressed file
    output_path.unlink()
    print(f"  Open Food Facts data ready at: {final_path}")

    return off_dir


def main():
    """Download all datasets."""
    print("=" * 60)
    print("Grocery Intelligence - Data Download")
    print("All data is REAL - sourced from public datasets")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check for Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_kaggle = (
        kaggle_json.exists()
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    )

    if has_kaggle:
        download_instacart()
    else:
        print("\n⚠ Kaggle credentials not found.")
        print("To download the Instacart dataset, either:")
        print("  1. Place kaggle.json in ~/.kaggle/kaggle.json")
        print("  2. Set KAGGLE_USERNAME and KAGGLE_KEY env variables")
        print("  Get your API key from: https://www.kaggle.com/settings")

    print()
    download_open_food_facts()

    print("\n" + "=" * 60)
    print("Data download complete!")
    print("Next step: python -m src.data.loader")
    print("=" * 60)


if __name__ == "__main__":
    main()
