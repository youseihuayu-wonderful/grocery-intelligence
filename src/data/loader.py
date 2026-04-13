"""Load and merge real grocery product data from Instacart and Open Food Facts."""

import os
from pathlib import Path

import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_instacart_products() -> pd.DataFrame:
    """Load product catalog from Instacart dataset.

    Expected files in data/raw/instacart/:
        - products.csv: product_id, product_name, aisle_id, department_id
        - aisles.csv: aisle_id, aisle
        - departments.csv: department_id, department
    """
    instacart_dir = RAW_DIR / "instacart"

    products = pd.read_csv(instacart_dir / "products.csv")
    aisles = pd.read_csv(instacart_dir / "aisles.csv")
    departments = pd.read_csv(instacart_dir / "departments.csv")

    # Merge to get full product info with aisle and department names
    df = products.merge(aisles, on="aisle_id", how="left")
    df = df.merge(departments, on="department_id", how="left")

    logger.info(f"Loaded {len(df)} products from Instacart dataset")
    return df


def load_instacart_orders() -> pd.DataFrame:
    """Load order history for popularity and co-purchase analysis.

    Returns DataFrame with order_id, product_id, reordered, add_to_cart_order.
    """
    instacart_dir = RAW_DIR / "instacart"
    prior = pd.read_csv(instacart_dir / "order_products__prior.csv")
    logger.info(f"Loaded {len(prior):,} order-product records from Instacart")
    return prior


def compute_product_popularity(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Compute order frequency and reorder rate per product from real order data."""
    stats = orders_df.groupby("product_id").agg(
        order_count=("product_id", "count"),
        reorder_rate=("reordered", "mean"),
    ).reset_index()
    logger.info(f"Computed popularity stats for {len(stats):,} products")
    return stats


def load_open_food_facts() -> pd.DataFrame:
    """Load nutrition and ingredient data from Open Food Facts.

    Uses the Kaggle version: en.openfoodfacts.org.products.tsv
    """
    off_dir = RAW_DIR / "openfoodfacts"
    filepath = off_dir / "en.openfoodfacts.org.products.tsv"

    usecols = [
        "product_name", "brands", "categories_en",
        "ingredients_text", "allergens_en",
        "energy_100g", "proteins_100g", "sugars_100g",
        "fat_100g", "fiber_100g",
        "nutrition_grade_fr",
    ]

    df = pd.read_csv(filepath, usecols=usecols, sep="\t", low_memory=False)
    df = df.dropna(subset=["product_name"])

    logger.info(f"Loaded {len(df)} products from Open Food Facts")
    return df


def enrich_products(
    instacart_df: pd.DataFrame,
    off_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich Instacart products with nutrition data from Open Food Facts.

    Uses normalized name matching to join the two datasets.
    """
    # Normalize product names for matching
    instacart_df = instacart_df.copy()
    off_df = off_df.copy()

    instacart_df["name_normalized"] = (
        instacart_df["product_name"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    off_df["name_normalized"] = (
        off_df["product_name"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
    )

    # Deduplicate OFF data — keep first match per normalized name
    off_dedup = off_df.drop_duplicates(subset=["name_normalized"], keep="first")

    # Exact match on normalized names
    merged = instacart_df.merge(
        off_dedup,
        on="name_normalized",
        how="left",
        suffixes=("", "_off"),
    )

    match_count = merged["brands"].notna().sum()
    logger.info(
        f"Matched {match_count}/{len(instacart_df)} products "
        f"({match_count / len(instacart_df) * 100:.1f}%) with nutrition data"
    )

    return merged


def build_product_catalog() -> pd.DataFrame:
    """Build the final product catalog by loading and enriching all data.

    Pipeline:
    1. Load Instacart products (49K real products)
    2. Load Open Food Facts (338K real products with nutrition)
    3. Merge on normalized product names
    4. Add popularity stats from real order history
    5. Clean and save

    Returns a DataFrame with columns:
        product_id, product_name, category, department, brand,
        ingredients, calories_100g, protein_100g, sugar_100g,
        fat_100g, fiber_100g, nutrition_grade,
        order_count, reorder_rate
    """
    # Step 1: Load Instacart products
    instacart_df = load_instacart_products()

    # Step 2: Load Open Food Facts
    off_df = load_open_food_facts()

    # Step 3: Enrich with nutrition data
    catalog = enrich_products(instacart_df, off_df)

    # Step 4: Add popularity from real order history
    logger.info("Loading order history for popularity stats...")
    orders = load_instacart_orders()
    popularity = compute_product_popularity(orders)
    catalog = catalog.merge(popularity, on="product_id", how="left")
    catalog["order_count"] = catalog["order_count"].fillna(0).astype(int)
    catalog["reorder_rate"] = catalog["reorder_rate"].fillna(0.0)

    # Step 5: Clean and rename columns
    catalog = catalog.rename(columns={
        "aisle": "category",
        "brands": "brand",
        "ingredients_text": "ingredients",
        "energy_100g": "calories_100g",
        "proteins_100g": "protein_100g",
        "sugars_100g": "sugar_100g",
        "nutrition_grade_fr": "nutrition_grade",
    })

    # Select final columns
    final_cols = [
        "product_id", "product_name", "category", "department", "brand",
        "ingredients", "calories_100g", "protein_100g", "sugar_100g",
        "fat_100g", "fiber_100g", "nutrition_grade", "allergens_en",
        "order_count", "reorder_rate",
    ]
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in catalog.columns]
    catalog = catalog[final_cols]

    # Remove duplicates (same product_id)
    catalog = catalog.drop_duplicates(subset=["product_id"], keep="first")

    # Save
    output_path = PROCESSED_DIR / "product_catalog.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output_path, index=False)

    logger.info(f"Final catalog: {len(catalog)} products saved to {output_path}")
    logger.info(f"  With nutrition data: {catalog['calories_100g'].notna().sum()}")
    logger.info(f"  With brand: {catalog['brand'].notna().sum()}")
    logger.info(f"  With ingredients: {catalog['ingredients'].notna().sum()}")

    return catalog


if __name__ == "__main__":
    catalog = build_product_catalog()
    print(f"\nCatalog built: {len(catalog)} products")
    print(catalog.head(10).to_string())
