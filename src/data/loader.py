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


def load_open_food_facts(country: str = "united-states") -> pd.DataFrame:
    """Load nutrition and ingredient data from Open Food Facts.

    Expected file in data/raw/openfoodfacts/:
        - en.openfoodfacts.org.products.csv (or filtered subset)
    """
    off_dir = RAW_DIR / "openfoodfacts"
    filepath = off_dir / "products.csv"

    # Open Food Facts has many columns — load only what we need
    usecols = [
        "code", "product_name", "brands", "categories_en",
        "ingredients_text", "allergens_en",
        "energy-kcal_100g", "proteins_100g", "sugars_100g",
        "fat_100g", "fiber_100g", "sodium_100g",
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

    Uses fuzzy name matching to join the two datasets.
    """
    # Normalize product names for matching
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

    # Exact match first
    merged = instacart_df.merge(
        off_df,
        on="name_normalized",
        how="left",
        suffixes=("", "_off"),
    )

    match_count = merged["code"].notna().sum()
    logger.info(
        f"Matched {match_count}/{len(instacart_df)} products "
        f"({match_count / len(instacart_df) * 100:.1f}%)"
    )

    return merged


def build_product_catalog() -> pd.DataFrame:
    """Build the final product catalog by loading and enriching data.

    Returns a DataFrame with columns:
        product_id, product_name, aisle, department, brand,
        ingredients, calories_100g, protein_100g, sugar_100g,
        fat_100g, fiber_100g, nutrition_grade
    """
    instacart_df = load_instacart_products()
    off_df = load_open_food_facts()
    catalog = enrich_products(instacart_df, off_df)

    # Select and rename final columns
    catalog = catalog.rename(columns={
        "aisle": "category",
        "department": "department",
        "brands": "brand",
        "ingredients_text": "ingredients",
        "energy-kcal_100g": "calories_100g",
        "proteins_100g": "protein_100g",
        "sugars_100g": "sugar_100g",
        "nutrition_grade_fr": "nutrition_grade",
    })

    output_path = PROCESSED_DIR / "product_catalog.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output_path, index=False)
    logger.info(f"Saved product catalog to {output_path}")

    return catalog
