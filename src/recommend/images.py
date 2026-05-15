"""Visual emoji icons for products.

Picks a single emoji that visually represents a grocery product. Resolution
priority is most-specific-first:

    1. KEYWORD_ICONS — match a substring in product_name (e.g. "banana" -> 🍌)
    2. CATEGORY_ICONS — match a substring in category (e.g. "yogurt" -> 🍦)
    3. DEPARTMENT_ICONS — match the department (e.g. "produce" -> 🥬)
    4. Fallback 📦 — generic box for products that don't match anything

The emoji can be rendered next to a product card so the UI stops looking
text-only when there is no real product photo available. Open Food Facts
matching by name is unreliable, so this emoji layer is the actual primary
visual representation.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

# ---------------------------------------------------------------------------
# Fallback emoji used when nothing matches. Centralised so build_emoji_map can
# treat this as "no specific match" for coverage stats.
# ---------------------------------------------------------------------------
FALLBACK_EMOJI: str = "📦"


# ---------------------------------------------------------------------------
# Department -> emoji. Keys are lowercased and match the `department` column
# values verbatim. Every department present in the real Instacart catalog is
# covered here so no product falls through to the box icon purely on the
# department layer.
# ---------------------------------------------------------------------------
DEPARTMENT_ICONS: dict[str, str] = {
    "produce": "🥬",
    "dairy eggs": "🥛",
    "frozen": "🧊",
    "beverages": "🧃",
    "snacks": "🍿",
    "pantry": "🥫",
    "deli": "🥪",
    "bakery": "🍞",
    "meat seafood": "🥩",
    "alcohol": "🍷",
    "household": "🧴",
    "personal care": "🧼",
    "babies": "🍼",
    "pets": "🐾",
    "breakfast": "🥣",
    "canned goods": "🥫",
    "dry goods pasta": "🍝",
    "international": "🌏",
    "missing": "📦",
    "other": "📦",
    "bulk": "🛍️",
}


# ---------------------------------------------------------------------------
# Category -> emoji. Matched as a substring of the lowercased category string,
# so "ice cream ice" matches "ice cream". Longer / more specific keys are
# preferred at lookup time, so e.g. "soft drinks" wins over "drinks".
# ---------------------------------------------------------------------------
CATEGORY_ICONS: dict[str, str] = {
    # Dairy
    "yogurt": "🍦",
    "cheese": "🧀",
    "milk": "🥛",
    "cream": "🥛",
    "butter": "🧈",
    "eggs": "🥚",
    "ice cream": "🍨",
    # Produce
    "fresh fruits": "🍎",
    "fresh vegetables": "🥦",
    "fresh herbs": "🌿",
    "packaged vegetables fruits": "🥗",
    "frozen produce": "🥦",
    "fruit vegetable snacks": "🍇",
    # Snacks
    "chips pretzels": "🍟",
    "chips": "🍟",
    "candy chocolate": "🍫",
    "candy": "🍬",
    "cookies cakes": "🍪",
    "cookies": "🍪",
    "crackers": "🍘",
    "popcorn jerky": "🍿",
    "nuts seeds dried fruit": "🥜",
    "energy granola bars": "🍫",
    # Bakery
    "bread": "🍞",
    "buns rolls": "🥐",
    "tortillas flat bread": "🌮",
    "bakery desserts": "🧁",
    # Beverages
    "water seltzer sparkling water": "💧",
    "water": "💧",
    "soft drinks": "🥤",
    "soda": "🥤",
    "juice nectars": "🧃",
    "juice": "🧃",
    "coffee": "☕",
    "tea": "🍵",
    "energy sports drinks": "🥤",
    "cocoa": "☕",
    # Alcohol
    "wine": "🍷",
    "beers coolers": "🍺",
    "spirits": "🥃",
    # Meat / seafood
    "poultry counter": "🍗",
    "meat counter": "🥩",
    "beef": "🥩",
    "hot dogs bacon sausage": "🌭",
    "lunch meat": "🥓",
    "seafood counter": "🐟",
    "fish seafood": "🐟",
    # Pasta / grains
    "dry pasta": "🍝",
    "pasta sauce": "🍝",
    "grains rice dried goods": "🍚",
    "cereal": "🥣",
    # Pantry
    "oils vinegars": "🫒",
    "spices seasonings": "🧂",
    "condiments": "🥫",
    "soup broth bouillon": "🍲",
    "canned meals beans": "🥫",
    "canned jarred vegetables": "🥫",
    "canned fruit applesauce": "🥫",
    "pickled goods olives": "🫒",
    "baking ingredients": "🧁",
    "spreads": "🍯",
    "honeys syrups nectars": "🍯",
    # Frozen
    "frozen meals": "🍱",
    "frozen pizza": "🍕",
    "frozen appetizers sides": "🍤",
    "frozen breakfast": "🥞",
    "frozen meat seafood": "🐟",
    "frozen dessert": "🍦",
    # Refrigerated / deli
    "refrigerated": "🧀",
    "prepared meals": "🍱",
    "prepared soups salads": "🥗",
    # International
    "asian foods": "🍜",
    "indian foods": "🍛",
    "mexican foods": "🌮",
    # Baby
    "baby food formula": "🍼",
    "baby accessories": "🍼",
    "diapers wipes": "🍼",
    # Pets
    "cat food care": "🐱",
    "dog food care": "🐶",
    # Personal / household
    "oral hygiene": "🪥",
    "hair care": "💇",
    "skin care": "🧴",
    "body lotions soap": "🧼",
    "soap": "🧼",
    "shave needs": "🪒",
    "vitamins supplements": "💊",
    "cold flu allergy": "💊",
    "first aid": "🩹",
    "feminine care": "🩷",
    "digestion": "💊",
    "cleaning products": "🧽",
    "laundry": "🧺",
    "paper goods": "🧻",
    "trash bags liners": "🗑️",
    "air fresheners candles": "🕯️",
    "kitchen supplies": "🍴",
    "plates bowls cups flatware": "🍽️",
    # Other common
    "salad dressing toppings": "🥗",
    "marinades meat preparation": "🍖",
    "doughs gelatins bake mixes": "🥧",
    "instant foods": "🍜",
    "tofu meat alternatives": "🥡",
}


# ---------------------------------------------------------------------------
# Keyword -> emoji. Most specific layer: matched as a whole-word substring of
# the lowercased product_name. Whole-word matching ("apple" doesn't match
# "pineapple") avoids the obvious false positives.
# ---------------------------------------------------------------------------
KEYWORD_ICONS: dict[str, str] = {
    # Fruits
    "banana": "🍌",
    "bananas": "🍌",
    "apple": "🍎",
    "apples": "🍎",
    "strawberry": "🍓",
    "strawberries": "🍓",
    "blueberry": "🫐",
    "blueberries": "🫐",
    "raspberry": "🍓",
    "raspberries": "🍓",
    "blackberry": "🫐",
    "blackberries": "🫐",
    "orange": "🍊",
    "oranges": "🍊",
    "lemon": "🍋",
    "lemons": "🍋",
    "lime": "🍋",
    "limes": "🍋",
    "avocado": "🥑",
    "avocados": "🥑",
    "watermelon": "🍉",
    "grape": "🍇",
    "grapes": "🍇",
    "peach": "🍑",
    "peaches": "🍑",
    "mango": "🥭",
    "mangoes": "🥭",
    "pineapple": "🍍",
    "kiwi": "🥝",
    "cherry": "🍒",
    "cherries": "🍒",
    "pear": "🍐",
    "pears": "🍐",
    "coconut": "🥥",
    "melon": "🍈",
    # Vegetables
    "broccoli": "🥦",
    "carrot": "🥕",
    "carrots": "🥕",
    "tomato": "🍅",
    "tomatoes": "🍅",
    "potato": "🥔",
    "potatoes": "🥔",
    "garlic": "🧄",
    "onion": "🧅",
    "onions": "🧅",
    "mushroom": "🍄",
    "mushrooms": "🍄",
    "pepper": "🌶️",
    "peppers": "🌶️",
    "corn": "🌽",
    "cucumber": "🥒",
    "cucumbers": "🥒",
    "eggplant": "🍆",
    "lettuce": "🥬",
    "spinach": "🥬",
    "kale": "🥬",
    "cabbage": "🥬",
    "zucchini": "🥒",
    "celery": "🥬",
    "asparagus": "🥬",
    # Pantry / specific food items
    "egg": "🥚",
    "eggs": "🥚",
    "milk": "🥛",
    "cheese": "🧀",
    "butter": "🧈",
    "yogurt": "🍦",
    "bread": "🍞",
    "bagel": "🥯",
    "bagels": "🥯",
    "croissant": "🥐",
    "donut": "🍩",
    "donuts": "🍩",
    "pretzel": "🥨",
    "pretzels": "🥨",
    "cookie": "🍪",
    "cookies": "🍪",
    "cake": "🍰",
    "pie": "🥧",
    "chocolate": "🍫",
    "candy": "🍬",
    "honey": "🍯",
    "popcorn": "🍿",
    "chips": "🍟",
    "rice": "🍚",
    "pasta": "🍝",
    "noodle": "🍜",
    "noodles": "🍜",
    "ramen": "🍜",
    "pizza": "🍕",
    "burger": "🍔",
    "taco": "🌮",
    "burrito": "🌯",
    "sushi": "🍣",
    "sandwich": "🥪",
    "salad": "🥗",
    "soup": "🍲",
    "stew": "🍲",
    "curry": "🍛",
    "ice cream": "🍨",
    # Proteins
    "chicken": "🍗",
    "beef": "🥩",
    "steak": "🥩",
    "pork": "🥓",
    "bacon": "🥓",
    "ham": "🥓",
    "sausage": "🌭",
    "hot dog": "🌭",
    "hotdog": "🌭",
    "turkey": "🦃",
    "lamb": "🍖",
    "fish": "🐟",
    "salmon": "🐟",
    "tuna": "🐟",
    "shrimp": "🦐",
    "lobster": "🦞",
    "crab": "🦀",
    "tofu": "🥡",
    # Beverages
    "coffee": "☕",
    "espresso": "☕",
    "latte": "☕",
    "tea": "🍵",
    "water": "💧",
    "soda": "🥤",
    "cola": "🥤",
    "juice": "🧃",
    "smoothie": "🥤",
    "wine": "🍷",
    "beer": "🍺",
    "vodka": "🍸",
    "whiskey": "🥃",
    "whisky": "🥃",
    "rum": "🥃",
    "tequila": "🥃",
    "champagne": "🍾",
    # Pantry / oils / sweets
    "oil": "🫒",
    "olive oil": "🫒",
    "vinegar": "🧪",
    "salt": "🧂",
    "sugar": "🍬",
    "flour": "🌾",
    "jam": "🍓",
    "jelly": "🍇",
    "syrup": "🍯",
    "ketchup": "🍅",
    "mustard": "🌭",
    "mayonnaise": "🥚",
    # Personal / household markers (rarer in product names but useful)
    "soap": "🧼",
    "shampoo": "🧴",
    "toothpaste": "🪥",
    "diaper": "🍼",
    "diapers": "🍼",
}


# Categories are scanned longest-key-first so that a more specific match wins
# over a more generic one ("soft drinks" before "drinks"). Pre-sort once at
# import time so per-product lookup stays cheap.
_CATEGORY_KEYS_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(CATEGORY_ICONS.keys(), key=len, reverse=True)
)
_KEYWORD_KEYS_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(KEYWORD_ICONS.keys(), key=len, reverse=True)
)


def _normalise(value: object) -> str:
    """Lowercase and strip a value; None / NaN safely becomes an empty string.

    Pandas can hand us NaN floats for missing cells, and the catalog uses the
    literal string "missing" in some columns. Both cases need to collapse to
    something that won't match any emoji key.
    """
    if value is None:
        return ""
    # NaN floats compare unequal to themselves; cheaper than importing math.
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip().lower()


def _word_in(text: str, keyword: str) -> bool:
    """Return True if `keyword` appears as a whole-word substring of `text`.

    "apple" should match "apple juice" and "honeycrisp apple" but NOT
    "pineapple". For multi-word keywords like "ice cream" we just check
    substring presence — there is no realistic ambiguity there.
    """
    if " " in keyword:
        return keyword in text
    # Whole-word check: keyword must be bounded by non-alphanumeric chars or
    # by the start / end of the string.
    idx = 0
    klen = len(keyword)
    tlen = len(text)
    while idx <= tlen - klen:
        found = text.find(keyword, idx)
        if found == -1:
            return False
        before_ok = found == 0 or not text[found - 1].isalnum()
        after = found + klen
        after_ok = after == tlen or not text[after].isalnum()
        if before_ok and after_ok:
            return True
        idx = found + 1
    return False


def get_emoji_for_product(product: Mapping[str, object]) -> str:
    """Return a single emoji that visually represents the product.

    Priority order:
        1. Match keyword from product_name (e.g. "banana" -> 🍌, "milk" -> 🥛)
        2. Match category (e.g. category contains "yogurt" -> 🍦)
        3. Match department (e.g. "produce" -> 🥬, "frozen" -> 🧊)
        4. Fallback: 📦 (generic box)

    Always returns a non-empty emoji string. Never returns None.
    """
    name = _normalise(product.get("product_name"))
    category = _normalise(product.get("category"))
    department = _normalise(product.get("department"))

    # 1. Keyword match against the product name. Longest keys first so
    #    "ice cream" beats "cream".
    if name:
        for keyword in _KEYWORD_KEYS_BY_LENGTH:
            if _word_in(name, keyword):
                return KEYWORD_ICONS[keyword]

    # 2. Category match. Substring is OK here because category strings are
    #    short and curated ("ice cream ice", "fresh fruits", etc.).
    if category:
        for key in _CATEGORY_KEYS_BY_LENGTH:
            if key in category:
                return CATEGORY_ICONS[key]

    # 3. Department match — exact lookup, since departments are a closed set.
    if department and department in DEPARTMENT_ICONS:
        return DEPARTMENT_ICONS[department]

    # 4. Generic fallback.
    return FALLBACK_EMOJI


def build_emoji_map(catalog: pd.DataFrame) -> dict[int, str]:
    """Run `get_emoji_for_product` over the whole catalog.

    Returns a dict mapping product_id -> emoji. Uses raw column arrays rather
    than DataFrame.iterrows / .to_dict so we stay fast on the full 50k catalog
    (under a second on a modern laptop).
    """
    if catalog.empty:
        return {}

    # Pull the columns we care about up-front. Missing columns fall back to
    # arrays of None so a slimmed-down catalog still works.
    product_ids = catalog["product_id"].to_numpy()
    names = (
        catalog["product_name"].to_numpy()
        if "product_name" in catalog.columns
        else [None] * len(catalog)
    )
    categories = (
        catalog["category"].to_numpy()
        if "category" in catalog.columns
        else [None] * len(catalog)
    )
    departments = (
        catalog["department"].to_numpy()
        if "department" in catalog.columns
        else [None] * len(catalog)
    )

    result: dict[int, str] = {}
    for pid, name, category, department in zip(
        product_ids, names, categories, departments
    ):
        result[int(pid)] = get_emoji_for_product(
            {
                "product_name": name,
                "category": category,
                "department": department,
            }
        )
    return result
