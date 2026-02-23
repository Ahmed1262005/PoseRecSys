#!/usr/bin/env python3
"""
Find and remove truly miscategorized products from the catalog.

This script is SURGICAL — it only targets products that are genuinely
wrong (actual underwear tagged as Tops, pure thongs not bodysuits, etc.).
It does NOT touch legitimate fashion items like boxer shorts, thong
bodysuits, garter-knit cardigans, or boxer-waistband trousers.

Actions:
  (default)     Dry-run: list matches without changing anything
  --remove      Set in_stock=false in Supabase + delete from Algolia

Usage:
    PYTHONPATH=src python scripts/remove_miscategorized.py
    PYTHONPATH=src python scripts/remove_miscategorized.py --remove
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client


def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


# ── Detection rules ──────────────────────────────────────────────────
# Each rule: (search_keyword, is_miscategorized_fn)
# The function receives a product dict and returns True if it should be removed.

def _is_pure_underwear_boxers(p):
    """Actual boxers/underwear, NOT fashion boxer shorts or boxer-waistband pants.
    
    Conservative: only remove if name has NO fashion garment type at all.
    If the name contains any garment word (shorts, jeans, trouser, skirt, etc.),
    it's a fashion item that uses "boxer" as a style descriptor — KEEP it.
    """
    name = (p.get("name") or "").lower()
    # If name contains ANY garment/fashion word, it's a fashion item → KEEP
    fashion_words = [
        "short", "shorts", "jean", "jeans", "trouser", "trousers",
        "pant", "pants", "skirt", "skort", "dress", "top", "blouse",
        "set", "pajama", "pyjama", "sleep", "lounge",
        "waistband", "wasitband",  # typo variant in data
        "detail", "stripe waist",
    ]
    if any(w in name for w in fashion_words):
        return False
    # REMOVE: standalone boxer underwear with no fashion garment type
    # e.g. "Relaxed Fit Boxers", "Cotton Rib Boxer", "Boyfriend Boxer"
    return True


def _is_pure_thong(p):
    """Actual thong underwear, NOT thong bodysuits (which are legitimate tops)."""
    name = (p.get("name") or "").lower()
    # KEEP: anything with "bodysuit" — thong bodysuits are fashion items
    if "bodysuit" in name:
        return False
    # REMOVE: pure thongs — "Airbrush Invisible Thong", "Fits Everybody Thong", etc.
    return True


def _is_pure_shapewear(p):
    """Actual shapewear undergarments, NOT shapewear-style outerwear."""
    name = (p.get("name") or "").lower()
    # KEEP: shapewear dresses/bodysuits — they're worn as outerwear
    if any(w in name for w in ["dress", "bodysuit", "top"]):
        return False
    # REMOVE: shapewear shorts, capris, briefs, etc.
    return True


def _is_pure_underwear_generic(p):
    """Pure underwear items (briefs, panties, bralettes as standalone)."""
    return True  # If it matched the keyword and isn't in an intimates category, remove


def _is_pure_undershirt(p):
    """Nursing undershirts etc."""
    return True


# Don't flag these keywords — too many false positives with fashion items:
# "garter" (garter-knit cardigans, garter tanks)
# "chemise" (chemise dresses from Farm Rio, Lane Bryant)
# "bralette" (often worn as tops)
# "lingerie" (lingerie-inspired tops are fashion)

RULES = [
    ("boxer",      _is_pure_underwear_boxers),
    ("boxers",     _is_pure_underwear_boxers),
    ("thong",      _is_pure_thong),
    ("shapewear",  _is_pure_shapewear),
    ("briefs",     _is_pure_underwear_generic),
    ("panties",    _is_pure_underwear_generic),
    ("undershirt", _is_pure_undershirt),
    ("jockstrap",  _is_pure_underwear_generic),
]

# Categories where underwear keywords are expected
INTIMATES_CATEGORIES = {"intimates", "underwear", "lingerie", "sleepwear"}


def find_miscategorized(supabase):
    """Find truly miscategorized products using surgical rules."""
    matches = []
    seen = set()

    for keyword, check_fn in RULES:
        result = (
            supabase.table("products")
            .select("id, name, brand, category, broad_category, price, in_stock")
            .eq("in_stock", True)
            .ilike("name", f"%{keyword}%")
            .execute()
        )

        for product in (result.data or []):
            if product["id"] in seen:
                continue

            cat = (product.get("broad_category") or "").lower()
            # Skip if already in an intimates category
            if cat in INTIMATES_CATEGORIES:
                continue

            # Apply the specific rule
            if check_fn(product):
                seen.add(product["id"])
                matches.append({
                    "id": product["id"],
                    "name": product["name"],
                    "brand": product.get("brand", ""),
                    "category": product.get("category", ""),
                    "broad_category": product.get("broad_category", ""),
                    "price": product.get("price"),
                    "keyword": keyword,
                })

    return matches


def remove_products(supabase, products):
    """Set in_stock=false in Supabase and remove from Algolia."""
    ids = [p["id"] for p in products]

    if not ids:
        print("Nothing to remove.")
        return

    # 1. Update Supabase: set in_stock = false
    print(f"\nSetting in_stock=false for {len(ids)} products in Supabase...")
    for pid in ids:
        supabase.table("products").update({"in_stock": False}).eq("id", pid).execute()
    print(f"  Updated {len(ids)} products in Supabase.")

    # 2. Remove from Algolia
    try:
        from algoliasearch.search_client import SearchClientSync
        from config.settings import get_settings
        settings = get_settings()
        client = SearchClientSync(settings.algolia_app_id, settings.algolia_write_key)
        str_ids = [str(pid) for pid in ids]
        print(f"Removing {len(ids)} products from Algolia...")
        client.delete_objects(
            index_name=settings.algolia_index_name,
            object_ids=str_ids,
        )
        print(f"  Removed {len(ids)} products from Algolia.")
    except Exception as e:
        print(f"  WARNING: Could not remove from Algolia: {e}")
        print("  Products are marked out-of-stock but may still be in Algolia.")
        print("  Re-run indexing to sync: PYTHONPATH=src python scripts/index_to_algolia.py")


def main():
    parser = argparse.ArgumentParser(description="Find and remove miscategorized products")
    parser.add_argument("--remove", action="store_true", help="Actually remove (default is dry-run)")
    args = parser.parse_args()

    supabase = get_supabase()
    print("Searching for miscategorized products (surgical mode)...\n")

    matches = find_miscategorized(supabase)

    if not matches:
        print("No miscategorized products found.")
        return

    print(f"Found {len(matches)} miscategorized products:\n")
    print(f"{'Brand':<20} {'Keyword':<12} Name")
    print("-" * 80)
    for m in matches:
        print(f"{(m['brand'] or ''):<20} {m['keyword']:<12} {(m['name'] or '')[:50]}")

    if args.remove:
        confirm = input(f"\nRemove {len(matches)} products? (yes/no): ")
        if confirm.lower() == "yes":
            remove_products(supabase, matches)
            print("\nDone.")
        else:
            print("Cancelled.")
    else:
        print(f"\nDry run — no changes made. Use --remove to actually remove these products.")


if __name__ == "__main__":
    main()
