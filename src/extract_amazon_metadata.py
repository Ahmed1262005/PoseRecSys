"""
Extract metadata for Amazon Fashion items that have embeddings.
Creates a lookup file for the API.
"""

import json
import gzip
import os
import pickle
from pathlib import Path

def main():
    project_root = "/home/ubuntu/recSys/outfitTransformer"

    # Load items that have embeddings
    embeddings_path = f"{project_root}/data/amazon_fashion/processed/amazon_mens_embeddings.pkl"
    print(f"Loading embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    item_ids = set(embeddings.keys())
    print(f"Found {len(item_ids)} items with embeddings")

    # Parse metadata file and extract info for our items
    meta_path = f"{project_root}/data/amazon_fashion/raw/meta_Clothing_Shoes_and_Jewelry.json.gz"
    print(f"Parsing metadata from {meta_path}...")

    item_metadata = {}
    found = 0
    processed = 0

    with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
        for line in f:
            processed += 1
            if processed % 500000 == 0:
                print(f"  Processed {processed:,} lines, found {found} matches...")

            try:
                data = json.loads(line)
                asin = data.get('asin', '')

                if asin in item_ids:
                    # Extract relevant fields
                    item_metadata[asin] = {
                        'title': data.get('title', ''),
                        'brand': data.get('brand', ''),
                        'category': data.get('category', []),
                        'price': data.get('price', ''),
                        'description': data.get('description', [''])[0] if data.get('description') else ''
                    }
                    found += 1

                    if found >= len(item_ids):
                        print(f"Found all {found} items!")
                        break

            except json.JSONDecodeError:
                continue

    print(f"Extracted metadata for {len(item_metadata)} items")

    # Save metadata
    output_path = f"{project_root}/data/amazon_fashion/processed/item_metadata.json"
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(item_metadata, f)

    # Also save as pickle for faster loading
    output_pkl = f"{project_root}/data/amazon_fashion/processed/item_metadata.pkl"
    with open(output_pkl, 'wb') as f:
        pickle.dump(item_metadata, f)

    print(f"Done! Saved {len(item_metadata)} items")

    # Show some examples
    print("\nSample items:")
    for i, (asin, meta) in enumerate(list(item_metadata.items())[:3]):
        print(f"\n{asin}:")
        print(f"  Title: {meta['title'][:80]}...")
        print(f"  Brand: {meta['brand']}")
        print(f"  Category: {meta['category'][:3] if meta['category'] else 'N/A'}")

if __name__ == "__main__":
    main()
