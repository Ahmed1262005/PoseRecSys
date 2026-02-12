#!/usr/bin/env python3
"""
One-time setup script: create Algolia virtual replica indices for sort-by.

Creates 3 virtual replicas on the primary index:
  - {index}_price_asc   — Price low to high
  - {index}_price_desc  — Price high to low
  - {index}_trending    — Trending score descending

Virtual replicas share the primary index data (no extra storage or
re-indexing). They only override customRanking.

Usage:
    PYTHONPATH=src python scripts/setup_algolia_replicas.py

Requires ALGOLIA_WRITE_KEY (not just SEARCH_KEY).
Safe to re-run — Algolia set_settings is idempotent.
"""

import sys
import os

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search.algolia_client import AlgoliaClient
from search.algolia_config import REPLICA_SUFFIXES


def main():
    print("Setting up Algolia virtual replicas for sort-by...")

    client = AlgoliaClient()
    print(f"Primary index: {client.index_name}")
    print(f"Replicas to create: {len(REPLICA_SUFFIXES)}")

    for suffix in REPLICA_SUFFIXES:
        print(f"  - {client.index_name}{suffix}")

    print()
    responses = client.configure_replicas()

    print(f"\nDone. Configured {len(responses)} virtual replicas.")

    # Verify by reading back settings from each replica
    print("\nVerification:")
    for suffix, expected_settings in REPLICA_SUFFIXES.items():
        replica_name = f"{client.index_name}{suffix}"
        try:
            settings = client.get_settings(index_name=replica_name)
            actual_ranking = settings.get("customRanking", [])
            expected_ranking = expected_settings["customRanking"]
            match = actual_ranking == expected_ranking
            status = "OK" if match else "MISMATCH"
            print(f"  {replica_name}: {status}")
            print(f"    customRanking: {actual_ranking}")
        except Exception as e:
            print(f"  {replica_name}: ERROR - {e}")

    client.close()


if __name__ == "__main__":
    main()
