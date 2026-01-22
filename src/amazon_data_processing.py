"""
Amazon Fashion Data Processing Module
Downloads, processes, and converts Amazon Reviews 2018 (Clothing, Shoes & Jewelry) to RecBole format

Dataset: Amazon Reviews 2018 - Clothing, Shoes and Jewelry
Source: https://nijianmo.github.io/amazon/

Usage:
    # Process reviews only (no metadata filtering)
    python amazon_data_processing.py --reviews data/amazon_fashion/raw/Clothing_Shoes_and_Jewelry_5.json.gz

    # Process with metadata for men's filtering
    python amazon_data_processing.py \
        --reviews data/amazon_fashion/raw/Clothing_Shoes_and_Jewelry_5.json.gz \
        --metadata data/amazon_fashion/raw/meta_Clothing_Shoes_and_Jewelry.json.gz \
        --filter-mens

    # Download images after processing
    python amazon_data_processing.py --download-images --output data/amazon_fashion
"""

import gzip
import json
import os
import sys
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Generator, Tuple
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


# ============================================================================
# Data Loading Functions
# ============================================================================

def parse_gzip_json(path: str) -> Generator[dict, None, None]:
    """
    Parse gzipped JSON file (one JSON object per line)

    Args:
        path: Path to .json.gz file

    Yields:
        Parsed JSON objects
    """
    with gzip.open(path, 'rb') as f:
        for line in f:
            try:
                yield json.loads(line.decode('utf-8'))
            except json.JSONDecodeError:
                continue


def load_amazon_reviews(path: str, show_progress: bool = True) -> List[dict]:
    """
    Load Amazon reviews from gzipped JSON file

    Args:
        path: Path to reviews .json.gz file

    Returns:
        List of review dictionaries with keys:
        - reviewerID: User ID
        - asin: Product ID
        - overall: Rating (1-5)
        - unixReviewTime: Timestamp
        - reviewText: Review text (optional)
        - summary: Review summary (optional)
    """
    reviews = []
    print(f"Loading reviews from {path}...")

    # Count lines first for progress bar
    if show_progress:
        with gzip.open(path, 'rb') as f:
            total = sum(1 for _ in f)
        iterator = tqdm(parse_gzip_json(path), total=total, desc="Loading reviews")
    else:
        iterator = parse_gzip_json(path)

    for review in iterator:
        # Extract only essential fields
        reviews.append({
            'reviewerID': review.get('reviewerID'),
            'asin': review.get('asin'),
            'overall': review.get('overall'),
            'unixReviewTime': review.get('unixReviewTime', 0),
        })

    print(f"Loaded {len(reviews):,} reviews")
    return reviews


def load_amazon_metadata(path: str, show_progress: bool = True) -> Dict[str, dict]:
    """
    Load Amazon product metadata from gzipped JSON file

    Args:
        path: Path to metadata .json.gz file

    Returns:
        Dictionary mapping asin to metadata dict with keys:
        - asin: Product ID
        - category: List of category hierarchy
        - image: List of image URLs
        - title: Product title
        - brand: Product brand
    """
    metadata = {}
    print(f"Loading metadata from {path}...")

    if show_progress:
        with gzip.open(path, 'rb') as f:
            total = sum(1 for _ in f)
        iterator = tqdm(parse_gzip_json(path), total=total, desc="Loading metadata")
    else:
        iterator = parse_gzip_json(path)

    for item in iterator:
        asin = item.get('asin')
        if asin:
            metadata[asin] = {
                'asin': asin,
                'category': item.get('category', []),
                'image': item.get('image', []),
                'title': item.get('title', ''),
                'brand': item.get('brand', ''),
            }

    print(f"Loaded metadata for {len(metadata):,} products")
    return metadata


# ============================================================================
# Filtering Functions
# ============================================================================

def filter_mens_items(metadata: Dict[str, dict]) -> Set[str]:
    """
    Filter items that belong to men's clothing categories

    Args:
        metadata: Dictionary mapping asin to metadata

    Returns:
        Set of asin IDs for men's items
    """
    mens_keywords = [
        "Men",
        "Men's",
        "Mens",
        "Boy",
        "Boys",
        "Male",
    ]

    # Exclude women's items that might have "men" in description
    exclude_keywords = [
        "Women",
        "Women's",
        "Womens",
        "Girl",
        "Girls",
        "Female",
    ]

    mens_items = set()

    print("Filtering for men's items...")
    for asin, item in tqdm(metadata.items(), desc="Filtering"):
        categories = item.get('category', [])
        category_str = ' '.join(str(c) for c in categories).lower()

        # Check for men's keywords
        is_mens = any(keyword.lower() in category_str for keyword in mens_keywords)

        # Exclude if it's also in women's category
        is_womens = any(keyword.lower() in category_str for keyword in exclude_keywords)

        if is_mens and not is_womens:
            mens_items.add(asin)

    print(f"Found {len(mens_items):,} men's items")
    return mens_items


def filter_reviews_by_items(reviews: List[dict], valid_items: Set[str]) -> List[dict]:
    """
    Filter reviews to only include reviews for valid items

    Args:
        reviews: List of review dictionaries
        valid_items: Set of valid asin IDs

    Returns:
        Filtered list of reviews
    """
    print(f"Filtering reviews for {len(valid_items):,} valid items...")
    filtered = [r for r in tqdm(reviews, desc="Filtering reviews") if r['asin'] in valid_items]
    print(f"Filtered to {len(filtered):,} reviews (from {len(reviews):,})")
    return filtered


def apply_k_core_filter(reviews: List[dict], k: int = 5) -> Tuple[List[dict], Set[str], Set[str]]:
    """
    Apply k-core filtering: keep only users and items with >= k interactions

    Args:
        reviews: List of review dictionaries
        k: Minimum number of interactions

    Returns:
        Tuple of (filtered_reviews, valid_users, valid_items)
    """
    print(f"Applying {k}-core filtering...")

    prev_count = len(reviews) + 1
    current_reviews = reviews
    iteration = 0

    while len(current_reviews) < prev_count:
        iteration += 1
        prev_count = len(current_reviews)

        # Count user and item frequencies
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)

        for r in current_reviews:
            user_counts[r['reviewerID']] += 1
            item_counts[r['asin']] += 1

        # Filter users and items with >= k interactions
        valid_users = {u for u, c in user_counts.items() if c >= k}
        valid_items = {i for i, c in item_counts.items() if c >= k}

        # Filter reviews
        current_reviews = [
            r for r in current_reviews
            if r['reviewerID'] in valid_users and r['asin'] in valid_items
        ]

        print(f"  Iteration {iteration}: {len(current_reviews):,} reviews, "
              f"{len(valid_users):,} users, {len(valid_items):,} items")

    final_users = {r['reviewerID'] for r in current_reviews}
    final_items = {r['asin'] for r in current_reviews}

    print(f"Final: {len(current_reviews):,} reviews, "
          f"{len(final_users):,} users, {len(final_items):,} items")

    return current_reviews, final_users, final_items


# ============================================================================
# RecBole Conversion
# ============================================================================

def convert_to_recbole(
    reviews: List[dict],
    output_dir: str,
    dataset_name: str = "amazon_mens"
) -> None:
    """
    Convert reviews to RecBole atomic file format

    Creates:
    - {dataset_name}.inter: Interaction file
    - {dataset_name}.item: Item metadata file (if metadata available)

    Args:
        reviews: List of review dictionaries
        output_dir: Output directory for RecBole files
        dataset_name: Name of the dataset
    """
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Create .inter file
    inter_file = output_path / f"{dataset_name}.inter"
    print(f"Writing interactions to {inter_file}...")

    with open(inter_file, 'w') as f:
        # Header with field types
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")

        # Data rows
        for review in tqdm(reviews, desc="Writing .inter"):
            user_id = review['reviewerID']
            item_id = review['asin']
            timestamp = review.get('unixReviewTime', 0)
            f.write(f"{user_id}\t{item_id}\t{timestamp}\n")

    print(f"Saved {len(reviews):,} interactions to {inter_file}")


def create_item_file(
    items: Set[str],
    metadata: Optional[Dict[str, dict]],
    output_dir: str,
    dataset_name: str = "amazon_mens"
) -> None:
    """
    Create RecBole .item file with item metadata

    Args:
        items: Set of item asin IDs
        metadata: Optional metadata dictionary
        output_dir: Output directory
        dataset_name: Name of the dataset
    """
    output_path = Path(output_dir) / dataset_name
    item_file = output_path / f"{dataset_name}.item"

    print(f"Writing item metadata to {item_file}...")

    with open(item_file, 'w') as f:
        # Header
        f.write("item_id:token\ttitle:token_seq\tcategory:token\n")

        # Data rows
        for asin in tqdm(items, desc="Writing .item"):
            if metadata and asin in metadata:
                meta = metadata[asin]
                title = meta.get('title', '').replace('\t', ' ').replace('\n', ' ')[:100]
                # Get last category in hierarchy (most specific)
                categories = meta.get('category', [])
                category = categories[-1] if categories else 'unknown'
                category = str(category).replace('\t', ' ')
            else:
                title = asin
                category = 'unknown'

            f.write(f"{asin}\t{title}\t{category}\n")

    print(f"Saved {len(items):,} items to {item_file}")


# ============================================================================
# Image Downloading
# ============================================================================

async def download_single_image(
    session: aiohttp.ClientSession,
    asin: str,
    url: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore
) -> bool:
    """
    Download a single image

    Args:
        session: aiohttp session
        asin: Product ID
        url: Image URL
        output_dir: Output directory
        semaphore: Semaphore for rate limiting

    Returns:
        True if successful, False otherwise
    """
    output_file = output_dir / f"{asin}.jpg"

    if output_file.exists():
        return True

    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(output_file, 'wb') as f:
                        f.write(content)
                    return True
        except Exception:
            pass

    return False


async def download_images_async(
    image_urls: Dict[str, List[str]],
    output_dir: str,
    max_concurrent: int = 50
) -> Tuple[int, int]:
    """
    Download images asynchronously

    Args:
        image_urls: Dictionary mapping asin to list of image URLs
        output_dir: Output directory for images
        max_concurrent: Maximum concurrent downloads

    Returns:
        Tuple of (successful_count, failed_count)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    # Flatten to list of (asin, url) pairs, using first image URL only
    tasks_data = [
        (asin, urls[0])
        for asin, urls in image_urls.items()
        if urls and len(urls) > 0
    ]

    print(f"Downloading {len(tasks_data):,} images...")

    successful = 0
    failed = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_single_image(session, asin, url, output_path, semaphore)
            for asin, url in tasks_data
        ]

        with tqdm(total=len(tasks), desc="Downloading images") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"Downloaded {successful:,} images, {failed:,} failed")
    return successful, failed


def extract_image_urls(
    metadata: Dict[str, dict],
    valid_items: Set[str]
) -> Dict[str, List[str]]:
    """
    Extract image URLs for valid items

    Args:
        metadata: Metadata dictionary
        valid_items: Set of valid item asin IDs

    Returns:
        Dictionary mapping asin to list of image URLs
    """
    image_urls = {}

    for asin in valid_items:
        if asin in metadata:
            urls = metadata[asin].get('image', [])
            if urls:
                image_urls[asin] = urls

    print(f"Found image URLs for {len(image_urls):,} items")
    return image_urls


# ============================================================================
# Statistics and Reporting
# ============================================================================

def compute_statistics(reviews: List[dict]) -> dict:
    """
    Compute dataset statistics

    Args:
        reviews: List of review dictionaries

    Returns:
        Statistics dictionary
    """
    users = set()
    items = set()

    for r in reviews:
        users.add(r['reviewerID'])
        items.add(r['asin'])

    stats = {
        'total_reviews': len(reviews),
        'total_users': len(users),
        'total_items': len(items),
        'avg_reviews_per_user': len(reviews) / len(users) if users else 0,
        'avg_reviews_per_item': len(reviews) / len(items) if items else 0,
        'sparsity': 1 - (len(reviews) / (len(users) * len(items))) if users and items else 0,
    }

    return stats


def print_statistics(stats: dict, title: str = "Dataset Statistics") -> None:
    """
    Print formatted statistics
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f"  Total reviews:      {stats['total_reviews']:,}")
    print(f"  Total users:        {stats['total_users']:,}")
    print(f"  Total items:        {stats['total_items']:,}")
    print(f"  Avg reviews/user:   {stats['avg_reviews_per_user']:.2f}")
    print(f"  Avg reviews/item:   {stats['avg_reviews_per_item']:.2f}")
    print(f"  Sparsity:           {stats['sparsity']*100:.4f}%")
    print(f"{'='*50}\n")


# ============================================================================
# Main Pipeline
# ============================================================================

def process_amazon_data(
    reviews_path: str,
    metadata_path: Optional[str] = None,
    output_dir: str = "data/amazon_fashion",
    dataset_name: str = "amazon_mens",
    filter_mens: bool = True,
    k_core: int = 5,
    download_images: bool = False,
    max_concurrent_downloads: int = 50
) -> dict:
    """
    Main processing pipeline

    Args:
        reviews_path: Path to reviews .json.gz file
        metadata_path: Optional path to metadata .json.gz file
        output_dir: Output directory
        dataset_name: Name for the dataset
        filter_mens: Whether to filter for men's items
        k_core: K-core filtering threshold
        download_images: Whether to download product images
        max_concurrent_downloads: Max concurrent image downloads

    Returns:
        Statistics dictionary
    """
    # Load reviews
    reviews = load_amazon_reviews(reviews_path)

    # Print initial stats
    initial_stats = compute_statistics(reviews)
    print_statistics(initial_stats, "Initial Dataset Statistics")

    # Load metadata if provided
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        metadata = load_amazon_metadata(metadata_path)

    # Filter for men's items if requested and metadata available
    if filter_mens and metadata:
        mens_items = filter_mens_items(metadata)
        reviews = filter_reviews_by_items(reviews, mens_items)

        filtered_stats = compute_statistics(reviews)
        print_statistics(filtered_stats, "After Men's Filtering")

    # Apply k-core filtering
    if k_core > 0:
        reviews, valid_users, valid_items = apply_k_core_filter(reviews, k_core)
    else:
        valid_users = {r['reviewerID'] for r in reviews}
        valid_items = {r['asin'] for r in reviews}

    # Final stats
    final_stats = compute_statistics(reviews)
    print_statistics(final_stats, "Final Dataset Statistics")

    # Convert to RecBole format
    recbole_dir = os.path.join(output_dir, "recbole")
    convert_to_recbole(reviews, recbole_dir, dataset_name)

    # Create item file if metadata available
    if metadata:
        create_item_file(valid_items, metadata, recbole_dir, dataset_name)

    # Save statistics
    stats_file = os.path.join(output_dir, "processed", "stats.json")
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")

    # Save valid items list
    items_file = os.path.join(output_dir, "processed", "valid_items.json")
    with open(items_file, 'w') as f:
        json.dump(list(valid_items), f)
    print(f"Saved {len(valid_items):,} valid items to {items_file}")

    # Download images if requested
    if download_images and metadata:
        image_urls = extract_image_urls(metadata, valid_items)

        # Save image URLs for later
        urls_file = os.path.join(output_dir, "processed", "image_urls.json")
        with open(urls_file, 'w') as f:
            json.dump(image_urls, f)
        print(f"Saved image URLs to {urls_file}")

        # Download
        images_dir = os.path.join(output_dir, "images")
        asyncio.run(download_images_async(image_urls, images_dir, max_concurrent_downloads))

    return final_stats


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process Amazon Fashion dataset for DuoRec training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing (no metadata)
  python amazon_data_processing.py \\
      --reviews data/amazon_fashion/raw/Clothing_Shoes_and_Jewelry_5.json.gz

  # Process with men's filtering
  python amazon_data_processing.py \\
      --reviews data/amazon_fashion/raw/Clothing_Shoes_and_Jewelry_5.json.gz \\
      --metadata data/amazon_fashion/raw/meta_Clothing_Shoes_and_Jewelry.json.gz \\
      --filter-mens

  # Download images
  python amazon_data_processing.py \\
      --reviews data/amazon_fashion/raw/Clothing_Shoes_and_Jewelry_5.json.gz \\
      --metadata data/amazon_fashion/raw/meta_Clothing_Shoes_and_Jewelry.json.gz \\
      --filter-mens --download-images
        """
    )

    parser.add_argument(
        '--reviews', '-r',
        required=True,
        help='Path to reviews .json.gz file'
    )
    parser.add_argument(
        '--metadata', '-m',
        help='Path to metadata .json.gz file (optional, required for filtering and images)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/amazon_fashion',
        help='Output directory (default: data/amazon_fashion)'
    )
    parser.add_argument(
        '--dataset-name', '-n',
        default='amazon_mens',
        help='Dataset name (default: amazon_mens)'
    )
    parser.add_argument(
        '--filter-mens',
        action='store_true',
        help="Filter for men's clothing only (requires metadata)"
    )
    parser.add_argument(
        '--no-filter-mens',
        action='store_true',
        help="Don't filter for men's clothing (use all items)"
    )
    parser.add_argument(
        '--k-core', '-k',
        type=int,
        default=5,
        help='K-core filtering threshold (default: 5, use 0 to disable)'
    )
    parser.add_argument(
        '--download-images',
        action='store_true',
        help='Download product images (requires metadata)'
    )
    parser.add_argument(
        '--max-concurrent', '-c',
        type=int,
        default=50,
        help='Max concurrent image downloads (default: 50)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.reviews):
        print(f"Error: Reviews file not found: {args.reviews}")
        sys.exit(1)

    if args.metadata and not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    if args.filter_mens and not args.metadata:
        print("Error: --filter-mens requires --metadata")
        sys.exit(1)

    if args.download_images and not args.metadata:
        print("Error: --download-images requires --metadata")
        sys.exit(1)

    # Determine filtering
    filter_mens = args.filter_mens and not args.no_filter_mens

    # Run processing
    process_amazon_data(
        reviews_path=args.reviews,
        metadata_path=args.metadata,
        output_dir=args.output,
        dataset_name=args.dataset_name,
        filter_mens=filter_mens,
        k_core=args.k_core,
        download_images=args.download_images,
        max_concurrent_downloads=args.max_concurrent
    )


if __name__ == "__main__":
    main()
