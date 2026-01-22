"""
Amazon Product Image Downloader and FashionCLIP Embedding Generator

Downloads product images for Amazon Fashion items and generates
FashionCLIP embeddings for use in recommendation models.
"""

import os
import re
import json
import gzip
import asyncio
import aiohttp
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pickle
import torch
from concurrent.futures import ThreadPoolExecutor


def convert_to_large_image_url(url: str, size: int = 500) -> str:
    """
    Convert Amazon thumbnail URL to larger image URL.

    Amazon URLs contain size codes like _SR38,50_ or _SS40_
    We replace these with _SL{size}_ for larger images.
    """
    # Pattern to match Amazon image size codes
    patterns = [
        r'_S[A-Z]\d+[,_]\d*_',  # _SR38,50_ or _SX300_
        r'_S[A-Z]\d+_',         # _SS40_ or _SL500_
        r'\._[^.]+_\.',         # ._AC_SL1500_.
    ]

    new_url = url
    for pattern in patterns:
        new_url = re.sub(pattern, f'_SL{size}_', new_url)

    return new_url


def extract_image_urls(metadata_path: str, valid_items_path: str, output_path: str):
    """
    Extract image URLs for all valid items from metadata.
    """
    print("Loading valid items...")
    with open(valid_items_path, 'r') as f:
        valid_items = set(json.load(f))

    print(f"Valid items: {len(valid_items)}")
    print("Extracting image URLs from metadata...")

    image_urls = {}
    items_without_images = []

    with gzip.open(metadata_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing metadata"):
            try:
                item = json.loads(line)
                asin = item.get('asin', '')

                if asin not in valid_items:
                    continue

                # Try different image fields
                images = (
                    item.get('imageURLHighRes') or
                    item.get('imageURL') or
                    item.get('image') or
                    []
                )

                if images:
                    # Get first image and convert to large size
                    img_url = images[0] if isinstance(images, list) else images
                    img_url = convert_to_large_image_url(img_url)
                    image_urls[asin] = img_url
                else:
                    items_without_images.append(asin)

            except Exception as e:
                continue

    print(f"\nItems with images: {len(image_urls)}")
    print(f"Items without images: {len(items_without_images)}")

    # Save URL mapping
    with open(output_path, 'w') as f:
        json.dump(image_urls, f)

    print(f"Saved image URLs to {output_path}")
    return image_urls


async def download_image(session: aiohttp.ClientSession, asin: str, url: str,
                         output_dir: Path, semaphore: asyncio.Semaphore) -> tuple:
    """
    Download a single image asynchronously.
    """
    async with semaphore:
        output_path = output_dir / f"{asin}.jpg"

        # Skip if already downloaded
        if output_path.exists():
            return asin, True, "exists"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.read()

                    # Verify it's a valid image
                    try:
                        img = Image.open(BytesIO(content))
                        img.verify()

                        # Save image
                        with open(output_path, 'wb') as f:
                            f.write(content)

                        return asin, True, "downloaded"
                    except Exception:
                        return asin, False, "invalid_image"
                else:
                    return asin, False, f"status_{response.status}"

        except asyncio.TimeoutError:
            return asin, False, "timeout"
        except Exception as e:
            return asin, False, str(e)[:50]


async def download_images_async(image_urls: dict, output_dir: Path,
                                max_concurrent: int = 50):
    """
    Download all images asynchronously with progress bar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=20)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            download_image(session, asin, url, output_dir, semaphore)
            for asin, url in image_urls.items()
        ]

        results = {"downloaded": 0, "exists": 0, "failed": 0}
        failed_items = []

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                        desc="Downloading images"):
            asin, success, status = await coro

            if success:
                if status == "exists":
                    results["exists"] += 1
                else:
                    results["downloaded"] += 1
            else:
                results["failed"] += 1
                failed_items.append((asin, status))

        print(f"\nDownload complete:")
        print(f"  Downloaded: {results['downloaded']}")
        print(f"  Already existed: {results['exists']}")
        print(f"  Failed: {results['failed']}")

        if failed_items[:10]:
            print(f"\nSample failures:")
            for asin, reason in failed_items[:10]:
                print(f"  {asin}: {reason}")

        return results


def generate_clip_embeddings(images_dir: Path, output_path: str,
                             batch_size: int = 32):
    """
    Generate FashionCLIP embeddings for all downloaded images.
    """
    from fashion_clip.fashion_clip import FashionCLIP

    print("Loading FashionCLIP model...")
    fclip = FashionCLIP('fashion-clip')

    # Get all image files
    image_files = list(images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images")

    embeddings = {}
    failed = []

    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc="Generating embeddings"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_asins = []

        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                batch_asins.append(img_path.stem)
            except Exception as e:
                failed.append((img_path.stem, str(e)))
                continue

        if batch_images:
            try:
                # Get embeddings
                with torch.no_grad():
                    batch_embeddings = fclip.encode_images(batch_images, batch_size=len(batch_images))

                # Store embeddings
                for asin, emb in zip(batch_asins, batch_embeddings):
                    embeddings[asin] = emb.cpu().numpy() if hasattr(emb, 'cpu') else emb

            except Exception as e:
                print(f"Batch error: {e}")
                failed.extend([(asin, str(e)) for asin in batch_asins])

    print(f"\nGenerated embeddings for {len(embeddings)} items")
    print(f"Failed: {len(failed)}")

    # Save embeddings
    print(f"Saving embeddings to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    # Also save as numpy array with ID mapping
    if embeddings:
        asin_list = list(embeddings.keys())
        emb_array = np.array([embeddings[a] for a in asin_list])

        np.save(output_path.replace('.pkl', '_array.npy'), emb_array)
        with open(output_path.replace('.pkl', '_ids.json'), 'w') as f:
            json.dump(asin_list, f)

        print(f"Embedding shape: {emb_array.shape}")

    return embeddings


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download Amazon images and generate CLIP embeddings')
    parser.add_argument('--step', choices=['extract', 'download', 'embed', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--max-concurrent', type=int, default=50,
                       help='Max concurrent downloads')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    args = parser.parse_args()

    # Paths
    base_dir = Path('/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion')
    metadata_path = base_dir / 'raw/meta_Clothing_Shoes_and_Jewelry.json.gz'
    valid_items_path = base_dir / 'processed/valid_items.json'
    image_urls_path = base_dir / 'processed/image_urls.json'
    images_dir = base_dir / 'images'
    embeddings_path = str(base_dir / 'processed/amazon_mens_embeddings.pkl')

    if args.step in ['extract', 'all']:
        print("=" * 60)
        print("Step 1: Extracting image URLs from metadata")
        print("=" * 60)
        image_urls = extract_image_urls(
            str(metadata_path),
            str(valid_items_path),
            str(image_urls_path)
        )
    else:
        with open(image_urls_path, 'r') as f:
            image_urls = json.load(f)

    if args.step in ['download', 'all']:
        print("\n" + "=" * 60)
        print("Step 2: Downloading images")
        print("=" * 60)
        asyncio.run(download_images_async(
            image_urls,
            images_dir,
            max_concurrent=args.max_concurrent
        ))

    if args.step in ['embed', 'all']:
        print("\n" + "=" * 60)
        print("Step 3: Generating FashionCLIP embeddings")
        print("=" * 60)
        generate_clip_embeddings(
            images_dir,
            embeddings_path,
            batch_size=args.batch_size
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
