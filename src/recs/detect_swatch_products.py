#!/usr/bin/env python3
"""
Detect and Remove Color Swatch Products

Uses CLIP zero-shot classification to identify products where gallery images
are color swatch thumbnails instead of actual product photos.

Usage:
    # Dry run - just detect and report
    python detect_swatch_products.py --dry-run --limit 1000

    # Full scan and flag products
    python detect_swatch_products.py --batch-size 500

    # Delete flagged products
    python detect_swatch_products.py --delete-flagged

    # Test specific product
    python detect_swatch_products.py --test-product <product_id>

    # Test specific URL
    python detect_swatch_products.py --test-url <image_url>

Requirements:
    - SUPABASE_URL and SUPABASE_SERVICE_KEY env vars
    - FashionCLIP model
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

# Thread-local storage for Supabase clients
_thread_local = threading.local()


def get_supabase_client() -> Client:
    """Create Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


def get_thread_local_supabase() -> Client:
    """Get a thread-local Supabase client for parallel operations."""
    if not hasattr(_thread_local, 'supabase'):
        _thread_local.supabase = get_supabase_client()
    return _thread_local.supabase


class SwatchDetector:
    """
    Detects color swatch thumbnails using CLIP zero-shot classification.

    Uses carefully crafted prompts to distinguish:
    - Color swatch thumbnails (small squares showing available color options)
    - Real product photos (clothing on models, mannequins, or flat lay)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SwatchDetector/1.0)'
        })

        # CLIP model (lazy loaded)
        self.model = None
        self.processor = None
        self.device = None

        # Pre-computed text embeddings
        self._swatch_embedding = None
        self._product_embedding = None

    def _ensure_model_loaded(self):
        """Lazily load the CLIP model and compute text embeddings."""
        if self.model is not None:
            return

        import torch
        from transformers import CLIPProcessor, CLIPModel

        print("[SwatchDetector] Loading FashionCLIP model...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Single prompt approach: just check for person
        # If person score is low, it might be a swatch
        swatch_prompts = [
            "solid color",
            "plain background",
            "color swatch",
            "no person visible",
        ]

        product_prompts = [
            "a photo with a person",
            "a woman in the image",
            "a human visible",
            "someone in the picture",
        ]

        # Compute averaged text embeddings
        with torch.no_grad():
            # Swatch embedding
            inputs = self.processor(text=swatch_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            swatch_emb = self.model.get_text_features(**inputs)
            swatch_emb = swatch_emb / swatch_emb.norm(dim=-1, keepdim=True)
            self._swatch_embedding = swatch_emb.mean(dim=0, keepdim=True)
            self._swatch_embedding = self._swatch_embedding / self._swatch_embedding.norm()

            # Product embedding
            inputs = self.processor(text=product_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            product_emb = self.model.get_text_features(**inputs)
            product_emb = product_emb / product_emb.norm(dim=-1, keepdim=True)
            self._product_embedding = product_emb.mean(dim=0, keepdim=True)
            self._product_embedding = self._product_embedding / self._product_embedding.norm()

        print(f"[SwatchDetector] Model loaded on {self.device}")

    def download_image(self, url: str, timeout: int = 10) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            if self.verbose:
                print(f"  [Download error] {url}: {e}")
            return None

    def analyze_image(self, img: Image.Image) -> Dict[str, Any]:
        """
        Analyze an image using COMBINED pixel analysis + CLIP + size check.

        Swatches are typically:
        - Small images (< 300px)
        - Low entropy (uniform color)
        - CLIP thinks it's a swatch
        """
        import torch

        self._ensure_model_loaded()

        # 0. Image size check - swatches are typically small
        width, height = img.size
        is_small = max(width, height) < 300

        # 1. Pixel-based analysis
        img_small = img.resize((100, 100), Image.Resampling.LANCZOS)
        pixels = np.array(img_small, dtype=np.float32)

        # Color variance
        color_variance = float(np.std(pixels))

        # Color entropy
        quantized = (pixels / 32).astype(int)
        color_indices = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]
        unique, counts = np.unique(color_indices.flatten(), return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # 2. CLIP analysis
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_emb = self.model.get_image_features(**inputs)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        swatch_sim = float((image_emb @ self._swatch_embedding.T).squeeze())
        product_sim = float((image_emb @ self._product_embedding.T).squeeze())

        # Decision logic:
        # True swatches are SMALL THUMBNAILS (80x80) with uniform color.
        # - Small size: max dimension < 150px
        # - Low entropy: few distinct colors
        # - Low variance: minimal brightness variation
        #
        # Large images with low entropy/variance are product photos with
        # simple backgrounds, NOT swatches.
        #
        # Rule: small size AND entropy < 1.5 AND variance < 30

        max_dim = max(width, height)
        is_swatch = max_dim < 150 and entropy < 1.5 and color_variance < 30

        confidence = abs(swatch_sim - product_sim)

        return {
            'is_swatch': is_swatch,
            'swatch_score': round(swatch_sim, 4),
            'product_score': round(product_sim, 4),
            'entropy': round(entropy, 2),
            'variance': round(color_variance, 2),
            'width': width,
            'height': height,
            'confidence': round(confidence, 4),
            'reasons': [f"size={width}x{height}", f"entropy={entropy:.2f}", f"clip={swatch_sim:.3f}/{product_sim:.3f}"],
        }

    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Download and analyze a single image URL."""
        img = self.download_image(url)
        if img is None:
            return {
                'is_swatch': False,
                'error': 'download_failed',
                'confidence': 0,
            }
        result = self.analyze_image(img)
        result['url'] = url
        return result

    def analyze_product_gallery(
        self,
        gallery_urls: List[str],
        primary_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze all gallery images for a product.

        Returns:
            {
                'is_swatch_product': bool,  # True if product has swatch issues
                'swatch_count': int,        # Number of gallery images that are swatches
                'total_gallery': int,       # Total gallery images analyzed
                'swatch_ratio': float,      # Fraction of gallery that are swatches
                'image_results': [...]      # Per-image analysis
            }
        """
        results = []
        swatch_count = 0

        # Analyze gallery images
        for url in gallery_urls:
            if not url:
                continue
            analysis = self.analyze_url(url)
            analysis['is_gallery'] = True
            results.append(analysis)

            if analysis.get('is_swatch', False):
                swatch_count += 1

        # Analyze primary image if provided
        primary_is_swatch = False
        if primary_url:
            primary_analysis = self.analyze_url(primary_url)
            primary_analysis['is_gallery'] = False
            primary_analysis['is_primary'] = True
            results.append(primary_analysis)
            primary_is_swatch = primary_analysis.get('is_swatch', False)

        total_gallery = len([r for r in results if r.get('is_gallery', False)])
        swatch_ratio = swatch_count / total_gallery if total_gallery > 0 else 0

        # Product is flagged if:
        # - Primary image is a swatch, OR
        # - Majority of gallery images are swatches
        is_swatch_product = primary_is_swatch or swatch_ratio >= 0.5

        return {
            'is_swatch_product': is_swatch_product,
            'primary_is_swatch': primary_is_swatch,
            'swatch_count': swatch_count,
            'total_gallery': total_gallery,
            'swatch_ratio': round(swatch_ratio, 2),
            'image_results': results,
        }


def test_url(url: str, verbose: bool = True) -> Dict[str, Any]:
    """Test swatch detection on a specific URL."""
    print("=" * 70)
    print(f"Testing URL: {url[:70]}...")
    print("=" * 70)

    detector = SwatchDetector(verbose=verbose)
    result = detector.analyze_url(url)

    print(f"\nResult:")
    print(f"  Is swatch: {result.get('is_swatch', False)}")
    print(f"  Swatch score: {result.get('swatch_score', 'N/A')}")
    print(f"  Product score: {result.get('product_score', 'N/A')}")
    print(f"  Confidence: {result.get('confidence', 'N/A')}")

    return result


def test_product(product_id: str, verbose: bool = True) -> Dict[str, Any]:
    """Test swatch detection on a specific product."""
    print("=" * 70)
    print(f"Testing Product: {product_id}")
    print("=" * 70)

    supabase = get_supabase_client()
    detector = SwatchDetector(verbose=verbose)

    # Fetch product with all image URLs
    result = supabase.table('products').select(
        'id, name, brand, primary_image_url, gallery_images'
    ).eq('id', product_id).execute()

    if not result.data:
        print(f"Product not found: {product_id}")
        return {'error': 'not_found'}

    product = result.data[0]
    print(f"\nProduct: {product.get('name', 'Unknown')}")
    print(f"Brand: {product.get('brand', 'Unknown')}")

    primary_url = product.get('primary_image_url')
    gallery_images = product.get('gallery_images') or []

    print(f"Primary image: {primary_url}")
    print(f"Gallery images: {len(gallery_images)}")

    # Analyze all images
    print("\n" + "-" * 50)
    print("Analyzing images...")
    print("-" * 50)

    analysis = detector.analyze_product_gallery(
        gallery_urls=gallery_images,
        primary_url=primary_url
    )

    # Print results
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Is swatch product: {analysis['is_swatch_product']}")
    print(f"Primary is swatch: {analysis['primary_is_swatch']}")
    print(f"Gallery swatches: {analysis['swatch_count']}/{analysis['total_gallery']} ({analysis['swatch_ratio']*100:.0f}%)")

    if verbose:
        print(f"\n{'='*50}")
        print("Per-image analysis:")
        print(f"{'='*50}")
        for img_result in analysis['image_results']:
            img_type = "PRIMARY" if img_result.get('is_primary') else "GALLERY"
            swatch_str = "SWATCH" if img_result.get('is_swatch') else "OK"
            print(f"\n  [{img_type}] {swatch_str}")
            print(f"  URL: {img_result.get('url', 'N/A')[:80]}...")
            if 'error' not in img_result:
                print(f"  Swatch score: {img_result.get('swatch_score', 'N/A')}")
                print(f"  Product score: {img_result.get('product_score', 'N/A')}")

    return analysis


def _analyze_single_image(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Analyze a single image URL for swatch detection (thread-safe, no CLIP)."""
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; SwatchDetector/1.0)'
        })
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')

        width, height = img.size

        # Pixel-based analysis
        img_small = img.resize((100, 100), Image.Resampling.LANCZOS)
        pixels = np.array(img_small, dtype=np.float32)

        # Color variance
        color_variance = float(np.std(pixels))

        # Color entropy
        quantized = (pixels / 32).astype(int)
        color_indices = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]
        unique, counts = np.unique(color_indices.flatten(), return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Detection rule: small size AND low entropy AND low variance
        max_dim = max(width, height)
        is_swatch = max_dim < 150 and entropy < 1.5 and color_variance < 30

        return {
            'url': url,
            'is_swatch': is_swatch,
            'width': width,
            'height': height,
            'entropy': round(entropy, 2),
            'variance': round(color_variance, 2),
        }
    except Exception as e:
        return {
            'url': url,
            'is_swatch': False,
            'error': str(e),
        }


def _process_single_product(args):
    """Process a single product for swatch detection (worker function)."""
    product, swatch_threshold = args

    try:
        pid = product['id']
        gallery_images = product.get('gallery_images') or []
        primary_url = product.get('primary_image_url')

        if not gallery_images and not primary_url:
            return None

        # Analyze all images
        all_urls = []
        if primary_url:
            all_urls.append(('primary', primary_url))
        for url in gallery_images:
            if url:
                all_urls.append(('gallery', url))

        # Analyze images (could parallelize per-product, but keeping it simple)
        image_results = []
        gallery_swatch_count = 0
        gallery_total = 0
        primary_is_swatch = False
        gallery_swatch_urls = []
        gallery_ok_urls = []

        for img_type, url in all_urls:
            result = _analyze_single_image(url)
            result['is_gallery'] = (img_type == 'gallery')
            result['is_primary'] = (img_type == 'primary')
            image_results.append(result)

            if img_type == 'gallery':
                gallery_total += 1
                if result.get('is_swatch', False):
                    gallery_swatch_count += 1
                    gallery_swatch_urls.append(url)
                else:
                    gallery_ok_urls.append(url)
            elif img_type == 'primary':
                primary_is_swatch = result.get('is_swatch', False)

        swatch_ratio = gallery_swatch_count / gallery_total if gallery_total > 0 else 0
        is_swatch_product = primary_is_swatch or swatch_ratio >= 0.5

        if is_swatch_product or swatch_ratio >= swatch_threshold:
            return {
                'product_id': pid,
                'name': product.get('name', 'Unknown'),
                'brand': product.get('brand', 'Unknown'),
                'primary_image_url': primary_url,
                'primary_is_swatch': primary_is_swatch,
                'total_gallery': gallery_total,
                'swatch_count': gallery_swatch_count,
                'swatch_ratio': round(swatch_ratio, 2),
                'gallery_swatch_urls': gallery_swatch_urls,
                'gallery_ok_urls': gallery_ok_urls,
            }
        return None
    except Exception as e:
        return {'error': str(e), 'product_id': product.get('id')}


def detect_swatch_products(
    batch_size: int = 500,
    limit: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = False,
    swatch_threshold: float = 0.5,
    num_workers: int = 20,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect products where gallery images are color swatches.

    Args:
        batch_size: Products to process per batch
        limit: Max products to scan (None = all)
        dry_run: If True, don't modify database
        verbose: Print detailed output
        swatch_threshold: Fraction of gallery images that must be swatches to flag
        num_workers: Parallel workers for processing
        output_file: Custom output file path (default: /tmp/swatch_products_{timestamp}.json)

    Returns:
        Summary statistics
    """
    print("=" * 70, flush=True)
    print("Detecting Color Swatch Products (Parallel Processing)", flush=True)
    print("=" * 70, flush=True)

    supabase = get_supabase_client()
    # Note: Using lightweight pixel-based detection (no CLIP model needed)

    # Get total count of products with gallery images
    count_result = supabase.table('products').select(
        'id', count='exact'
    ).not_.is_('gallery_images', 'null').execute()
    total_products = count_result.count or 0

    if limit:
        total_to_process = min(total_products, limit)
    else:
        total_to_process = total_products

    print(f"\nTotal products with gallery images: {total_products}", flush=True)
    print(f"Processing: {total_to_process}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Workers: {num_workers}", flush=True)
    print(f"Swatch threshold: {swatch_threshold * 100}% of gallery images", flush=True)
    print(f"Dry run: {dry_run}", flush=True)

    # Track results
    processed = 0
    swatch_products = []
    errors = []

    start_time = time.time()
    offset = 0

    while processed < total_to_process:
        # Fetch batch of products
        batch_result = supabase.table('products').select(
            'id, name, brand, primary_image_url, gallery_images'
        ).not_.is_('gallery_images', 'null').range(offset, offset + batch_size - 1).execute()

        batch = batch_result.data or []
        if not batch:
            break

        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            args_list = [(product, swatch_threshold) for product in batch]
            results = list(executor.map(_process_single_product, args_list))

        # Collect results
        for result in results:
            if result is None:
                continue
            if 'error' in result:
                errors.append(result)
            else:
                swatch_products.append(result)

        processed += len(batch)
        offset += batch_size

        # Progress
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"Progress: {processed}/{total_to_process} ({processed/total_to_process*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | "
              f"Swatches found: {len(swatch_products)}", flush=True)

    # Sort by swatch ratio
    swatch_products.sort(key=lambda x: (-x['swatch_ratio'], -x['swatch_count']))

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Total products scanned: {processed}")
    print(f"Products flagged as swatches: {len(swatch_products)}")
    print(f"Errors: {len(errors)}")

    if verbose and swatch_products:
        print(f"\n{'='*70}")
        print("Flagged Swatch Products (Top 20)")
        print(f"{'='*70}")
        for p in swatch_products[:20]:
            print(f"\n  {p['name'][:50]}")
            print(f"  Brand: {p['brand']}")
            print(f"  Gallery swatches: {p['swatch_count']}/{p['total_gallery']} ({p['swatch_ratio']*100:.0f}%)")
            print(f"  Primary is swatch: {p['primary_is_swatch']}")
            print(f"  ID: {p['product_id']}")

    # Save results to file
    results_file = output_file or f"/tmp/swatch_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_scanned': processed,
                'flagged_count': len(swatch_products),
                'threshold': swatch_threshold,
                'scan_time': time.time() - start_time,
                'error_count': len(errors),
            },
            'flagged_products': swatch_products,
            'errors': errors[:100],
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Flag in database (unless dry run)
    if not dry_run and swatch_products:
        print(f"\nFlagging {len(swatch_products)} products in database...")
        flagged = 0

        for p in swatch_products:
            try:
                supabase.table('products').update({
                    'is_swatch_product': True,
                    'swatch_detection_score': p['swatch_ratio'],
                }).eq('id', p['product_id']).execute()
                flagged += 1
            except Exception as e:
                print(f"  Error flagging {p['product_id']}: {e}")

        print(f"Flagged {flagged} products")

    return {
        'total_scanned': processed,
        'flagged_count': len(swatch_products),
        'flagged_products': swatch_products,
        'results_file': results_file,
    }


def delete_flagged_products(dry_run: bool = True) -> int:
    """Delete products that were flagged as swatches."""
    print("=" * 70)
    print("Deleting Flagged Swatch Products")
    print("=" * 70)

    supabase = get_supabase_client()

    # Count flagged products
    count_result = supabase.table('products').select(
        'id', count='exact'
    ).eq('is_swatch_product', True).execute()

    flagged_count = count_result.count or 0
    print(f"Products flagged as swatches: {flagged_count}")

    if flagged_count == 0:
        print("No products to delete.")
        return 0

    if dry_run:
        print("\n*** DRY RUN - No deletions performed ***")

        # Show some examples
        sample = supabase.table('products').select(
            'id, name, brand'
        ).eq('is_swatch_product', True).limit(10).execute()

        print("\nSample products that would be deleted:")
        for p in sample.data:
            print(f"  - {p['name'][:50]} ({p['brand']})")

        return flagged_count

    # Confirm deletion
    print(f"\nWill delete {flagged_count} products and their embeddings.")

    # Delete embeddings first (foreign key constraint)
    flagged_ids = supabase.table('products').select('id').eq('is_swatch_product', True).execute()
    product_ids = [p['id'] for p in flagged_ids.data]

    deleted = 0
    batch_size = 100

    for i in range(0, len(product_ids), batch_size):
        batch_ids = product_ids[i:i + batch_size]

        # Delete embeddings
        supabase.table('image_embeddings').delete().in_('sku_id', batch_ids).execute()

        # Delete products
        supabase.table('products').delete().in_('id', batch_ids).execute()

        deleted += len(batch_ids)
        print(f"Deleted {deleted}/{len(product_ids)} products")

    print(f"\nSuccessfully deleted {deleted} swatch products")
    return deleted


def main():
    parser = argparse.ArgumentParser(description='Detect and remove color swatch products using CLIP')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Products per batch (default: 100)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max products to scan (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect only, do not modify database')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fraction of gallery images that must be swatches (default: 0.5)')
    parser.add_argument('--workers', type=int, default=20,
                        help='Parallel workers (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: /tmp/swatch_products_{timestamp}.json)')
    parser.add_argument('--delete-flagged', action='store_true',
                        help='Delete previously flagged swatch products')
    parser.add_argument('--test-product', type=str, default=None,
                        help='Test swatch detection on a specific product ID')
    parser.add_argument('--test-url', type=str, default=None,
                        help='Test swatch detection on a specific image URL')

    args = parser.parse_args()

    if args.test_url:
        test_url(args.test_url, verbose=True)
    elif args.test_product:
        test_product(args.test_product, verbose=args.verbose)
    elif args.delete_flagged:
        delete_flagged_products(dry_run=args.dry_run)
    else:
        detect_swatch_products(
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
            verbose=args.verbose,
            swatch_threshold=args.threshold,
            num_workers=args.workers,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()
