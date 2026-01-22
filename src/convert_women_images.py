"""
Convert all women's fashion images to WebP format.

This script handles: JPG, PNG, AVIF, and other formats
Output: data/women_fashion/images_webp/{category}/{subcategory}/{id}.webp
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
import shutil

# Enable AVIF support
try:
    import pillow_avif
    print("AVIF support enabled")
except ImportError:
    print("Warning: pillow-avif-plugin not installed, AVIF files may fail")

# Paths
BASE_DIR = Path("/home/ubuntu/recSys/outfitTransformer")
WOMEN_DATA_DIR = BASE_DIR / "womenClothes" / "Women_s Brands"
OUTPUT_DIR = BASE_DIR / "data" / "women_fashion" / "images_webp"
ITEMS_PATH = BASE_DIR / "data" / "women_fashion" / "processed" / "women_items.pkl"


def convert_image_to_webp(
    src_path: Path,
    dst_path: Path,
    quality: int = 85
) -> bool:
    """
    Convert a single image to WebP format.

    Args:
        src_path: Source image path
        dst_path: Destination WebP path
        quality: WebP quality (1-100)

    Returns:
        True if successful, False otherwise
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P', 'LA'):
                # Handle transparency by compositing on white background
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'LA':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            img.save(dst_path, 'WebP', quality=quality)

        return True
    except Exception as e:
        print(f"  Error converting {src_path}: {e}")
        return False


def convert_all_images(
    items_path: Path = ITEMS_PATH,
    output_dir: Path = OUTPUT_DIR,
    quality: int = 85
) -> dict:
    """
    Convert all images in women_items.pkl to WebP format.

    Args:
        items_path: Path to women_items.pkl
        output_dir: Output directory for WebP images
        quality: WebP quality

    Returns:
        Dict with conversion stats and updated items
    """
    print("=" * 60)
    print("Women's Fashion Image Conversion to WebP")
    print("=" * 60)

    # Load items
    print(f"\nLoading items from {items_path}...")
    with open(items_path, 'rb') as f:
        data = pickle.load(f)

    items = data['items']
    print(f"  Loaded {len(items)} items")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert images
    print(f"\nConverting images to WebP (quality={quality})...")

    success = 0
    failed = 0
    failed_items = []
    updated_items = {}

    for item_id, item_data in tqdm(items.items(), desc="Converting"):
        src_path = Path(item_data['image_path'])

        if not src_path.exists():
            failed += 1
            failed_items.append(item_id)
            continue

        # Generate output path
        category = item_data.get('category', 'unknown')
        subcategory = item_data.get('subcategory', 'unknown')
        image_idx = item_data.get('image_index', item_id.split('/')[-1])

        dst_path = output_dir / category / subcategory / f"{image_idx}.webp"

        if convert_image_to_webp(src_path, dst_path, quality):
            success += 1
            # Update item with new path
            updated_item = item_data.copy()
            updated_item['image_path'] = str(dst_path)
            updated_item['original_path'] = str(src_path)
            updated_items[item_id] = updated_item
        else:
            failed += 1
            failed_items.append(item_id)

    print(f"\n  Successfully converted: {success}")
    print(f"  Failed: {failed}")

    # Save updated items
    output_data = {
        'items': updated_items,
        'stats': {
            'total_items': len(items),
            'converted': success,
            'failed': failed,
            'failed_items': failed_items[:20],  # First 20 failed
        }
    }

    # Save to new pickle file
    output_items_path = items_path.parent / "women_items_webp.pkl"
    print(f"\nSaving updated items to {output_items_path}...")
    with open(output_items_path, 'wb') as f:
        pickle.dump(output_data, f)

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*.webp'))
    print(f"  Total WebP size: {total_size / (1024*1024):.1f} MB")
    print("Done!")

    return output_data


def scan_source_images() -> dict:
    """Scan source images and show format distribution."""
    print("Scanning source images...")

    formats = {}
    total = 0

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.avif']:
        count = len(list(WOMEN_DATA_DIR.rglob(ext)))
        formats[ext] = count
        total += count

    print(f"\nImage format distribution:")
    for fmt, count in sorted(formats.items(), key=lambda x: -x[1]):
        print(f"  {fmt}: {count} ({count/total*100:.1f}%)")
    print(f"  Total: {total}")

    return formats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert women fashion images to WebP')
    parser.add_argument(
        '--scan-only',
        action='store_true',
        help='Only scan images, do not convert'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=85,
        help='WebP quality (1-100)'
    )
    parser.add_argument(
        '--items',
        default=str(ITEMS_PATH),
        help='Path to women_items.pkl'
    )
    parser.add_argument(
        '--output',
        default=str(OUTPUT_DIR),
        help='Output directory for WebP images'
    )

    args = parser.parse_args()

    if args.scan_only:
        scan_source_images()
    else:
        convert_all_images(
            items_path=Path(args.items),
            output_dir=Path(args.output),
            quality=args.quality
        )
