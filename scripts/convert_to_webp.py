#!/usr/bin/env python3
"""
Convert all images to WebP format without quality loss.

Required: pip install Pillow pillow-avif-plugin
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Enable AVIF support
try:
    import pillow_avif
except ImportError:
    print("Installing pillow-avif-plugin...")
    os.system("pip install pillow-avif-plugin")
    import pillow_avif


def convert_to_webp(image_path: Path, delete_original: bool = True) -> bool:
    """Convert a single image to WebP lossless format."""
    try:
        webp_path = image_path.with_suffix('.webp')

        # Skip if already webp
        if image_path.suffix.lower() == '.webp':
            return True

        # Skip if webp version already exists
        if webp_path.exists():
            print(f"  Skipping {image_path.name} (webp exists)")
            if delete_original:
                image_path.unlink()
            return True

        # Open and convert
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for PNG with transparency, keep RGBA)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                # Keep alpha channel
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')

            # Save as lossless WebP
            img.save(webp_path, 'WEBP', lossless=True, quality=100)

        # Verify the new file exists and has content
        if webp_path.exists() and webp_path.stat().st_size > 0:
            if delete_original:
                image_path.unlink()
            print(f"  Converted: {image_path.name} -> {webp_path.name}")
            return True
        else:
            print(f"  ERROR: Failed to create {webp_path.name}")
            return False

    except Exception as e:
        print(f"  ERROR converting {image_path.name}: {e}")
        return False


def main():
    # Default to HPImages folder
    base_dir = Path("/home/ubuntu/recSys/outfitTransformer/HPImages")

    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])

    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        sys.exit(1)

    # Find all non-webp images
    extensions = {'.jpg', '.jpeg', '.png', '.avif', '.gif', '.bmp', '.tiff'}
    images_to_convert = []

    for ext in extensions:
        images_to_convert.extend(base_dir.rglob(f'*{ext}'))
        images_to_convert.extend(base_dir.rglob(f'*{ext.upper()}'))

    # Remove duplicates and sort
    images_to_convert = sorted(set(images_to_convert))

    print(f"Found {len(images_to_convert)} images to convert")
    print(f"Source directory: {base_dir}")
    print("-" * 50)

    converted = 0
    failed = 0

    for img_path in images_to_convert:
        if convert_to_webp(img_path, delete_original=True):
            converted += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"Converted: {converted}")
    print(f"Failed: {failed}")

    # Count final webp files
    webp_count = len(list(base_dir.rglob('*.webp')))
    print(f"Total WebP files: {webp_count}")


if __name__ == "__main__":
    main()
