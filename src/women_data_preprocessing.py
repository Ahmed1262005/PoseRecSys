"""
Women's Fashion Data Preprocessing Pipeline

Loads Excel files from womenClothes/Women_s Brands/, normalizes columns,
maps to unified schema, and prepares data for FashionCLIP embedding generation.

Data Source: /home/ubuntu/recSys/outfitTransformer/womenClothes/Women_s Brands/
Output: data/women_fashion/processed/women_items.pkl
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


# Base paths
BASE_DIR = Path("/home/ubuntu/recSys/outfitTransformer")
WOMEN_DATA_DIR = BASE_DIR / "womenClothes" / "Women_s Brands"
OUTPUT_DIR = BASE_DIR / "data" / "women_fashion"
PROCESSED_DIR = OUTPUT_DIR / "processed"
IMAGES_WEBP_DIR = OUTPUT_DIR / "images_webp"


# Column name normalization mapping
COLUMN_NORMALIZATION = {
    # Color variations
    'color ': 'color',
    'color': 'color',
    'color/wash': 'color',

    # Fit variations
    'fit': 'fit',
    'fit/style': 'fit',

    # Sleeve variations
    'sleeve length': 'sleeve_length',
    'sleeve lenght': 'sleeve_length',  # Typo in data
    'strap type': 'sleeve_type',

    # Neckline variations
    'neckline': 'neckline',
    'neckline ': 'neckline',

    # Brand variations
    'brand': 'brand',
    'brand ': 'brand',

    # Fabric/Material variations
    'fabric composition': 'fabric',
    'fabric type': 'fabric',
    'fabric': 'fabric',

    # Item ID variations
    'item ': 'item_group',
    'item': 'item_group',
    'item list': 'image_index',
    'image title': 'image_index',
    'image #': 'image_index',
    'image': 'image_index',
    'item_number': 'image_index',
    'id': 'image_index',

    # Other attributes
    'url': 'url',
    'texture': 'texture',
    'pattern': 'pattern',
    'pocket': 'pocket',
    'knit type': 'knit_type',
    'occasion': 'occasion',
    'length': 'length',
    'rise': 'rise',
    'leg opening': 'leg_opening',
    'waist style / type': 'waist_style',
    'details': 'details',
    'rips': 'rips',
}


# Category configuration - maps Excel files to categories
# Image directories are based on actual scan of the dataset
EXCEL_CONFIGS = [
    # Tops - Knitwear (543 items)
    {
        'excel_path': 'Tops/Tops - Knitwear - Done/Knitwear - Menna_.xlsx',
        'category': 'tops_knitwear',
        'image_dirs': [
            'Tops/Tops - Knitwear - Done/Sweaters/Sweaters',
            'Tops/Tops - Knitwear - Done/Pullovers/Pullovers',
            'Tops/Tops - Knitwear - Done/Cardigans/Cardi',
            'Tops/Tops - Knitwear - Done/Knit Tops',
        ],
    },
    # Tops - Woven (Omar) - T-shirts, Shirts, Dressy, Peplum, Tunics, Wrap
    {
        'excel_path': 'Tops/Tops - Woven/Tops – Woven - Omar_.xlsx',
        'category': 'tops_woven',
        'image_dirs': [
            'Tops/Tops - Woven/Blouses/images',
            'Tops/Tops - Woven/Shirts',
            'Tops/Tops - Woven/T-shirts',
            'Tops/Tops - Woven/Dressy tops',
            'Tops/Tops - Woven/Peplum Tops',
            'Tops/Tops - Woven/Tunics',
            'Tops/Tops - Woven/Wrap Tops',
        ],
    },
    # Tops - Sleeveless/Summer (500 items)
    {
        'excel_path': 'Tops/Sleeveless - Summer Tops - Done/Summer Tops - Menna.xlsx',
        'category': 'tops_sleeveless',
        'image_dirs': [
            'Tops/Sleeveless - Summer Tops - Done/Tank Tops',
            'Tops/Sleeveless - Summer Tops - Done/Camisoles',
            'Tops/Sleeveless - Summer Tops - Done/Tube Tops',
            'Tops/Sleeveless - Summer Tops - Done/Crop Tops',
        ],
    },
    # Tops - Special Styles (520 items)
    {
        'excel_path': 'Tops/Tops Special Styles/Tops Special Styles - Kareem.xlsx',
        'category': 'tops_special',
        'image_dirs': [
            'Tops/Tops Special Styles/Bodysuits',
            'Tops/Tops Special Styles/Polo Tops',
            'Tops/Tops Special Styles/Henley Tops',
        ],
    },
    # Dresses (608 items)
    {
        'excel_path': 'Dresses /Dresses - Mai + Abdelrahman.xlsx',
        'category': 'dresses',
        'image_dirs': [
            'Dresses /Images',
        ],
    },
    # Bottoms - Trousers (includes shorts, culottes)
    {
        'excel_path': 'Bottoms /Trousers/AI_TROUSERS_DATABASE.xlsx',
        'category': 'bottoms_trousers',
        'image_dirs': [
            'Bottoms /Trousers/trousers_images',
            'Bottoms /Trousers/Trousers-M',
            'Bottoms /Shorts',
            'Bottoms /Culottes/Culottes ',
        ],
    },
    # Bottoms - Skorts (178 items)
    {
        'excel_path': 'Bottoms /Skorts/Women skorts.xlsx',
        'category': 'bottoms_skorts',
        'image_dirs': [
            'Bottoms /Skorts/women skorts',
        ],
    },
    # Bottoms - Skirts (127 items)
    {
        'excel_path': 'Bottoms /Skirts/Women skirts.xlsx',  # May not exist
        'category': 'bottoms_skirts',
        'image_dirs': [
            'Bottoms /Skirts/Images /women skirt',
        ],
    },
    # Outerwear (Hamza) - 1393 items total
    {
        'excel_path': 'Outerwear/Outerwear 1 - Hamza.xlsx',
        'category': 'outerwear',
        'image_dirs': [
            'Outerwear/Blazers',
            'Outerwear/Coats',
            'Outerwear/Jackets',
            'Outerwear/Vests',
            'Outerwear/Capes/ Capes',
            'Outerwear/Ponchos',
        ],
    },
    # Sportswear (612 items)
    {
        'excel_path': 'Sportswear /Sportswear - Lina.xlsx',
        'category': 'sportswear',
        'image_dirs': [
            'Sportswear /Leggings',
            'Sportswear /Joggers',
        ],
    },
]


@dataclass
class WomenItem:
    """Unified data class for women's fashion items."""
    item_id: str
    category: str
    subcategory: str
    image_path: str

    # Common attributes
    color: Optional[str] = None
    fit: Optional[str] = None
    neckline: Optional[str] = None
    sleeve_length: Optional[str] = None
    sleeve_type: Optional[str] = None
    fabric: Optional[str] = None
    texture: Optional[str] = None
    pattern: Optional[str] = None
    brand: Optional[str] = None
    url: Optional[str] = None

    # Category-specific
    occasion: Optional[str] = None
    length: Optional[str] = None
    knit_type: Optional[str] = None

    # Bottoms-specific
    rise: Optional[str] = None
    leg_opening: Optional[str] = None
    waist_style: Optional[str] = None

    # Metadata
    source_excel: Optional[str] = None
    image_index: Optional[int] = None


def normalize_column_name(col: str) -> str:
    """Normalize column name to standard format."""
    col_lower = col.strip().lower()
    return COLUMN_NORMALIZATION.get(col_lower, col_lower.replace(' ', '_'))


def get_image_index_column(df: pd.DataFrame) -> str:
    """Find the best image index column from the dataframe."""
    # Priority order: check columns likely to have image indices first
    priority_cols = ['Item List', 'Image Title', 'Image #', 'Image', 'ID', 'item_number', 'item list']

    # First pass: check priority columns by exact match
    for prio_col in priority_cols:
        for col in df.columns:
            if col.strip().lower() == prio_col.lower():
                # Check if it has valid numeric data
                series = df[col].dropna()
                if len(series) > 0:
                    first_val = series.iloc[0]
                    # Check if numeric (including numpy types)
                    try:
                        int(float(first_val))
                        return col
                    except (ValueError, TypeError):
                        pass

    # Second pass: look for any column with numeric values
    for col in df.columns:
        col_lower = col.strip().lower()
        if 'item' in col_lower or 'image' in col_lower or 'id' in col_lower:
            series = df[col].dropna()
            if len(series) > 0:
                first_val = series.iloc[0]
                try:
                    int(float(first_val))
                    return col
                except (ValueError, TypeError):
                    pass

    return None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names in a DataFrame."""
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def find_image_path(
    image_index: int,
    image_dirs: List[str],
    base_dir: Path
) -> Optional[Path]:
    """Find the actual image path for a given index across multiple directories."""
    extensions = ['.webp', '.jpg', '.jpeg', '.png', '.avif']

    for img_dir in image_dirs:
        dir_path = base_dir / img_dir
        if not dir_path.exists():
            continue

        for ext in extensions:
            # Try direct numeric filename
            img_path = dir_path / f"{image_index}{ext}"
            if img_path.exists():
                return img_path

    return None


def determine_subcategory(image_path: Path) -> str:
    """Determine subcategory from image path."""
    parts = image_path.parts

    # Find the subcategory from folder structure
    subcategory_keywords = {
        'sweaters': 'sweaters',
        'pullovers': 'pullovers',
        'cardigans': 'cardigans',
        'knit tops': 'knit_tops',
        'blouses': 'blouses',
        'shirts': 'shirts',
        't-shirts': 'tshirts',
        'peplum': 'peplum',
        'tunics': 'tunics',
        'wrap tops': 'wrap_tops',
        'tank tops': 'tank_tops',
        'camisoles': 'camisoles',
        'tube tops': 'tube_tops',
        'crop tops': 'crop_tops',
        'bodysuits': 'bodysuits',
        'body suits': 'bodysuits',
        'polo tops': 'polo_tops',
        'henley': 'henley_tops',
        'dresses': 'dresses',
        'culottes': 'culottes',
        'shorts': 'shorts',
        'trousers': 'trousers',
        'skorts': 'skorts',
        'blazers': 'blazers',
        'coats': 'coats',
        'jackets': 'jackets',
        'vests': 'vests',
        'capes': 'capes',
        'ponchos': 'ponchos',
        'leggings': 'leggings',
        'joggers': 'joggers',
    }

    for part in parts:
        part_lower = part.lower()
        for keyword, subcategory in subcategory_keywords.items():
            if keyword in part_lower:
                return subcategory

    return 'other'


def convert_to_webp(
    src_path: Path,
    dst_dir: Path,
    item_id: str,
    quality: int = 85
) -> Path:
    """Convert image to WebP format."""
    dst_path = dst_dir / f"{item_id.replace('/', '_')}.webp"
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        return dst_path

    try:
        with Image.open(src_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(dst_path, 'WebP', quality=quality)
        return dst_path
    except Exception as e:
        print(f"Error converting {src_path}: {e}")
        return None


def process_excel_file(
    config: dict,
    base_dir: Path,
    convert_images: bool = False
) -> List[WomenItem]:
    """Process a single Excel file and return list of items."""
    excel_path = base_dir / config['excel_path']

    if not excel_path.exists():
        print(f"  Warning: Excel file not found: {excel_path}")
        return []

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"  Error reading {excel_path}: {e}")
        return []

    # Find the image index column BEFORE normalizing (to avoid column conflicts)
    image_col = get_image_index_column(df)
    if image_col is None:
        print(f"  Warning: No valid image index column found")
        return []

    # Store the image index values
    image_indices = df[image_col].copy()

    # Normalize columns
    df = normalize_dataframe(df)

    items = []
    category = config['category']
    image_dirs = config.get('image_dirs', [])

    for idx, row in df.iterrows():
        # Get image index from our saved column
        image_idx = image_indices.iloc[idx]
        if pd.isna(image_idx):
            continue

        try:
            image_idx = int(float(image_idx))
        except (ValueError, TypeError):
            continue

        # Find image path
        image_path = find_image_path(image_idx, image_dirs, base_dir)
        if image_path is None:
            continue

        # Determine subcategory from image location
        subcategory = determine_subcategory(image_path)

        # Create unique item ID
        item_id = f"{category}/{subcategory}/{image_idx}"

        # Convert to WebP if requested
        final_image_path = str(image_path)
        if convert_images:
            webp_path = convert_to_webp(
                image_path,
                IMAGES_WEBP_DIR / category / subcategory,
                str(image_idx)
            )
            if webp_path:
                final_image_path = str(webp_path)

        # Extract attributes
        item = WomenItem(
            item_id=item_id,
            category=category,
            subcategory=subcategory,
            image_path=final_image_path,
            color=str(row.get('color', '')).strip() if pd.notna(row.get('color')) else None,
            fit=str(row.get('fit', '')).strip() if pd.notna(row.get('fit')) else None,
            neckline=str(row.get('neckline', '')).strip() if pd.notna(row.get('neckline')) else None,
            sleeve_length=str(row.get('sleeve_length', '')).strip() if pd.notna(row.get('sleeve_length')) else None,
            sleeve_type=str(row.get('sleeve_type', '')).strip() if pd.notna(row.get('sleeve_type')) else None,
            fabric=str(row.get('fabric', '')).strip() if pd.notna(row.get('fabric')) else None,
            texture=str(row.get('texture', '')).strip() if pd.notna(row.get('texture')) else None,
            pattern=str(row.get('pattern', '')).strip() if pd.notna(row.get('pattern')) else None,
            brand=str(row.get('brand', '')).strip() if pd.notna(row.get('brand')) else None,
            url=str(row.get('url', '')).strip() if pd.notna(row.get('url')) else None,
            occasion=str(row.get('occasion', '')).strip() if pd.notna(row.get('occasion')) else None,
            length=str(row.get('length', '')).strip() if pd.notna(row.get('length')) else None,
            knit_type=str(row.get('knit_type', '')).strip() if pd.notna(row.get('knit_type')) else None,
            rise=str(row.get('rise', '')).strip() if pd.notna(row.get('rise')) else None,
            leg_opening=str(row.get('leg_opening', '')).strip() if pd.notna(row.get('leg_opening')) else None,
            waist_style=str(row.get('waist_style', '')).strip() if pd.notna(row.get('waist_style')) else None,
            source_excel=config['excel_path'],
            image_index=image_idx,
        )
        items.append(item)

    return items


def process_all_excel_files(
    convert_images: bool = False,
    save_output: bool = True
) -> Dict:
    """Process all Excel files and create unified dataset."""
    print("=" * 60)
    print("Women's Fashion Data Preprocessing Pipeline")
    print("=" * 60)

    # Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if convert_images:
        IMAGES_WEBP_DIR.mkdir(parents=True, exist_ok=True)

    all_items = []
    stats = {
        'by_category': {},
        'by_subcategory': {},
        'total_items': 0,
        'excel_files_processed': 0,
    }

    for config in tqdm(EXCEL_CONFIGS, desc="Processing Excel files"):
        print(f"\nProcessing: {config['excel_path']}")
        items = process_excel_file(config, WOMEN_DATA_DIR, convert_images)

        if items:
            all_items.extend(items)
            stats['excel_files_processed'] += 1

            # Update category stats
            category = config['category']
            stats['by_category'][category] = stats['by_category'].get(category, 0) + len(items)

            # Update subcategory stats
            for item in items:
                key = f"{item.category}/{item.subcategory}"
                stats['by_subcategory'][key] = stats['by_subcategory'].get(key, 0) + 1

            print(f"  Found {len(items)} items")

    stats['total_items'] = len(all_items)

    # Convert to dict format
    items_dict = {item.item_id: asdict(item) for item in all_items}

    output = {
        'items': items_dict,
        'stats': stats,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nTotal items: {stats['total_items']}")
    print(f"Excel files processed: {stats['excel_files_processed']}")
    print("\nBy category:")
    for cat, count in sorted(stats['by_category'].items()):
        print(f"  {cat}: {count}")
    print("\nBy subcategory:")
    for subcat, count in sorted(stats['by_subcategory'].items()):
        print(f"  {subcat}: {count}")

    # Save output
    if save_output:
        output_path = PROCESSED_DIR / "women_items.pkl"
        print(f"\nSaving to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        print("Done!")

        # Also save a JSON version for inspection
        json_path = PROCESSED_DIR / "women_items_stats.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)

    return output


def scan_all_images(base_dir: Path = WOMEN_DATA_DIR) -> Dict[str, List[Path]]:
    """Scan all images in the dataset and organize by directory."""
    print("Scanning all images in dataset...")

    images_by_dir = {}
    extensions = ('.jpg', '.jpeg', '.png', '.webp', '.avif')

    for img_path in tqdm(list(base_dir.rglob('*')), desc="Scanning"):
        if img_path.suffix.lower() in extensions:
            parent = str(img_path.parent.relative_to(base_dir))
            if parent not in images_by_dir:
                images_by_dir[parent] = []
            images_by_dir[parent].append(img_path)

    print(f"\nFound {sum(len(v) for v in images_by_dir.values())} images in {len(images_by_dir)} directories")

    for dir_path, images in sorted(images_by_dir.items(), key=lambda x: -len(x[1]))[:20]:
        print(f"  {dir_path}: {len(images)} images")

    return images_by_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess women\'s fashion data')
    parser.add_argument(
        '--scan-only',
        action='store_true',
        help='Only scan images, do not process Excel files'
    )
    parser.add_argument(
        '--convert-images',
        action='store_true',
        help='Convert images to WebP format'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output files'
    )

    args = parser.parse_args()

    if args.scan_only:
        scan_all_images()
    else:
        process_all_excel_files(
            convert_images=args.convert_images,
            save_output=not args.no_save
        )
