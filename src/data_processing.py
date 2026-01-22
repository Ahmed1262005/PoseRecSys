"""
Data Processing Module for Polyvore Dataset
Converts Polyvore outfit data to RecBole format

Logic:
- Each outfit becomes a "user" (set_id -> user_id)
- Each item in outfit is a positive interaction (rating=1.0)
- Item ID is extracted from image URL (tid parameter)
- Timestamp is sequential for ordering
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from tqdm import tqdm


def extract_item_id_from_url(image_url: str) -> Optional[str]:
    """
    Extract item ID (tid) from Polyvore image URL

    Example URL: http://img2.polyvoreimg.com/cgi/img-thing?.out=jpg&size=m&tid=194508109
    Returns: "194508109"
    """
    if not image_url:
        return None

    # Try to extract tid parameter
    match = re.search(r'tid=(\d+)', image_url)
    if match:
        return match.group(1)

    return None


def load_polyvore_outfits(data_dir: str) -> Tuple[List, List, List]:
    """
    Load Polyvore outfit JSON files

    Handles both formats:
    - Original: train_no_dup.json, valid_no_dup.json, test_no_dup.json
    - Alternative: train.json, valid.json, test.json

    Args:
        data_dir: Path to polyvore data directory

    Returns:
        Tuple of (train_data, valid_data, test_data) as lists
    """
    data_path = Path(data_dir)

    # Try original file names first
    file_patterns = [
        ("train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"),
        ("train.json", "valid.json", "test.json"),
    ]

    train_data, valid_data, test_data = [], [], []

    for train_name, valid_name, test_name in file_patterns:
        train_file = data_path / train_name
        valid_file = data_path / valid_name
        test_file = data_path / test_name

        if train_file.exists():
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            with open(valid_file, 'r') as f:
                valid_data = json.load(f)
            with open(test_file, 'r') as f:
                test_data = json.load(f)

            # Convert list to dict if needed (using index or set_id as key)
            if isinstance(train_data, list):
                train_data = {d.get('set_id', str(i)): d for i, d in enumerate(train_data)}
            if isinstance(valid_data, list):
                valid_data = {d.get('set_id', str(i)): d for i, d in enumerate(valid_data)}
            if isinstance(test_data, list):
                test_data = {d.get('set_id', str(i)): d for i, d in enumerate(test_data)}

            break

    if not train_data:
        raise FileNotFoundError(f"Could not find outfit files in {data_dir}")

    print(f"Loaded outfits: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    return train_data, valid_data, test_data


def extract_item_metadata(data_dir: str) -> Dict[str, Dict]:
    """
    Extract item metadata from outfit files

    Since Polyvore doesn't have a separate metadata file,
    we extract item info from the outfit data.

    Args:
        data_dir: Path to polyvore data directory

    Returns:
        Dictionary mapping item_id to metadata
    """
    train_data, valid_data, test_data = load_polyvore_outfits(data_dir)
    all_outfits = {**train_data, **valid_data, **test_data}

    metadata = {}

    for outfit_data in all_outfits.values():
        items = outfit_data.get('items', [])
        for item in items:
            # Extract item ID from image URL
            image_url = item.get('image', '')
            item_id = extract_item_id_from_url(image_url)

            if item_id and item_id not in metadata:
                metadata[item_id] = {
                    'item_id': item_id,
                    'name': item.get('name', ''),
                    'price': item.get('price', 0),
                    'categoryid': item.get('categoryid', 0),
                    'image_url': image_url,
                    'likes': item.get('likes', 0),
                }

    print(f"Extracted metadata for {len(metadata)} unique items")
    return metadata


def load_item_metadata(data_dir: str) -> Dict:
    """
    Load item metadata - extracts from outfit data if no separate file exists

    Args:
        data_dir: Path to polyvore data directory

    Returns:
        Dictionary mapping item_id to metadata
    """
    data_path = Path(data_dir)

    # Try to load from file first
    possible_files = [
        data_path / "polyvore_item_metadata.json",
        data_path / "item_metadata.json",
        data_path / "metadata.json",
    ]

    for metadata_file in possible_files:
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata for {len(metadata)} items from file")
            return metadata

    # Extract from outfit data
    return extract_item_metadata(data_dir)


def load_category_mapping(data_dir: str) -> Dict[int, str]:
    """
    Load category ID to name mapping

    Args:
        data_dir: Path to polyvore data directory

    Returns:
        Dictionary mapping category_id to category_name
    """
    data_path = Path(data_dir)
    category_file = data_path / "category_id.txt"

    mapping = {}
    if category_file.exists():
        with open(category_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on first space (format: "2 Clothing")
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    try:
                        cat_id = int(parts[0])
                        cat_name = parts[1]
                        mapping[cat_id] = cat_name
                    except ValueError:
                        continue

    print(f"Loaded {len(mapping)} category mappings")
    return mapping


def create_recbole_interactions(
    data_dir: str,
    output_path: Optional[str] = None,
    include_negative: bool = False
) -> pd.DataFrame:
    """
    Convert Polyvore outfits to RecBole interaction format

    Each outfit becomes a user, each item in outfit is a positive interaction.

    Args:
        data_dir: Path to polyvore data directory
        output_path: Optional path to save .inter file
        include_negative: Whether to generate negative samples

    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp
    """
    train_data, valid_data, test_data = load_polyvore_outfits(data_dir)

    # Combine all outfits
    all_outfits = {}
    all_outfits.update(train_data)
    all_outfits.update(valid_data)
    all_outfits.update(test_data)

    interactions = []
    timestamp = 1000000

    print("Creating interactions...")
    for outfit_id, outfit_data in tqdm(all_outfits.items()):
        user_id = f"user_{outfit_id}"

        items = outfit_data.get('items', [])

        for item in items:
            # Extract item ID from image URL
            if isinstance(item, dict):
                image_url = item.get('image', '')
                item_id = extract_item_id_from_url(image_url)

                if not item_id:
                    # Fallback to index
                    item_id = str(item.get('index', ''))
            else:
                item_id = str(item)

            if item_id:
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': 1.0,
                    'timestamp': timestamp
                })
                timestamp += 1

    df = pd.DataFrame(interactions)

    print(f"Created {len(df)} interactions")
    print(f"  Unique users: {df['user_id'].nunique()}")
    print(f"  Unique items: {df['item_id'].nunique()}")

    # Save to RecBole format if output path provided
    if output_path:
        save_recbole_inter_file(df, output_path)

    return df


def save_recbole_inter_file(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to RecBole .inter format

    Args:
        df: DataFrame with user_id, item_id, rating, timestamp
        output_path: Path to save .inter file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w') as f:
        # Header with types
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')

        # Data rows
        for _, row in df.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\t{row['rating']}\t{row['timestamp']}\n")

    print(f"Saved RecBole interactions to {output_path}")


def create_recbole_item_file(data_dir: str, output_path: str):
    """
    Create RecBole .item metadata file

    Args:
        data_dir: Path to polyvore data directory
        output_path: Path to save .item file
    """
    metadata = load_item_metadata(data_dir)
    category_mapping = load_category_mapping(data_dir)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w') as f:
        # Header with types
        f.write('item_id:token\tcategory:token\n')

        # Data rows
        for item_id, item_data in metadata.items():
            # Get category
            cat_id = item_data.get('categoryid', 0)
            category = category_mapping.get(cat_id, str(cat_id))

            # Clean category string
            category = str(category).replace('\t', ' ').replace('\n', ' ')
            f.write(f"{item_id}\t{category}\n")

    print(f"Saved RecBole item metadata to {output_path}")


def save_item_metadata_json(data_dir: str, output_path: str):
    """
    Save extracted item metadata to JSON file

    Args:
        data_dir: Path to polyvore data directory
        output_path: Path to save JSON file
    """
    metadata = extract_item_metadata(data_dir)
    category_mapping = load_category_mapping(data_dir)

    # Add category names
    for item_id, item_data in metadata.items():
        cat_id = item_data.get('categoryid', 0)
        item_data['category_name'] = category_mapping.get(cat_id, 'unknown')

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved item metadata to {output_path}")


def validate_data_integrity(data_dir: str, images_dir: Optional[str] = None) -> Dict:
    """
    Validate data integrity

    Args:
        data_dir: Path to polyvore data directory
        images_dir: Optional path to images directory

    Returns:
        Dictionary with validation results
    """
    train_data, valid_data, test_data = load_polyvore_outfits(data_dir)
    all_outfits = {**train_data, **valid_data, **test_data}

    # Collect all item IDs
    all_item_ids = set()
    items_with_url = 0

    for outfit_data in all_outfits.values():
        items = outfit_data.get('items', [])
        for item in items:
            if isinstance(item, dict):
                image_url = item.get('image', '')
                item_id = extract_item_id_from_url(image_url)
                if item_id:
                    all_item_ids.add(item_id)
                    items_with_url += 1

    result = {
        'total_outfits': len(all_outfits),
        'total_items': len(all_item_ids),
        'items_with_url': items_with_url,
    }

    print(f"Validation Results:")
    print(f"  Total outfits: {result['total_outfits']}")
    print(f"  Unique items: {result['total_items']}")
    print(f"  Items with valid URLs: {result['items_with_url']}")

    return result


def get_outfit_splits(data_dir: str) -> Dict[str, List[str]]:
    """
    Get outfit IDs split by train/valid/test

    Args:
        data_dir: Path to polyvore data directory

    Returns:
        Dictionary with 'train', 'valid', 'test' keys mapping to outfit IDs
    """
    train_data, valid_data, test_data = load_polyvore_outfits(data_dir)

    return {
        'train': list(train_data.keys()),
        'valid': list(valid_data.keys()),
        'test': list(test_data.keys()),
    }


def get_item_categories(data_dir: str) -> Dict[str, str]:
    """
    Get mapping of item IDs to categories

    Args:
        data_dir: Path to polyvore data directory

    Returns:
        Dictionary mapping item_id to category
    """
    metadata = load_item_metadata(data_dir)
    category_mapping = load_category_mapping(data_dir)

    categories = {}
    for item_id, item_data in metadata.items():
        cat_id = item_data.get('categoryid', 0)
        categories[item_id] = category_mapping.get(cat_id, 'unknown')

    return categories


if __name__ == "__main__":
    import sys

    # Default data directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/polyvore"

    print(f"Processing Polyvore data from {data_dir}")
    print("=" * 50)

    # Create RecBole format files
    output_inter = os.path.join(data_dir, "polyvore.inter")
    output_item = os.path.join(data_dir, "polyvore.item")
    output_metadata = os.path.join(data_dir, "polyvore_item_metadata.json")

    # Process
    df = create_recbole_interactions(data_dir, output_inter)

    # Create item metadata
    create_recbole_item_file(data_dir, output_item)

    # Save full metadata as JSON
    save_item_metadata_json(data_dir, output_metadata)

    # Validate
    print("\n" + "=" * 50)
    validate_data_integrity(data_dir)

    print("\nDone!")
