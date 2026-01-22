"""
Sequential Data Processing for BERT4Rec
Converts Polyvore-U dataset to RecBole sequential format
"""
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


def load_polyvore_u(data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load Polyvore-U dataset

    Args:
        data_dir: Path to polyvore_u data directory

    Returns:
        Tuple of (train_data, valid_data, test_data)
    """
    data_path = Path(data_dir)

    train = np.load(data_path / 'train.npy', allow_pickle=True).item()
    valid = np.load(data_path / 'valid_grd_dict.npy', allow_pickle=True).item()
    test = np.load(data_path / 'test_grd_dict.npy', allow_pickle=True).item()

    return train, valid, test


def convert_to_sequential_format(
    train_data: Dict,
    output_path: str,
    max_seq_length: int = 50
) -> Dict[str, List]:
    """
    Convert Polyvore-U to RecBole sequential format

    For BERT4Rec, we need user sequences where each user has
    a list of items they've interacted with in order.

    Strategy: For each user, flatten all outfit items into a single
    sequence ordered by outfit_id (temporal order).

    Args:
        train_data: Polyvore-U training data dict
        output_path: Path to save .inter file
        max_seq_length: Maximum sequence length

    Returns:
        Dict with statistics
    """
    # Group interactions by user
    user_sequences = defaultdict(list)

    uids = train_data['uids']
    oids = train_data['oids']
    outfits = train_data['outfits']

    # Build user -> [(oid, outfit_items), ...] mapping
    user_outfit_map = defaultdict(list)
    for uid, oid, outfit in zip(uids, oids, outfits):
        user_outfit_map[uid].append((oid, outfit))

    # Sort each user's outfits by oid (temporal order) and flatten items
    for uid, outfit_list in user_outfit_map.items():
        # Sort by outfit_id
        outfit_list_sorted = sorted(outfit_list, key=lambda x: x[0])

        # Flatten items maintaining outfit order
        items_sequence = []
        for oid, items in outfit_list_sorted:
            items_sequence.extend(items)

        # Truncate to max length (keep most recent)
        if len(items_sequence) > max_seq_length:
            items_sequence = items_sequence[-max_seq_length:]

        user_sequences[uid] = items_sequence

    # Convert to RecBole .inter format
    # Each line: user_id, item_id, timestamp
    interactions = []
    base_timestamp = 1000000

    for uid, item_sequence in tqdm(user_sequences.items(), desc="Converting"):
        for idx, item_id in enumerate(item_sequence):
            interactions.append({
                'user_id': f'user_{uid}',
                'item_id': str(item_id),
                'timestamp': base_timestamp + idx
            })
        base_timestamp += max_seq_length + 100  # Gap between users

    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        for inter in interactions:
            f.write(f"{inter['user_id']}\t{inter['item_id']}\t{inter['timestamp']}\n")

    stats = {
        'num_users': len(user_sequences),
        'num_interactions': len(interactions),
        'num_unique_items': len(set(i['item_id'] for i in interactions)),
        'avg_seq_length': sum(len(s) for s in user_sequences.values()) / len(user_sequences),
        'output_path': output_path
    }

    print(f"Created RecBole .inter file:")
    print(f"  Users: {stats['num_users']}")
    print(f"  Interactions: {stats['num_interactions']}")
    print(f"  Unique items: {stats['num_unique_items']}")
    print(f"  Avg sequence length: {stats['avg_seq_length']:.1f}")
    print(f"  Saved to: {output_path}")

    return stats


def create_item_mapping(
    data_dir: str,
    output_path: str
) -> Dict[int, str]:
    """
    Create mapping from Polyvore-U item IDs to our existing item IDs

    This is needed if we want to link with FashionCLIP embeddings.

    Args:
        data_dir: Path to polyvore_u data
        output_path: Path to save mapping

    Returns:
        Dict mapping polyvore_u item_id -> original tid
    """
    import json

    # Load the map files if they exist
    map_dir = Path(data_dir) / 'map'

    if map_dir.exists():
        # Check for existing mappings
        mapping = {}
        for f in map_dir.iterdir():
            if f.suffix == '.npy':
                data = np.load(f, allow_pickle=True)
                print(f"Found mapping file: {f.name}")

        return mapping

    return {}


def convert_polyvore_u_to_recbole(
    data_dir: str = "data/polyvore_u",
    output_dir: str = "data/polyvore_u_recbole",
    max_seq_length: int = 50
) -> Dict:
    """
    Main function to convert Polyvore-U to RecBole format

    Args:
        data_dir: Path to polyvore_u data
        output_dir: Path to save RecBole format data
        max_seq_length: Maximum sequence length for BERT4Rec

    Returns:
        Statistics dict
    """
    print("Loading Polyvore-U dataset...")
    train_data, valid_data, test_data = load_polyvore_u(data_dir)

    print(f"\nDataset statistics:")
    print(f"  Training records: {len(train_data['uids'])}")
    print(f"  Unique users: {len(set(train_data['uids']))}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert training data
    output_path = os.path.join(output_dir, "polyvore_u.inter")
    stats = convert_to_sequential_format(train_data, output_path, max_seq_length)

    # Create user file (optional but helpful)
    user_path = os.path.join(output_dir, "polyvore_u.user")
    unique_users = set(train_data['uids'])
    with open(user_path, 'w') as f:
        f.write('user_id:token\n')
        for uid in unique_users:
            f.write(f'user_{uid}\n')
    print(f"  User file saved to: {user_path}")

    # Create item file
    all_items = set()
    for outfit in train_data['outfits']:
        all_items.update(outfit)

    item_path = os.path.join(output_dir, "polyvore_u.item")
    with open(item_path, 'w') as f:
        f.write('item_id:token\n')
        for item_id in all_items:
            f.write(f'{item_id}\n')
    print(f"  Item file saved to: {item_path}")

    stats['user_file'] = user_path
    stats['item_file'] = item_path

    return stats


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/polyvore_u"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/polyvore_u_recbole"

    stats = convert_polyvore_u_to_recbole(data_dir, output_dir)

    print("\n" + "="*50)
    print("Conversion complete!")
    print("="*50)
    print(f"\nTo train BERT4Rec, run:")
    print(f"  python src/train_models.py BERT4Rec {output_dir} polyvore_u")
