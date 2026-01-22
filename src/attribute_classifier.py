"""
CLIP Zero-Shot Attribute Classifier for Style Learning

Uses FashionCLIP to classify items into attribute dimensions:
- Pattern: solid, striped, graphic, logo, textured, checkered
- Style: classic, athletic, streetwear, minimalist, luxury
- Color family: neutral, bright, cool, pastel, dark
- Fit vibe: fitted, relaxed, oversized
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# CLIP prompts for each attribute dimension
ATTRIBUTE_PROMPTS = {
    'pattern': [
        ("solid", "a solid color t-shirt, single color, no pattern, plain"),
        ("striped", "a striped t-shirt with horizontal or vertical stripes"),
        ("graphic", "a t-shirt with graphic print, image, or artwork"),
        ("logo", "a t-shirt with small brand logo or emblem"),
        ("textured", "a textured t-shirt with fabric texture pattern, ribbed"),
        ("checkered", "a checkered or plaid pattern t-shirt"),
    ],
    'style': [
        ("classic", "a classic preppy polo style t-shirt, traditional"),
        ("athletic", "an athletic sporty performance t-shirt, workout wear"),
        ("streetwear", "a streetwear urban style t-shirt, hip hop fashion"),
        ("minimalist", "a minimalist simple clean t-shirt, understated"),
        ("luxury", "a luxury designer high-end t-shirt, premium quality"),
    ],
    'color_family': [
        ("neutral", "a neutral colored t-shirt in white, black, gray, or beige"),
        ("bright", "a bright colorful t-shirt in red, yellow, or orange"),
        ("cool", "a cool colored t-shirt in blue, green, or purple"),
        ("pastel", "a pastel soft colored t-shirt, light pink, baby blue"),
        ("dark", "a dark colored t-shirt in black, navy, or charcoal"),
    ],
    'fit_vibe': [
        ("fitted", "a fitted slim tailored t-shirt, form-fitting"),
        ("relaxed", "a relaxed loose casual t-shirt, comfortable fit"),
        ("oversized", "an oversized baggy t-shirt, extra loose"),
    ]
}

# Weights for combining attribute scores in prediction
ATTRIBUTE_WEIGHTS = {
    'pattern': 0.35,
    'style': 0.25,
    'color_family': 0.20,
    'fit_vibe': 0.10,
    'cluster': 0.10  # Keep cluster as minor signal
}


class AttributeClassifier:
    """Zero-shot attribute classifier using FashionCLIP."""

    def __init__(self, embeddings_path: Optional[str] = None):
        """
        Initialize the classifier.

        Args:
            embeddings_path: Path to pre-computed item embeddings (optional)
        """
        self.embeddings_data = None
        self.text_embeddings = None
        self.clip_model = None
        self.clip_processor = None

        if embeddings_path:
            self.load_embeddings(embeddings_path)

    def load_embeddings(self, embeddings_path: str):
        """Load pre-computed item embeddings."""
        print(f"Loading embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            self.embeddings_data = pickle.load(f)
        print(f"  Loaded {len(self.embeddings_data)} item embeddings")

    def _load_clip_model(self):
        """Load FashionCLIP model for text encoding."""
        if self.clip_model is not None:
            return

        print("Loading FashionCLIP model...")
        try:
            from fashion_clip.fashion_clip import FashionCLIP
            self.clip_model = FashionCLIP('fashion-clip')
            print("  FashionCLIP loaded successfully")
        except ImportError:
            print("  fashion_clip not available, trying transformers...")
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            print("  CLIP model loaded via transformers")

    def _encode_text_prompts(self) -> Dict[str, np.ndarray]:
        """Encode all attribute prompts into embeddings."""
        if self.text_embeddings is not None:
            return self.text_embeddings

        self._load_clip_model()

        print("Encoding attribute prompts...")
        self.text_embeddings = {}

        for attr_name, prompts in ATTRIBUTE_PROMPTS.items():
            # Extract just the descriptions for encoding
            descriptions = [desc for _, desc in prompts]
            labels = [label for label, _ in prompts]

            # Encode using FashionCLIP
            if hasattr(self.clip_model, 'encode_text'):
                # fashion_clip library
                embeddings = self.clip_model.encode_text(descriptions, batch_size=len(descriptions))
                embeddings = np.array(embeddings)
            else:
                # transformers library
                import torch
                inputs = self.clip_processor(text=descriptions, return_tensors="pt", padding=True)
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                    embeddings = text_features.cpu().numpy()

            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            self.text_embeddings[attr_name] = {
                'labels': labels,
                'embeddings': embeddings
            }
            print(f"  {attr_name}: {len(labels)} categories encoded")

        return self.text_embeddings

    def classify_item(self, item_embedding: np.ndarray) -> Dict[str, str]:
        """
        Classify a single item into all attribute dimensions.

        Args:
            item_embedding: 512-dim FashionCLIP embedding

        Returns:
            Dict mapping attribute name to predicted value
        """
        text_embeds = self._encode_text_prompts()

        # Normalize item embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding)

        result = {}
        for attr_name, data in text_embeds.items():
            # Compute similarities
            similarities = item_embedding @ data['embeddings'].T

            # Get best match
            best_idx = np.argmax(similarities)
            result[attr_name] = data['labels'][best_idx]

        return result

    def classify_item_with_scores(self, item_embedding: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Classify item and return scores for all attribute values.

        Args:
            item_embedding: 512-dim FashionCLIP embedding

        Returns:
            Dict mapping attribute name to dict of value -> score
        """
        text_embeds = self._encode_text_prompts()

        # Normalize item embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding)

        result = {}
        for attr_name, data in text_embeds.items():
            # Compute similarities
            similarities = item_embedding @ data['embeddings'].T

            # Softmax to get probabilities
            exp_sims = np.exp(similarities * 10)  # Temperature scaling
            probs = exp_sims / np.sum(exp_sims)

            result[attr_name] = {
                label: float(prob)
                for label, prob in zip(data['labels'], probs)
            }

        return result

    def classify_all_items(
        self,
        embeddings_data: Optional[Dict] = None,
        excluded_items: Optional[set] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Classify all items in the embeddings data.

        Args:
            embeddings_data: Dict of item_id -> embedding or item_id -> {embedding: ...}
            excluded_items: Set of item IDs to skip

        Returns:
            Dict mapping item_id to attribute dict
        """
        if embeddings_data is None:
            embeddings_data = self.embeddings_data

        if embeddings_data is None:
            raise ValueError("No embeddings data provided or loaded")

        excluded_items = excluded_items or set()

        # Pre-encode all text prompts
        self._encode_text_prompts()

        print(f"Classifying {len(embeddings_data)} items...")
        results = {}

        # Get all items to process
        items = [(k, v) for k, v in embeddings_data.items() if k not in excluded_items]

        for item_id, item_data in tqdm(items, desc="Classifying"):
            # Handle both flat embeddings and nested dict format
            if isinstance(item_data, dict):
                embedding = item_data.get('embedding')
            else:
                embedding = item_data

            if embedding is not None:
                results[item_id] = self.classify_item(embedding)

        print(f"  Classified {len(results)} items")
        return results

    def get_attribute_distribution(
        self,
        item_attributes: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Get distribution of attribute values across all items.

        Args:
            item_attributes: Output from classify_all_items

        Returns:
            Dict mapping attribute name to value -> count
        """
        distribution = {}

        for attr_name in ATTRIBUTE_PROMPTS.keys():
            distribution[attr_name] = {}

            for item_id, attrs in item_attributes.items():
                value = attrs.get(attr_name, 'unknown')
                distribution[attr_name][value] = distribution[attr_name].get(value, 0) + 1

        return distribution


def precompute_attributes(
    embeddings_path: str,
    output_path: str,
    exclusions_path: Optional[str] = None
):
    """
    Pre-compute attributes for all items and save to file.

    Args:
        embeddings_path: Path to item embeddings pickle
        output_path: Path to save item attributes pickle
        exclusions_path: Optional path to duplicate exclusions JSON
    """
    # Load exclusions if provided
    excluded_items = set()
    if exclusions_path:
        import json
        try:
            with open(exclusions_path, 'r') as f:
                excluded_items = set(json.load(f))
            print(f"Loaded {len(excluded_items)} exclusions")
        except Exception as e:
            print(f"Warning: Could not load exclusions: {e}")

    # Create classifier and load embeddings
    classifier = AttributeClassifier(embeddings_path)

    # Classify all items
    item_attributes = classifier.classify_all_items(excluded_items=excluded_items)

    # Get distribution for verification
    distribution = classifier.get_attribute_distribution(item_attributes)

    print("\nAttribute Distribution:")
    for attr_name, counts in distribution.items():
        print(f"\n  {attr_name}:")
        for value, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / len(item_attributes) * 100
            print(f"    {value}: {count} ({pct:.1f}%)")

    # Save results
    output = {
        'item_attributes': item_attributes,
        'distribution': distribution,
        'prompts': ATTRIBUTE_PROMPTS
    }

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"Done! Saved attributes for {len(item_attributes)} items")

    return item_attributes


def load_item_attributes(attributes_path: str) -> Dict[str, Dict[str, str]]:
    """Load pre-computed item attributes."""
    with open(attributes_path, 'rb') as f:
        data = pickle.load(f)
    return data['item_attributes']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pre-compute item attributes')
    parser.add_argument(
        '--embeddings',
        default='/home/ubuntu/recSys/outfitTransformer/models/hp_embeddings.pkl',
        help='Path to embeddings pickle'
    )
    parser.add_argument(
        '--output',
        default='/home/ubuntu/recSys/outfitTransformer/models/item_attributes.pkl',
        help='Path to save attributes'
    )
    parser.add_argument(
        '--exclusions',
        default='/home/ubuntu/recSys/outfitTransformer/HPdataset/duplicate_exclusions.json',
        help='Path to duplicate exclusions JSON'
    )

    args = parser.parse_args()

    precompute_attributes(
        embeddings_path=args.embeddings,
        output_path=args.output,
        exclusions_path=args.exclusions
    )
