"""
CLIP Zero-Shot Attribute Classifier for Women's Fashion

Uses FashionCLIP to classify women's items into attribute dimensions:
- Pattern: solid, striped, floral, geometric, animal_print, plaid, polka_dots, lace
- Style: casual, office, evening, bohemian, minimalist, romantic, athletic
- Neckline: crew, v_neck, scoop, off_shoulder, sweetheart, halter, square, turtleneck, cowl
- Sleeve Type: sleeveless, short_sleeve, long_sleeve, puff_sleeve, bell_sleeve, flutter_sleeve
- Fit/Vibe: fitted, relaxed, oversized, cropped, flowy
- Occasion: everyday, work, date_night, party, beach
- Color Family: neutral, bright, cool, pastel, dark

Dress-specific:
- Dress Length: mini, midi, maxi
- Dress Silhouette: bodycon, a_line, wrap, shift

Bottoms-specific:
- Bottom Fit: skinny, straight, wide_leg, bootcut
- Bottom Rise: high_rise, mid_rise, low_rise
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


# Women-specific CLIP prompts for each attribute dimension
WOMEN_ATTRIBUTE_PROMPTS = {
    'pattern': [
        ("solid", "a solid color women's top, single color, no pattern, plain"),
        ("striped", "a striped women's garment with horizontal or vertical stripes"),
        ("floral", "a floral print women's clothing with flowers, botanical pattern"),
        ("geometric", "a geometric pattern women's clothing with shapes, abstract"),
        ("animal_print", "an animal print women's top, leopard, zebra, snake pattern"),
        ("plaid", "a plaid or checkered women's clothing, tartan pattern"),
        ("polka_dots", "a polka dot women's garment with round dots"),
        ("lace", "a lace pattern women's top with delicate lace fabric"),
    ],
    'style': [
        ("casual", "a casual everyday women's top, relaxed informal wear"),
        ("office", "a professional office women's blouse, business wear"),
        ("evening", "an elegant evening women's top, dressy sophisticated"),
        ("bohemian", "a bohemian boho style women's top, free-spirited artistic"),
        ("minimalist", "a minimalist clean women's garment, simple understated"),
        ("romantic", "a romantic feminine women's blouse, soft delicate"),
        ("athletic", "an athletic sporty women's top, performance activewear"),
    ],
    'neckline': [
        ("crew", "a crew neck round neckline women's top"),
        ("v_neck", "a v-neck deep neckline women's top"),
        ("scoop", "a scoop neck wide rounded neckline women's top"),
        ("off_shoulder", "an off-shoulder bare shoulders women's top"),
        ("sweetheart", "a sweetheart neckline curved heart shape women's top"),
        ("halter", "a halter neckline ties at neck women's top"),
        ("square", "a square neckline straight across women's top"),
        ("turtleneck", "a turtleneck high neck women's top"),
        ("cowl", "a cowl neck draped neckline women's top"),
    ],
    'sleeve_type': [
        ("sleeveless", "a sleeveless women's top with no sleeves"),
        ("short_sleeve", "a short sleeve women's top"),
        ("long_sleeve", "a long sleeve women's top"),
        ("puff_sleeve", "a puff sleeve women's blouse with voluminous sleeves"),
        ("bell_sleeve", "a bell sleeve women's top with flared wide sleeves"),
        ("flutter_sleeve", "a flutter sleeve women's top with ruffled flowing sleeves"),
    ],
    'fit_vibe': [
        ("fitted", "a fitted form-fitting slim women's top"),
        ("relaxed", "a relaxed loose comfortable women's top"),
        ("oversized", "an oversized baggy extra loose women's top"),
        ("cropped", "a cropped shorter length above waist women's top"),
        ("flowy", "a flowy drapey soft feminine women's top"),
    ],
    'occasion': [
        ("everyday", "an everyday casual daily wear women's top"),
        ("work", "a work appropriate professional women's top"),
        ("date_night", "a date night romantic evening women's top"),
        ("party", "a party celebration festive women's top"),
        ("beach", "a beach vacation resort wear women's top"),
    ],
    'color_family': [
        ("neutral", "a neutral colored women's top in white, black, gray, beige, cream"),
        ("bright", "a bright colorful women's top in red, yellow, orange"),
        ("cool", "a cool colored women's top in blue, green, purple"),
        ("pastel", "a pastel soft colored women's top, light pink, baby blue, lavender"),
        ("dark", "a dark colored women's top in black, navy, charcoal, dark green"),
    ],
    # Dress-specific attributes
    'dress_length': [
        ("mini", "a mini dress above knee short length"),
        ("midi", "a midi dress below knee mid-calf length"),
        ("maxi", "a maxi dress floor length long dress"),
    ],
    'dress_silhouette': [
        ("bodycon", "a bodycon tight fitted body-hugging dress"),
        ("a_line", "an a-line dress fitted waist flared skirt"),
        ("wrap", "a wrap dress with tie closure surplice style"),
        ("shift", "a shift dress loose straight boxy silhouette"),
    ],
    # Bottoms-specific attributes
    'bottom_fit': [
        ("skinny", "skinny tight fitted jeans or pants"),
        ("straight", "straight leg pants classic fit"),
        ("wide_leg", "wide leg pants palazzo flared"),
        ("bootcut", "bootcut jeans slightly flared at bottom"),
    ],
    'bottom_rise': [
        ("high_rise", "high rise waist above navel pants"),
        ("mid_rise", "mid rise waist at navel pants"),
        ("low_rise", "low rise waist below navel pants"),
    ],
}

# Weights for combining attribute scores in prediction
WOMEN_ATTRIBUTE_WEIGHTS = {
    'pattern': 0.25,
    'style': 0.20,
    'color_family': 0.15,
    'fit_vibe': 0.15,
    'neckline': 0.10,
    'occasion': 0.10,
    'sleeve_type': 0.05,
}

# Category-specific attribute selection
CATEGORY_ATTRIBUTES = {
    'tops_knitwear': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family', 'sleeve_type'],
    'tops_woven': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family', 'sleeve_type'],
    'tops_sleeveless': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family'],
    'tops_special': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family'],
    'dresses': ['pattern', 'style', 'neckline', 'occasion', 'color_family', 'dress_length', 'dress_silhouette'],
    'bottoms_trousers': ['pattern', 'color_family', 'bottom_fit', 'bottom_rise'],
    'bottoms_skorts': ['pattern', 'color_family', 'fit_vibe'],
    'bottoms_skirts': ['pattern', 'color_family', 'fit_vibe'],
    'outerwear': ['pattern', 'style', 'fit_vibe', 'color_family'],
    'sportswear': ['pattern', 'color_family', 'fit_vibe', 'occasion'],
}


class WomenAttributeClassifier:
    """Zero-shot attribute classifier for women's fashion using FashionCLIP."""

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

    def _encode_text_prompts(self) -> Dict[str, Dict]:
        """Encode all attribute prompts into embeddings."""
        if self.text_embeddings is not None:
            return self.text_embeddings

        self._load_clip_model()

        print("Encoding attribute prompts...")
        self.text_embeddings = {}

        for attr_name, prompts in WOMEN_ATTRIBUTE_PROMPTS.items():
            # Extract descriptions for encoding
            descriptions = [desc for _, desc in prompts]
            labels = [label for label, _ in prompts]

            # Encode using FashionCLIP
            if hasattr(self.clip_model, 'encode_text'):
                embeddings = self.clip_model.encode_text(descriptions, batch_size=len(descriptions))
                embeddings = np.array(embeddings)
            else:
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

    def classify_item(self, item_embedding: np.ndarray, category: str = None) -> Dict[str, str]:
        """
        Classify a single item into all relevant attribute dimensions.

        Args:
            item_embedding: 512-dim FashionCLIP embedding
            category: Optional category to select relevant attributes

        Returns:
            Dict mapping attribute name to predicted value
        """
        text_embeds = self._encode_text_prompts()

        # Normalize item embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding)

        # Select relevant attributes for category
        if category and category in CATEGORY_ATTRIBUTES:
            relevant_attrs = CATEGORY_ATTRIBUTES[category]
        else:
            relevant_attrs = list(WOMEN_ATTRIBUTE_PROMPTS.keys())

        result = {}
        for attr_name in relevant_attrs:
            if attr_name not in text_embeds:
                continue

            data = text_embeds[attr_name]
            # Compute similarities
            similarities = item_embedding @ data['embeddings'].T

            # Get best match
            best_idx = np.argmax(similarities)
            result[attr_name] = data['labels'][best_idx]

        return result

    def classify_item_with_scores(self, item_embedding: np.ndarray, category: str = None) -> Dict[str, Dict[str, float]]:
        """
        Classify item and return scores for all attribute values.

        Args:
            item_embedding: 512-dim FashionCLIP embedding
            category: Optional category to select relevant attributes

        Returns:
            Dict mapping attribute name to dict of value -> score
        """
        text_embeds = self._encode_text_prompts()

        # Normalize item embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding)

        # Select relevant attributes
        if category and category in CATEGORY_ATTRIBUTES:
            relevant_attrs = CATEGORY_ATTRIBUTES[category]
        else:
            relevant_attrs = list(WOMEN_ATTRIBUTE_PROMPTS.keys())

        result = {}
        for attr_name in relevant_attrs:
            if attr_name not in text_embeds:
                continue

            data = text_embeds[attr_name]
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
            embeddings_data: Dict of item_id -> {embedding, category, ...}
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

        items = [(k, v) for k, v in embeddings_data.items() if k not in excluded_items]

        for item_id, item_data in tqdm(items, desc="Classifying"):
            # Handle both flat embeddings and nested dict format
            if isinstance(item_data, dict):
                embedding = item_data.get('embedding')
                category = item_data.get('category', '')
            else:
                embedding = item_data
                category = ''

            if embedding is not None:
                results[item_id] = self.classify_item(embedding, category)
                results[item_id]['category'] = category

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

        all_attrs = set()
        for item_attrs in item_attributes.values():
            all_attrs.update(item_attrs.keys())

        for attr_name in all_attrs:
            if attr_name == 'category':
                continue
            distribution[attr_name] = {}

            for item_id, attrs in item_attributes.items():
                value = attrs.get(attr_name, 'unknown')
                distribution[attr_name][value] = distribution[attr_name].get(value, 0) + 1

        return distribution


def precompute_women_attributes(
    embeddings_path: str,
    output_path: str,
    exclusions_path: Optional[str] = None
):
    """
    Pre-compute attributes for all women's items and save to file.

    Args:
        embeddings_path: Path to women's embeddings pickle
        output_path: Path to save item attributes pickle
        exclusions_path: Optional path to exclusions JSON
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
    classifier = WomenAttributeClassifier(embeddings_path)

    # Classify all items
    item_attributes = classifier.classify_all_items(excluded_items=excluded_items)

    # Get distribution for verification
    distribution = classifier.get_attribute_distribution(item_attributes)

    print("\nAttribute Distribution:")
    for attr_name, counts in sorted(distribution.items()):
        print(f"\n  {attr_name}:")
        for value, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            pct = count / len(item_attributes) * 100
            print(f"    {value}: {count} ({pct:.1f}%)")

    # Save results
    output = {
        'item_attributes': item_attributes,
        'distribution': distribution,
        'prompts': WOMEN_ATTRIBUTE_PROMPTS,
        'category_attributes': CATEGORY_ATTRIBUTES,
    }

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"Done! Saved attributes for {len(item_attributes)} items")

    return item_attributes


def load_women_attributes(attributes_path: str) -> Dict[str, Dict[str, str]]:
    """Load pre-computed women's item attributes."""
    with open(attributes_path, 'rb') as f:
        data = pickle.load(f)
    return data['item_attributes']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Pre-compute women's item attributes")
    parser.add_argument(
        '--embeddings',
        default='/home/ubuntu/recSys/outfitTransformer/models/women_embeddings.pkl',
        help='Path to embeddings pickle'
    )
    parser.add_argument(
        '--output',
        default='/home/ubuntu/recSys/outfitTransformer/models/women_item_attributes.pkl',
        help='Path to save attributes'
    )
    parser.add_argument(
        '--exclusions',
        default=None,
        help='Path to exclusions JSON (optional)'
    )

    args = parser.parse_args()

    precompute_women_attributes(
        embeddings_path=args.embeddings,
        output_path=args.output,
        exclusions_path=args.exclusions
    )
