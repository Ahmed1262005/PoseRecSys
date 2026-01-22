"""
Gender-Aware Configuration for Fashion Style Learning

Provides unified configuration for both men's and women's fashion style learning.
"""

from pathlib import Path
from typing import Dict, List

# Base directory
BASE_DIR = Path("/home/ubuntu/recSys/outfitTransformer")


# === Men's Configuration ===
MENS_CONFIG = {
    'embeddings_path': BASE_DIR / "models" / "hp_embeddings.pkl",
    'attributes_path': BASE_DIR / "models" / "item_attributes.pkl",
    'images_dir': BASE_DIR / "HPImages",

    'category_order': [
        'Graphics T-shirts',
        'Plain T-shirts',
        'Small logos',
        'Athletic'
    ],

    'attribute_prompts': {
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
    },

    'attribute_weights': {
        'pattern': 0.35,
        'style': 0.25,
        'color_family': 0.20,
        'fit_vibe': 0.10,
        'cluster': 0.10
    }
}


# === Women's Configuration ===
WOMENS_CONFIG = {
    'embeddings_path': BASE_DIR / "models" / "women_embeddings.pkl",
    'attributes_path': BASE_DIR / "models" / "women_item_attributes.pkl",
    'images_dir': BASE_DIR / "data" / "women_fashion" / "images_webp",
    'exclusions_path': BASE_DIR / "data" / "women_fashion" / "processed" / "duplicate_exclusions.json",

    'category_order': [
        # Tops first
        'tops_knitwear',
        'tops_woven',
        'tops_sleeveless',
        'tops_special',
        # Then dresses
        'dresses',
        # Then bottoms
        'bottoms_trousers',
        'bottoms_skorts',
        # Outerwear
        'outerwear',
        # Sportswear
        'sportswear',
    ],

    'attribute_prompts': {
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
    },

    'attribute_weights': {
        'pattern': 0.25,
        'style': 0.20,
        'color_family': 0.15,
        'fit_vibe': 0.15,
        'neckline': 0.10,
        'occasion': 0.10,
        'sleeve_type': 0.05,
    },

    # Category-specific attributes
    'category_attributes': {
        'tops_knitwear': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family', 'sleeve_type'],
        'tops_woven': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family', 'sleeve_type'],
        'tops_sleeveless': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family'],
        'tops_special': ['pattern', 'style', 'neckline', 'fit_vibe', 'color_family'],
        'dresses': ['pattern', 'style', 'neckline', 'occasion', 'color_family'],
        'bottoms_trousers': ['pattern', 'color_family', 'fit_vibe'],
        'bottoms_skorts': ['pattern', 'color_family', 'fit_vibe'],
        'outerwear': ['pattern', 'style', 'fit_vibe', 'color_family'],
        'sportswear': ['pattern', 'color_family', 'fit_vibe', 'occasion'],
    }
}


def get_config(gender: str = "male") -> Dict:
    """Get configuration for specified gender."""
    if gender.lower() in ("female", "women", "woman", "f"):
        return WOMENS_CONFIG
    else:
        return MENS_CONFIG


def get_embeddings_path(gender: str = "male") -> Path:
    """Get embeddings path for specified gender."""
    return get_config(gender)['embeddings_path']


def get_attributes_path(gender: str = "male") -> Path:
    """Get attributes path for specified gender."""
    return get_config(gender)['attributes_path']


def get_category_order(gender: str = "male") -> List[str]:
    """Get category order for specified gender."""
    return get_config(gender)['category_order']


def get_attribute_prompts(gender: str = "male") -> Dict:
    """Get attribute prompts for specified gender."""
    return get_config(gender)['attribute_prompts']


def get_attribute_weights(gender: str = "male") -> Dict[str, float]:
    """Get attribute weights for specified gender."""
    return get_config(gender)['attribute_weights']
