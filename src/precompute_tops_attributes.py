"""
Pre-compute filterable attributes for tops items from Amazon Fashion dataset.

This script extracts and enriches tops items with:
1. Text-based attributes: color, material, fit, style, neckline, graphics, pattern
2. Visual attributes via FashionCLIP: pattern clusters, graphics detection, style embeddings

Output: data/amazon_fashion/processed/tops_enriched.pkl
"""

import os
import re
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
import torch
from tqdm import tqdm

# ============================================================================
# CATEGORY MAPPING: Outrove Core Types -> Amazon Categories
# ============================================================================

OUTROVE_TO_AMAZON_CATEGORIES = {
    "t-shirts": [
        "T-Shirts",
        "Active Shirts & Tees",
        "Undershirts",
    ],
    "polos": [
        "Polos",
    ],
    "sweaters": [
        "Sweaters",
        "Pullovers",
        "Pullover",
    ],
    "hoodies": [
        "Fashion Hoodies & Sweatshirts",
        "Active Hoodies",
        "Hoodies",
        "Active Sweatshirts",
    ],
    "shirts": [  # Button-downs
        "Casual Button-Down Shirts",
        "Dress Shirts",
        "Button-Down Shirts",
        "Tuxedo Shirts",
    ],
    "sweatshirts": [
        "Fashion Hoodies & Sweatshirts",  # Overlap with hoodies
        "Active Sweatshirts",
    ],
    "quarter-zips": [
        "Pullovers",  # Subset - need text filter for "quarter-zip" or "half-zip"
    ],
    "henleys": [
        "Henleys",
    ],
}

# Flatten for quick lookup
ALL_TOPS_CATEGORIES = set()
for cats in OUTROVE_TO_AMAZON_CATEGORIES.values():
    ALL_TOPS_CATEGORIES.update(cats)

# ============================================================================
# COLOR DEFINITIONS (from Outrove schema)
# ============================================================================

COLORS = {
    # Neutrals
    "white": ["white", "ivory", "cream", "off-white", "offwhite"],
    "beige": ["beige", "khaki", "sand", "camel"],
    "tan": ["tan", "taupe"],
    "gray": ["gray", "grey", "heather gray", "heather grey", "charcoal heather"],
    "charcoal": ["charcoal", "dark gray", "dark grey", "anthracite"],
    "black": ["black", "jet black", "onyx"],

    # Warm colors
    "yellow": ["yellow", "lemon", "gold", "golden"],
    "mustard": ["mustard", "ochre", "amber"],
    "orange": ["orange", "tangerine", "rust", "burnt orange"],
    "coral": ["coral", "salmon", "peach"],
    "red": ["red", "crimson", "scarlet", "ruby", "cherry"],
    "burgundy": ["burgundy", "maroon", "wine", "oxblood", "bordeaux"],
    "pink": ["pink", "rose", "blush", "fuchsia", "magenta", "hot pink"],

    # Cool colors
    "lavender": ["lavender", "lilac", "mauve"],
    "purple": ["purple", "violet", "plum", "eggplant", "grape"],
    "light-blue": ["light blue", "sky blue", "baby blue", "powder blue", "ice blue"],
    "blue": ["blue", "royal blue", "cobalt", "azure", "sapphire"],
    "navy": ["navy", "navy blue", "dark blue", "midnight blue", "indigo"],
    "teal": ["teal", "turquoise", "aqua", "cyan"],
    "mint": ["mint", "seafoam", "sage green"],
    "green": ["green", "emerald", "forest green", "kelly green", "lime"],
    "olive": ["olive", "army green", "military green", "khaki green"],
    "brown": ["brown", "chocolate", "coffee", "espresso", "mocha", "chestnut"],
}

# Build reverse lookup: keyword -> color_id
COLOR_KEYWORDS = {}
for color_id, keywords in COLORS.items():
    for kw in keywords:
        COLOR_KEYWORDS[kw.lower()] = color_id

# ============================================================================
# MATERIAL DEFINITIONS (from Outrove schema)
# ============================================================================

MATERIALS = {
    "cotton": ["cotton", "100% cotton", "cotton blend", "pima cotton", "combed cotton", "ring-spun cotton", "jersey"],
    "polyester": ["polyester", "poly", "100% polyester", "moisture-wicking", "dry-fit", "dri-fit"],
    "wool": ["wool", "merino", "lambswool", "cashmere"],
    "linen": ["linen", "linen blend"],
    "silk": ["silk", "satin"],
    "leather": ["leather", "faux leather", "pu leather", "pleather"],
    "synthetics": ["synthetic", "nylon", "spandex", "elastane", "lycra", "rayon", "viscose", "modal"],
    "fleece": ["fleece", "micro-fleece", "polar fleece"],
    "cashmere": ["cashmere"],
    "merino-wool": ["merino", "merino wool"],
    "cotton-blend": ["cotton blend", "cotton/poly", "cotton polyester", "50/50"],
    "acrylic": ["acrylic"],
}

MATERIAL_KEYWORDS = {}
for mat_id, keywords in MATERIALS.items():
    for kw in keywords:
        MATERIAL_KEYWORDS[kw.lower()] = mat_id

# ============================================================================
# FIT DEFINITIONS
# ============================================================================

FITS = {
    "slim": ["slim fit", "slim-fit", "slimfit", "fitted", "tailored", "athletic fit", "muscle fit"],
    "regular": ["regular fit", "regular-fit", "classic fit", "standard fit", "traditional fit"],
    "relaxed": ["relaxed fit", "relaxed-fit", "loose fit", "loose-fit", "comfort fit", "easy fit"],
    "oversized": ["oversized", "over-sized", "oversize", "boxy", "baggy"],
}

FIT_PATTERNS = []
for fit_id, keywords in FITS.items():
    for kw in keywords:
        FIT_PATTERNS.append((re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE), fit_id))

# ============================================================================
# NECKLINE DEFINITIONS (for T-shirts primarily)
# ============================================================================

NECKLINES = {
    "crew": ["crew neck", "crewneck", "crew-neck", "round neck", "roundneck"],
    "v-neck": ["v-neck", "vneck", "v neck", "v-neckline"],
    "henley": ["henley", "henley neck", "button placket"],
    "scoop": ["scoop neck", "scoop-neck", "scoopneck"],
    "mock": ["mock neck", "mock-neck", "mockneck", "mock turtleneck"],
    "turtleneck": ["turtleneck", "turtle neck", "turtle-neck", "roll neck", "rollneck"],
    "half-zip": ["half-zip", "half zip", "quarter-zip", "quarter zip", "1/4 zip", "1/2 zip"],
}

NECKLINE_PATTERNS = []
for neck_id, keywords in NECKLINES.items():
    for kw in keywords:
        NECKLINE_PATTERNS.append((re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE), neck_id))

# ============================================================================
# STYLE VARIANT DEFINITIONS
# ============================================================================

STYLE_VARIANTS = {
    # T-shirt styles
    "plain": ["plain", "solid", "basic", "essential", "blank"],
    "graphic-tees": ["graphic", "printed", "print tee", "screen print", "novelty"],
    "small-graphics": ["small logo", "chest logo", "pocket logo", "embroidered logo", "mini logo"],
    "pocket-tees": ["pocket tee", "pocket t-shirt", "chest pocket"],
    "athletic": ["athletic", "performance", "sport", "training", "workout", "gym", "active", "dri-fit", "dry-fit", "moisture"],

    # Polo styles
    "classic-cotton": ["pique", "cotton pique", "classic polo", "traditional polo"],
    "performance": ["performance polo", "golf polo", "sport polo", "moisture-wicking polo"],

    # Button-down fabrics
    "oxford": ["oxford", "oxford cloth", "ocbd"],
    "poplin": ["poplin", "broadcloth"],
    "flannel": ["flannel", "plaid flannel", "brushed cotton"],
    "denim": ["denim", "chambray", "jean shirt"],
    "linen": ["linen shirt", "linen blend"],
    "overshirts": ["overshirt", "shirt jacket", "shacket"],

    # Hoodie/sweatshirt styles
    "zip-up": ["zip-up", "zipup", "zip up", "full zip", "full-zip", "zipper"],
    "pullover": ["pullover", "pull-over", "pull over"],

    # Sweater styles
    "knit": ["knit", "knitted", "cable knit", "ribbed"],
}

STYLE_PATTERNS = []
for style_id, keywords in STYLE_VARIANTS.items():
    for kw in keywords:
        STYLE_PATTERNS.append((re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE), style_id))

# ============================================================================
# PATTERN DEFINITIONS (visual patterns on fabric)
# ============================================================================

PATTERNS = {
    "solid": ["solid", "plain", "single color", "one color"],
    "stripes": ["stripe", "striped", "stripes", "pinstripe", "horizontal stripe", "vertical stripe"],
    "plaid": ["plaid", "tartan", "checkered", "check", "gingham", "buffalo plaid", "buffalo check"],
    "dots": ["polka dot", "dots", "dotted", "spotted"],
    "floral": ["floral", "flower", "flowers", "botanical"],
    "geometric": ["geometric", "abstract", "pattern"],
    "camo": ["camo", "camouflage", "military print"],
    "novelty": ["novelty", "graphic", "print", "printed"],
}

PATTERN_PATTERNS = []
for pat_id, keywords in PATTERNS.items():
    for kw in keywords:
        PATTERN_PATTERNS.append((re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE), pat_id))

# ============================================================================
# BRANDING/LOGO DEFINITIONS
# ============================================================================

BRANDING_LEVELS = {
    "no-logos": ["plain", "blank", "no logo", "unbranded", "logoless"],
    "small-logos": ["small logo", "chest logo", "embroidered", "subtle logo", "minimal logo", "tone-on-tone"],
    "branded": ["logo", "branded", "brand name", "designer"],
    "bold-graphics": ["large logo", "big logo", "all-over print", "bold graphic", "statement"],
}

# ============================================================================
# SLEEVE LENGTH
# ============================================================================

SLEEVE_LENGTHS = {
    "short": ["short sleeve", "short-sleeve", "shortsleeve", "s/s"],
    "long": ["long sleeve", "long-sleeve", "longsleeve", "l/s"],
    "sleeveless": ["sleeveless", "tank", "muscle tee"],
    "three-quarter": ["3/4 sleeve", "three-quarter", "3/4-sleeve"],
}

SLEEVE_PATTERNS = []
for sleeve_id, keywords in SLEEVE_LENGTHS.items():
    for kw in keywords:
        SLEEVE_PATTERNS.append((re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE), sleeve_id))

# ============================================================================
# PRICE EXTRACTION
# ============================================================================

PRICE_PATTERN = re.compile(r'\$(\d+(?:\.\d{2})?)')

def extract_price(price_str: str) -> Optional[float]:
    """Extract numeric price from string like '$23.99 - $29.99'"""
    if not price_str:
        return None
    matches = PRICE_PATTERN.findall(price_str)
    if matches:
        # Return the first (lowest) price
        return float(matches[0])
    return None

# ============================================================================
# TEXT ATTRIBUTE EXTRACTION
# ============================================================================

def extract_colors(text: str) -> List[str]:
    """Extract color IDs from text."""
    text_lower = text.lower()
    found_colors = set()

    # Sort keywords by length (longer first) to match "light blue" before "blue"
    sorted_keywords = sorted(COLOR_KEYWORDS.keys(), key=len, reverse=True)

    for kw in sorted_keywords:
        if kw in text_lower:
            found_colors.add(COLOR_KEYWORDS[kw])

    return list(found_colors)

def extract_materials(text: str) -> List[str]:
    """Extract material IDs from text."""
    text_lower = text.lower()
    found_materials = set()

    sorted_keywords = sorted(MATERIAL_KEYWORDS.keys(), key=len, reverse=True)

    for kw in sorted_keywords:
        if kw in text_lower:
            found_materials.add(MATERIAL_KEYWORDS[kw])

    return list(found_materials)

def extract_fit(text: str) -> Optional[str]:
    """Extract fit type from text."""
    for pattern, fit_id in FIT_PATTERNS:
        if pattern.search(text):
            return fit_id
    return None

def extract_neckline(text: str) -> Optional[str]:
    """Extract neckline type from text."""
    for pattern, neck_id in NECKLINE_PATTERNS:
        if pattern.search(text):
            return neck_id
    return None

def extract_styles(text: str) -> List[str]:
    """Extract style variants from text."""
    found_styles = set()
    for pattern, style_id in STYLE_PATTERNS:
        if pattern.search(text):
            found_styles.add(style_id)
    return list(found_styles)

def extract_pattern(text: str) -> Optional[str]:
    """Extract fabric pattern from text."""
    for pattern, pat_id in PATTERN_PATTERNS:
        if pattern.search(text):
            return pat_id
    return None

def extract_sleeve_length(text: str) -> Optional[str]:
    """Extract sleeve length from text."""
    for pattern, sleeve_id in SLEEVE_PATTERNS:
        if pattern.search(text):
            return sleeve_id
    return None

def detect_graphics_from_text(text: str) -> str:
    """Detect graphics/branding level from text."""
    text_lower = text.lower()

    # Check for bold graphics indicators
    bold_indicators = ["graphic tee", "graphic t-shirt", "all-over print", "large print", "bold print", "novelty"]
    for ind in bold_indicators:
        if ind in text_lower:
            return "bold-graphics"

    # Check for small logo indicators
    small_indicators = ["embroidered", "chest logo", "small logo", "pocket logo", "embroidery"]
    for ind in small_indicators:
        if ind in text_lower:
            return "small-logos"

    # Check for general branding
    brand_indicators = ["logo", "branded", "brand"]
    for ind in brand_indicators:
        if ind in text_lower:
            return "branded"

    # Check for plain indicators
    plain_indicators = ["plain", "solid", "basic", "blank", "essential"]
    for ind in plain_indicators:
        if ind in text_lower:
            return "no-logos"

    return "unknown"

def map_amazon_category_to_outrove(amazon_category: List[str]) -> Optional[str]:
    """Map Amazon category path to Outrove core type."""
    if not amazon_category:
        return None

    # Check last few levels of category
    for level in reversed(amazon_category[-3:]):
        for outrove_type, amazon_cats in OUTROVE_TO_AMAZON_CATEGORIES.items():
            if level in amazon_cats:
                return outrove_type

    return None

def is_tops_item(amazon_category: List[str]) -> bool:
    """Check if item is a tops category."""
    if not amazon_category:
        return False

    for level in amazon_category:
        if level in ALL_TOPS_CATEGORIES:
            return True

    return False

# ============================================================================
# FASHIONCLIP VISUAL ATTRIBUTE EXTRACTION
# ============================================================================

# Text queries for FashionCLIP to create style embeddings
VISUAL_QUERIES = {
    # Pattern detection
    "pattern_solid": "solid color plain fabric clothing",
    "pattern_stripes": "striped pattern fabric clothing horizontal vertical stripes",
    "pattern_plaid": "plaid tartan checkered pattern fabric clothing",
    "pattern_dots": "polka dot spotted pattern fabric clothing",
    "pattern_floral": "floral flower botanical print fabric clothing",
    "pattern_camo": "camouflage military camo pattern fabric clothing",

    # Graphics detection
    "graphics_none": "plain solid basic minimalist t-shirt no graphics",
    "graphics_small": "small chest logo embroidered minimal branding t-shirt",
    "graphics_large": "large graphic print bold design t-shirt front print",

    # Fit detection (visual cues)
    "fit_slim": "slim fitted tight athletic body-hugging clothing",
    "fit_regular": "regular classic standard fit clothing",
    "fit_relaxed": "relaxed loose comfortable oversized baggy clothing",

    # Neckline detection
    "neck_crew": "crew neck round neckline t-shirt",
    "neck_vneck": "v-neck deep neckline t-shirt",
    "neck_henley": "henley button placket neckline shirt",
    "neck_polo": "polo collar button placket shirt",

    # Style detection
    "style_casual": "casual everyday relaxed streetwear clothing",
    "style_athletic": "athletic sporty performance workout activewear",
    "style_formal": "formal dress professional business attire",
}

def load_fashionclip():
    """Load FashionCLIP model."""
    try:
        from fashion_clip.fashion_clip import FashionCLIP
        fclip = FashionCLIP('fashion-clip')
        return fclip
    except ImportError:
        print("Warning: fashion-clip not installed. Visual features will be skipped.")
        return None

def compute_query_embeddings(fclip) -> Dict[str, np.ndarray]:
    """Pre-compute embeddings for all visual queries."""
    if fclip is None:
        return {}

    print("Computing FashionCLIP query embeddings...")
    query_embeddings = {}

    for query_name, query_text in tqdm(VISUAL_QUERIES.items()):
        embedding = fclip.encode_text([query_text], batch_size=1)[0]
        query_embeddings[query_name] = embedding / np.linalg.norm(embedding)  # Normalize

    return query_embeddings

def compute_visual_attributes(item_embedding: np.ndarray, query_embeddings: Dict[str, np.ndarray]) -> Dict[str, any]:
    """Compute visual attributes by comparing item embedding to query embeddings."""
    if not query_embeddings:
        return {}

    # Normalize item embedding
    item_norm = item_embedding / np.linalg.norm(item_embedding)

    # Compute similarity scores for each query
    scores = {}
    for query_name, query_emb in query_embeddings.items():
        scores[query_name] = float(np.dot(item_norm, query_emb))

    # Determine pattern (highest pattern score)
    pattern_scores = {k: v for k, v in scores.items() if k.startswith("pattern_")}
    if pattern_scores:
        visual_pattern = max(pattern_scores, key=pattern_scores.get).replace("pattern_", "")
        visual_pattern_confidence = pattern_scores[f"pattern_{visual_pattern}"]
    else:
        visual_pattern = None
        visual_pattern_confidence = 0.0

    # Determine graphics level
    graphics_scores = {k: v for k, v in scores.items() if k.startswith("graphics_")}
    if graphics_scores:
        visual_graphics = max(graphics_scores, key=graphics_scores.get).replace("graphics_", "")
        visual_graphics_confidence = graphics_scores[f"graphics_{visual_graphics}"]
    else:
        visual_graphics = None
        visual_graphics_confidence = 0.0

    # Determine fit
    fit_scores = {k: v for k, v in scores.items() if k.startswith("fit_")}
    if fit_scores:
        visual_fit = max(fit_scores, key=fit_scores.get).replace("fit_", "")
        visual_fit_confidence = fit_scores[f"fit_{visual_fit}"]
    else:
        visual_fit = None
        visual_fit_confidence = 0.0

    # Determine neckline
    neck_scores = {k: v for k, v in scores.items() if k.startswith("neck_")}
    if neck_scores:
        visual_neckline = max(neck_scores, key=neck_scores.get).replace("neck_", "")
        visual_neckline_confidence = neck_scores[f"neck_{visual_neckline}"]
    else:
        visual_neckline = None
        visual_neckline_confidence = 0.0

    # Determine style
    style_scores = {k: v for k, v in scores.items() if k.startswith("style_")}
    if style_scores:
        visual_style = max(style_scores, key=style_scores.get).replace("style_", "")
        visual_style_confidence = style_scores[f"style_{visual_style}"]
    else:
        visual_style = None
        visual_style_confidence = 0.0

    return {
        "visual_pattern": visual_pattern,
        "visual_pattern_confidence": visual_pattern_confidence,
        "visual_graphics": visual_graphics,
        "visual_graphics_confidence": visual_graphics_confidence,
        "visual_fit": visual_fit,
        "visual_fit_confidence": visual_fit_confidence,
        "visual_neckline": visual_neckline,
        "visual_neckline_confidence": visual_neckline_confidence,
        "visual_style": visual_style,
        "visual_style_confidence": visual_style_confidence,
        "all_visual_scores": scores,
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_amazon_tops(
    metadata_path: str,
    embeddings_path: str,
    output_path: str,
    use_visual: bool = True,
):
    """
    Process Amazon Fashion metadata to extract tops items with enriched attributes.

    Args:
        metadata_path: Path to item_metadata.pkl
        embeddings_path: Path to amazon_mens_embeddings.pkl
        output_path: Path to save enriched tops data
        use_visual: Whether to compute visual attributes using FashionCLIP
    """
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded {len(metadata)} items")

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)

    # Handle different embedding file formats
    if isinstance(embeddings_data, dict):
        if 'embeddings' in embeddings_data:
            # Format: {'embeddings': array, 'item_ids': list}
            embeddings = embeddings_data['embeddings']
            item_ids = embeddings_data['item_ids']
            item_to_embedding = {item_id: embeddings[idx] for idx, item_id in enumerate(item_ids)}
        else:
            # Format: {item_id: embedding} - direct mapping
            item_to_embedding = embeddings_data
            item_ids = list(item_to_embedding.keys())
    else:
        raise ValueError(f"Unknown embeddings format: {type(embeddings_data)}")

    print(f"Loaded {len(item_to_embedding)} embeddings")

    # Load FashionCLIP and compute query embeddings
    query_embeddings = {}
    if use_visual:
        fclip = load_fashionclip()
        query_embeddings = compute_query_embeddings(fclip)

    # Filter and process tops items
    print("\nProcessing tops items...")
    tops_items = {}
    tops_embeddings = []
    tops_item_ids = []

    category_counts = defaultdict(int)

    for item_id, meta in tqdm(metadata.items()):
        amazon_category = meta.get('category', [])

        # Check if this is a tops item
        if not is_tops_item(amazon_category):
            continue

        # Map to Outrove type
        outrove_type = map_amazon_category_to_outrove(amazon_category)
        if not outrove_type:
            continue

        category_counts[outrove_type] += 1

        # Combine title and description for text analysis
        title = meta.get('title', '')
        description = meta.get('description', '')
        combined_text = f"{title} {description}"

        # Extract text-based attributes
        enriched = {
            # Original metadata
            "item_id": item_id,
            "title": title,
            "brand": meta.get('brand', ''),
            "description": description,
            "amazon_category": amazon_category,
            "price_raw": meta.get('price', ''),

            # Mapped type
            "outrove_type": outrove_type,

            # Extracted attributes
            "colors": extract_colors(combined_text),
            "materials": extract_materials(combined_text),
            "fit": extract_fit(combined_text),
            "neckline": extract_neckline(combined_text),
            "styles": extract_styles(combined_text),
            "pattern": extract_pattern(combined_text),
            "sleeve_length": extract_sleeve_length(combined_text),
            "graphics_level": detect_graphics_from_text(combined_text),
            "price": extract_price(meta.get('price', '')),
        }

        # Add visual attributes if embedding exists
        if use_visual and item_id in item_to_embedding:
            item_embedding = item_to_embedding[item_id]
            visual_attrs = compute_visual_attributes(item_embedding, query_embeddings)
            enriched.update(visual_attrs)

            # Store embedding for this tops item
            tops_embeddings.append(item_embedding)
            tops_item_ids.append(item_id)

        tops_items[item_id] = enriched

    print(f"\nExtracted {len(tops_items)} tops items")
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Compute statistics
    stats = {
        "total_items": len(tops_items),
        "items_with_colors": sum(1 for item in tops_items.values() if item['colors']),
        "items_with_materials": sum(1 for item in tops_items.values() if item['materials']),
        "items_with_fit": sum(1 for item in tops_items.values() if item['fit']),
        "items_with_neckline": sum(1 for item in tops_items.values() if item['neckline']),
        "items_with_pattern": sum(1 for item in tops_items.values() if item['pattern']),
        "items_with_price": sum(1 for item in tops_items.values() if item['price']),
        "items_with_visual": sum(1 for item in tops_items.values() if item.get('visual_pattern')),
        "category_counts": dict(category_counts),
    }

    print("\nAttribute coverage:")
    for key, value in stats.items():
        if key != "category_counts" and key != "total_items":
            pct = value / stats["total_items"] * 100 if stats["total_items"] > 0 else 0
            print(f"  {key}: {value} ({pct:.1f}%)")

    # Save enriched data
    output_data = {
        "items": tops_items,
        "embeddings": np.array(tops_embeddings) if tops_embeddings else None,
        "item_ids": tops_item_ids,
        "query_embeddings": query_embeddings,
        "stats": stats,
        "version": "1.0",
        "outrove_types": list(OUTROVE_TO_AMAZON_CATEGORIES.keys()),
        "color_options": list(COLORS.keys()),
        "material_options": list(MATERIALS.keys()),
        "fit_options": list(FITS.keys()),
        "neckline_options": list(NECKLINES.keys()),
        "pattern_options": list(PATTERNS.keys()),
        "style_options": list(STYLE_VARIANTS.keys()),
    }

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    # Calculate file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {file_size:.1f} MB")

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-compute tops attributes for Amazon Fashion")
    parser.add_argument("--metadata", default="data/amazon_fashion/processed/item_metadata.pkl",
                       help="Path to item metadata")
    parser.add_argument("--embeddings", default="data/amazon_fashion/processed/amazon_mens_embeddings.pkl",
                       help="Path to embeddings")
    parser.add_argument("--output", default="data/amazon_fashion/processed/tops_enriched.pkl",
                       help="Output path")
    parser.add_argument("--no-visual", action="store_true",
                       help="Skip visual attribute extraction")

    args = parser.parse_args()

    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    metadata_path = os.path.join(project_root, args.metadata)
    embeddings_path = os.path.join(project_root, args.embeddings)
    output_path = os.path.join(project_root, args.output)

    process_amazon_tops(
        metadata_path=metadata_path,
        embeddings_path=embeddings_path,
        output_path=output_path,
        use_visual=not args.no_visual,
    )
