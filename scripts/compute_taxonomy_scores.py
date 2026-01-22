"""
Compute Taxonomy Scores for HP Dataset

Instead of clustering, assign each item continuous scores on a fixed taxonomy:
1. Archetypes (4 dimensions) - the "why" / identity
2. Visual Anchors (12 dimensions) - the "what you notice first"
3. Attributes (from metadata) - the "how it's executed"

Uses CLIP text embeddings to compute archetype/anchor affinities.
This approach SCALES because taxonomy is fixed, only scores change.
"""

import pickle
import numpy as np
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ============================================
# TAXONOMY DEFINITION
# ============================================

ARCHETYPES = {
    'classic': {
        'description': 'timeless, balanced, restrained, elegant, sophisticated, minimal, refined',
        'prompts': [
            'a classic timeless t-shirt',
            'elegant minimalist menswear',
            'sophisticated simple clothing',
            'refined traditional style',
            'understated luxury basics'
        ]
    },
    'natural_sporty': {
        'description': 'functional, comfortable, performance-driven, athletic, casual, relaxed',
        'prompts': [
            'athletic performance t-shirt',
            'comfortable sporty casual wear',
            'functional activewear',
            'relaxed natural style clothing',
            'outdoor adventure apparel'
        ]
    },
    'dramatic_street': {
        'description': 'bold, expressive, high contrast, edgy, urban, statement-making',
        'prompts': [
            'bold streetwear t-shirt',
            'edgy urban fashion',
            'high contrast statement clothing',
            'expressive street style',
            'dramatic urban menswear'
        ]
    },
    'creative_artistic': {
        'description': 'experimental, unconventional, graphic-led, artistic, avant-garde',
        'prompts': [
            'artistic creative t-shirt design',
            'experimental avant-garde clothing',
            'unconventional graphic menswear',
            'artistic statement fashion',
            'creative designer streetwear'
        ]
    }
}

VISUAL_ANCHORS = {
    # Classic anchors
    'solids': ['solid color garment', 'plain single color clothing', 'monochrome minimalist'],
    'symmetry': ['symmetrical balanced design', 'centered even proportions', 'classic balanced layout'],
    'smooth_surface': ['smooth fabric texture', 'clean polished finish', 'refined surface quality'],

    # Natural/Sporty anchors
    'texture': ['textured fabric clothing', 'visible material texture', 'tactile surface quality'],
    'utility': ['utility functional design', 'practical pocket details', 'functional workwear elements'],
    'athletic_cues': ['athletic performance features', 'sporty technical details', 'activewear construction'],

    # Dramatic/Street anchors
    'contrast': ['high contrast design', 'bold black and white', 'stark color contrast'],
    'typography': ['bold typography text', 'graphic text design', 'statement lettering'],
    'tech_details': ['technical construction details', 'visible seams and zippers', 'modern technical features'],

    # Creative/Artistic anchors
    'geometry': ['geometric pattern design', 'abstract shapes', 'angular graphic elements'],
    'graphics': ['graphic print design', 'artistic illustration', 'visual artwork print'],
    'asymmetry': ['asymmetrical design', 'off-center placement', 'unconventional layout']
}

# ============================================
# LOAD CLIP MODEL
# ============================================

print("Loading FashionCLIP model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use fashion-clip for better fashion understanding
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

model.eval()
print(f"Model loaded on {device}")

# ============================================
# COMPUTE TEXT EMBEDDINGS FOR TAXONOMY
# ============================================

print("\nComputing taxonomy text embeddings...")

def get_text_embedding(texts):
    """Get averaged CLIP text embedding for a list of prompts."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Average if multiple prompts
    avg_embedding = text_features.mean(dim=0)
    avg_embedding = avg_embedding / avg_embedding.norm()

    return avg_embedding.cpu().numpy()

# Compute archetype embeddings
archetype_embeddings = {}
for arch_name, arch_data in ARCHETYPES.items():
    archetype_embeddings[arch_name] = get_text_embedding(arch_data['prompts'])
    print(f"  Archetype '{arch_name}': {archetype_embeddings[arch_name].shape}")

# Compute visual anchor embeddings
anchor_embeddings = {}
for anchor_name, prompts in VISUAL_ANCHORS.items():
    anchor_embeddings[anchor_name] = get_text_embedding(prompts)
    print(f"  Anchor '{anchor_name}': {anchor_embeddings[anchor_name].shape}")

# ============================================
# LOAD ITEM EMBEDDINGS
# ============================================

print("\nLoading item embeddings...")
embeddings_path = "/home/ubuntu/recSys/outfitTransformer/models/hp_embeddings.pkl"
with open(embeddings_path, 'rb') as f:
    embeddings_data = pickle.load(f)

item_ids = list(embeddings_data.keys())
print(f"Loaded {len(item_ids)} items")

# ============================================
# COMPUTE TAXONOMY SCORES FOR EACH ITEM
# ============================================

print("\nComputing taxonomy scores for each item...")

taxonomy_scores = {}

for item_id in tqdm(item_ids):
    item_emb = embeddings_data[item_id]['embedding']
    item_emb = item_emb / np.linalg.norm(item_emb)  # Normalize

    # Compute archetype scores (cosine similarity)
    arch_scores = {}
    for arch_name, arch_emb in archetype_embeddings.items():
        similarity = float(np.dot(item_emb, arch_emb))
        # Convert to 0-1 scale (similarity is typically -1 to 1)
        arch_scores[arch_name] = (similarity + 1) / 2

    # Compute visual anchor scores
    anchor_scores = {}
    for anchor_name, anchor_emb in anchor_embeddings.items():
        similarity = float(np.dot(item_emb, anchor_emb))
        anchor_scores[anchor_name] = (similarity + 1) / 2

    # Determine dominant archetype
    dominant_archetype = max(arch_scores, key=arch_scores.get)

    # Determine dominant anchors (top 3)
    sorted_anchors = sorted(anchor_scores.items(), key=lambda x: x[1], reverse=True)
    dominant_anchors = [a[0] for a in sorted_anchors[:3]]

    taxonomy_scores[item_id] = {
        'archetype_scores': arch_scores,
        'anchor_scores': anchor_scores,
        'dominant_archetype': dominant_archetype,
        'dominant_anchors': dominant_anchors
    }

# ============================================
# ANALYZE DISTRIBUTION
# ============================================

print("\n=== Taxonomy Distribution Analysis ===")

# Archetype distribution
arch_counts = {a: 0 for a in ARCHETYPES.keys()}
for item_id, scores in taxonomy_scores.items():
    arch_counts[scores['dominant_archetype']] += 1

print("\nDominant Archetype Distribution:")
for arch, count in sorted(arch_counts.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(item_ids) * 100
    print(f"  {arch}: {count} ({pct:.1f}%)")

# Anchor distribution
anchor_counts = {a: 0 for a in VISUAL_ANCHORS.keys()}
for item_id, scores in taxonomy_scores.items():
    for anchor in scores['dominant_anchors']:
        anchor_counts[anchor] += 1

print("\nTop Visual Anchors (in top-3 for items):")
for anchor, count in sorted(anchor_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {anchor}: {count}")

# Show some examples
print("\n=== Example Items ===")
for arch_name in ARCHETYPES.keys():
    # Find item with highest score for this archetype
    best_item = max(
        taxonomy_scores.items(),
        key=lambda x: x[1]['archetype_scores'][arch_name]
    )
    item_id, scores = best_item
    print(f"\n{arch_name.upper()} example: {item_id}")
    print(f"  Archetype scores: {scores['archetype_scores']}")
    print(f"  Top anchors: {scores['dominant_anchors']}")

# ============================================
# SAVE TAXONOMY SCORES
# ============================================

output = {
    'item_scores': taxonomy_scores,
    'archetype_names': list(ARCHETYPES.keys()),
    'anchor_names': list(VISUAL_ANCHORS.keys()),
    'archetype_descriptions': {k: v['description'] for k, v in ARCHETYPES.items()},
    'archetype_embeddings': archetype_embeddings,
    'anchor_embeddings': anchor_embeddings
}

output_path = "/home/ubuntu/recSys/outfitTransformer/models/hp_taxonomy_scores.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(output, f)

print(f"\n\nSaved taxonomy scores to: {output_path}")
print(f"Each item now has:")
print(f"  - 4 archetype scores (classic, natural_sporty, dramatic_street, creative_artistic)")
print(f"  - 12 visual anchor scores")
print(f"  - Dominant archetype & anchors")
