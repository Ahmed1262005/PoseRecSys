"""
Hierarchical Clustering for HP Dataset

Creates 2-level clustering:
- Level 1: Broad style categories (6 macro-clusters)
- Level 2: Finer subclusters within each macro (2-4 per macro)

Brand is tracked as an attribute, NOT a cluster.
"""

import pickle
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter
import pandas as pd
from pathlib import Path

# Load embeddings
embeddings_path = "/home/ubuntu/recSys/outfitTransformer/models/hp_embeddings.pkl"
with open(embeddings_path, 'rb') as f:
    embeddings_data = pickle.load(f)

item_ids = list(embeddings_data.keys())
embeddings_matrix = np.array([embeddings_data[k]['embedding'] for k in item_ids])

print(f"Loaded {len(item_ids)} items with {embeddings_matrix.shape[1]}-dim embeddings")

# Load metadata for analysis
csv_dir = Path("/home/ubuntu/recSys/outfitTransformer/HPdataset")
metadata = {}

# Plain T-shirts
plain = pd.read_csv(csv_dir / "Formatted - Plain T-shirts.csv")
for _, row in plain.iterrows():
    item_id = f"Plain T-shirts/{int(row['Item List'])}"
    metadata[item_id] = {
        'color': str(row['Color ']).strip().lower(),
        'fit': str(row['Fit']).strip(),
        'brand': str(row['Brand ']).strip(),
        'fabric': str(row['Fabric']).strip(),
        'category': 'Plain T-shirts'
    }

# Graphics T-shirts
graphics = pd.read_csv(csv_dir / "Formatted - Graphics T-shirts.csv")
for _, row in graphics.iterrows():
    item_id = f"Graphics T-shirts/{int(row['Image title'])}"
    metadata[item_id] = {
        'color': str(row['Color ']).strip().lower(),
        'fit': str(row['Fit']).strip(),
        'brand': str(row['Brand ']).strip(),
        'fabric': str(row['Fabric']).strip(),
        'category': 'Graphics T-shirts'
    }

# Small logos
small = pd.read_csv(csv_dir / "Formatted - Small logos.csv")
for _, row in small.iterrows():
    if pd.isna(row['Item List']):
        continue
    item_id = f"Small logos/{int(row['Item List'])}"
    metadata[item_id] = {
        'color': str(row.get('Color ', '')).strip().lower() if pd.notna(row.get('Color ')) else '',
        'fit': str(row.get('Fit', '')).strip() if pd.notna(row.get('Fit')) else '',
        'brand': str(row.get('Brand ', '')).strip() if pd.notna(row.get('Brand ')) else '',
        'fabric': str(row.get('Fabric', '')).strip() if pd.notna(row.get('Fabric')) else '',
        'category': 'Small logos'
    }

print(f"Loaded metadata for {len(metadata)} items")

# ============================================
# LEVEL 1: Macro-clusters (broad style categories)
# ============================================
print("\n=== Level 1: Macro-Clustering ===")

# Test K=5,6,7,8 for macro level
best_k = 6
best_silhouette = -1

for k in [5, 6, 7, 8]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_matrix)
    score = silhouette_score(embeddings_matrix, labels)
    print(f"K={k}: silhouette={score:.4f}")

    if score > best_silhouette:
        best_silhouette = score
        best_k = k

print(f"\nBest macro K={best_k} (silhouette={best_silhouette:.4f})")

# Final macro clustering
kmeans_macro = KMeans(n_clusters=best_k, random_state=42, n_init=10)
macro_labels = kmeans_macro.fit_predict(embeddings_matrix)

# Analyze macro clusters
print(f"\n--- Macro Cluster Analysis ---")
macro_clusters = {}
for i, (item_id, label) in enumerate(zip(item_ids, macro_labels)):
    if label not in macro_clusters:
        macro_clusters[label] = []
    macro_clusters[label].append(item_id)

for cluster_id, items in sorted(macro_clusters.items()):
    # Get category distribution
    categories = [metadata.get(i, {}).get('category', 'Unknown') for i in items]
    category_counts = Counter(categories)

    # Get top brands
    brands = [metadata.get(i, {}).get('brand', 'Unknown') for i in items]
    brand_counts = Counter(brands).most_common(3)

    # Get color distribution
    colors = [metadata.get(i, {}).get('color', 'Unknown') for i in items]
    color_counts = Counter(colors).most_common(3)

    print(f"\nMacro Cluster {cluster_id} ({len(items)} items):")
    print(f"  Categories: {dict(category_counts)}")
    print(f"  Top brands: {brand_counts}")
    print(f"  Top colors: {color_counts}")

# ============================================
# LEVEL 2: Sub-clusters within each macro
# ============================================
print("\n\n=== Level 2: Sub-Clustering ===")

sub_clusters = {}
sub_labels = np.zeros(len(item_ids), dtype=int)
sub_cluster_id = 0

# Map: macro_cluster -> list of sub_cluster_ids
macro_to_sub = {}

for macro_id in range(best_k):
    macro_mask = macro_labels == macro_id
    macro_items = [item_ids[i] for i, m in enumerate(macro_mask) if m]
    macro_embeddings = embeddings_matrix[macro_mask]

    # Determine optimal sub-clusters (2-4) based on size
    n_items = len(macro_items)

    if n_items < 20:
        # Too small to sub-cluster
        sub_k = 1
    elif n_items < 50:
        sub_k = 2
    else:
        # Test 2, 3, 4 subclusters
        best_sub_k = 2
        best_sub_sil = -1

        for k in [2, 3, 4]:
            if k >= n_items:
                continue
            kmeans_sub = KMeans(n_clusters=k, random_state=42, n_init=10)
            sub_lab = kmeans_sub.fit_predict(macro_embeddings)
            if len(set(sub_lab)) > 1:
                score = silhouette_score(macro_embeddings, sub_lab)
                if score > best_sub_sil:
                    best_sub_sil = score
                    best_sub_k = k

        sub_k = best_sub_k

    print(f"\nMacro {macro_id} ({n_items} items) → {sub_k} sub-clusters")

    # Create sub-clusters
    if sub_k == 1:
        sub_ids_for_macro = [sub_cluster_id]
        for i, idx in enumerate(np.where(macro_mask)[0]):
            sub_labels[idx] = sub_cluster_id
            if sub_cluster_id not in sub_clusters:
                sub_clusters[sub_cluster_id] = {'macro': macro_id, 'items': []}
            sub_clusters[sub_cluster_id]['items'].append(item_ids[idx])
        sub_cluster_id += 1
    else:
        kmeans_sub = KMeans(n_clusters=sub_k, random_state=42, n_init=10)
        local_sub_labels = kmeans_sub.fit_predict(macro_embeddings)

        sub_ids_for_macro = []
        for local_sub in range(sub_k):
            sub_ids_for_macro.append(sub_cluster_id)

            for i, (idx, local_label) in enumerate(zip(np.where(macro_mask)[0], local_sub_labels)):
                if local_label == local_sub:
                    sub_labels[idx] = sub_cluster_id
                    if sub_cluster_id not in sub_clusters:
                        sub_clusters[sub_cluster_id] = {'macro': macro_id, 'items': []}
                    sub_clusters[sub_cluster_id]['items'].append(item_ids[idx])

            # Analyze sub-cluster
            sub_items = sub_clusters[sub_cluster_id]['items']
            brands = [metadata.get(i, {}).get('brand', '') for i in sub_items]
            colors = [metadata.get(i, {}).get('color', '') for i in sub_items]
            fits = [metadata.get(i, {}).get('fit', '') for i in sub_items]

            print(f"  Sub {sub_cluster_id}: {len(sub_items)} items | brands={Counter(brands).most_common(2)} | colors={Counter(colors).most_common(2)}")

            sub_cluster_id += 1

    macro_to_sub[macro_id] = sub_ids_for_macro

# ============================================
# Name the clusters (BRAND-CENTRIC)
# ============================================
print("\n\n=== Naming Clusters (Brand-Centric) ===")

# Define brand tiers for naming
LUXURY_BRANDS = {'GUCCI', 'Tom Ford', 'Dior', 'Zegna', 'Off-White', 'Palm Angels'}
PREMIUM_BRANDS = {'Hugo Boss', 'Calvin Klein', 'Ralph Lauren', 'Tommy Hilfiger', 'Ted Baker'}
CONTEMPORARY_BRANDS = {'Zara', 'Mango', 'Banana Republic', 'Suitsupply', 'Perry Ellis'}
FAST_FASHION_BRANDS = {'Uniqlo', 'Pacsun', 'Pull & Bear', 'H&M'}
STREETWEAR_BRANDS = {'TRUE RELIGION', 'Off-White', 'Palm Angels', 'Cotopaxi'}

def get_brand_tier(brands):
    brand_counts = Counter(brands)
    for brand, _ in brand_counts.most_common():
        if brand in LUXURY_BRANDS:
            return "Luxury"
        elif brand in STREETWEAR_BRANDS:
            return "Streetwear"
        elif brand in PREMIUM_BRANDS:
            return "Premium"
        elif brand in CONTEMPORARY_BRANDS:
            return "Contemporary"
        elif brand in FAST_FASHION_BRANDS:
            return "Essential"
    return "Mixed"

# Macro names (based on dominant brand + category)
macro_names = {}
for macro_id in range(best_k):
    macro_item_ids = [item_ids[i] for i, l in enumerate(macro_labels) if l == macro_id]

    # Get dominant attributes
    brands = [metadata.get(i, {}).get('brand', '') for i in macro_item_ids if metadata.get(i, {}).get('brand')]
    categories = [metadata.get(i, {}).get('category', '') for i in macro_item_ids]

    top_brands = Counter(brands).most_common(3)
    top_category = Counter(categories).most_common(1)[0][0] if categories else ""
    tier = get_brand_tier(brands)

    # Use dominant brand in name
    dominant_brand = top_brands[0][0] if top_brands else ""

    # Create descriptive name
    if "Graphics" in top_category:
        style = "Graphics"
    elif "Small logos" in top_category:
        style = "Logos"
    else:
        style = "Basics"

    name = f"{tier} {style}"
    if dominant_brand:
        name = f"{dominant_brand} & {style}"

    macro_names[macro_id] = name
    print(f"Macro {macro_id}: {name}")
    print(f"  Brands: {top_brands}")

# Sub-cluster names (brand + distinguishing attribute)
sub_names = {}
for sub_id, sub_data in sub_clusters.items():
    macro_id = sub_data['macro']
    items = sub_data['items']

    brands = [metadata.get(i, {}).get('brand', '') for i in items if metadata.get(i, {}).get('brand')]
    colors = [metadata.get(i, {}).get('color', '') for i in items if metadata.get(i, {}).get('color')]
    fits = [metadata.get(i, {}).get('fit', '') for i in items if metadata.get(i, {}).get('fit')]
    categories = [metadata.get(i, {}).get('category', '') for i in items if metadata.get(i, {}).get('category')]

    top_brand = Counter(brands).most_common(1)[0][0] if brands else "Mixed"
    top_color = Counter(colors).most_common(1)[0][0] if colors else ""
    top_fit = Counter(fits).most_common(1)[0][0] if fits else ""
    top_cat = Counter(categories).most_common(1)[0][0] if categories else ""

    # Short category
    cat_short = "Plain" if "Plain" in top_cat else ("Logo" if "Small" in top_cat else "Graphic")

    # Build name: Brand + Category style
    sub_names[sub_id] = f"{top_brand} {cat_short}"

    print(f"  Sub {sub_id}: {sub_names[sub_id]} ({len(items)} items)")

# ============================================
# Save hierarchical clusters
# ============================================
output = {
    'item_ids': item_ids,
    'macro_labels': macro_labels.tolist(),
    'sub_labels': sub_labels.tolist(),
    'macro_names': macro_names,
    'sub_names': sub_names,
    'macro_to_sub': macro_to_sub,
    'n_macro_clusters': best_k,
    'n_sub_clusters': len(sub_clusters),
    'sub_cluster_info': {
        sub_id: {
            'macro': data['macro'],
            'n_items': len(data['items']),
            'name': sub_names[sub_id]
        }
        for sub_id, data in sub_clusters.items()
    }
}

output_path = "/home/ubuntu/recSys/outfitTransformer/models/hp_hierarchical_clusters.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(output, f)

print(f"\n\n=== Summary ===")
print(f"Saved to: {output_path}")
print(f"Macro clusters: {best_k}")
print(f"Sub clusters: {len(sub_clusters)}")
print(f"\nHierarchy:")
for macro_id, macro_name in macro_names.items():
    sub_ids = macro_to_sub[macro_id]
    print(f"  {macro_name}:")
    for sub_id in sub_ids:
        print(f"    - {sub_names[sub_id]} ({output['sub_cluster_info'][sub_id]['n_items']} items)")
