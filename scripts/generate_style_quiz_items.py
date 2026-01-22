#!/usr/bin/env python3
"""
Generate Style Quiz Items for User Onboarding

Uses FashionCLIP text-to-image search to find style-representative items.
Generates HTML preview for manual curation, then saves curated items.

9 Style Categories:
- Casual, Formal, Streetwear, Minimalist, Bohemian
- Athleisure, Vintage/Retro, Preppy, Romantic
"""

import os
import sys
import pickle
import argparse
import random
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Style definitions with FashionCLIP prompts
STYLE_CONFIG = {
    'Casual': {
        'description': 'Relaxed, everyday clothing for comfort',
        'prompts': {
            'women': [
                "casual relaxed everyday women's outfit",
                "comfortable women's casual wear t-shirt jeans",
                "laid back casual women's style"
            ],
            'men': [
                "casual relaxed men's outfit",
                "comfortable men's casual wear t-shirt jeans",
                "laid back casual men's style"
            ]
        },
        'color': '#6CB4EE'
    },
    'Formal': {
        'description': 'Professional and elegant attire',
        'prompts': {
            'women': [
                "formal elegant women's business attire blazer",
                "professional women's office wear suit",
                "elegant formal women's dress classy"
            ],
            'men': [
                "formal professional men's business suit",
                "elegant men's dress shirt tie",
                "professional men's office attire"
            ]
        },
        'color': '#2C3E50'
    },
    'Streetwear': {
        'description': 'Urban, trendy street fashion',
        'prompts': {
            'women': [
                "streetwear urban edgy women's fashion hoodie",
                "trendy street style women's oversized",
                "urban fashion women's sneakers casual"
            ],
            'men': [
                "streetwear urban men's fashion hoodie sneakers",
                "trendy street style men's oversized",
                "urban fashion men's graphic tee"
            ]
        },
        'color': '#E74C3C'
    },
    'Minimalist': {
        'description': 'Clean lines, neutral colors, simple',
        'prompts': {
            'women': [
                "minimalist clean simple women's style neutral",
                "women's minimalist fashion basic white black",
                "simple elegant women's monochrome outfit"
            ],
            'men': [
                "minimalist clean men's style neutral colors",
                "men's minimalist fashion basic monochrome",
                "simple clean men's wardrobe essentials"
            ]
        },
        'color': '#95A5A6'
    },
    'Bohemian': {
        'description': 'Free-spirited, artistic, natural',
        'prompts': {
            'women': [
                "bohemian boho free-spirited women's dress",
                "boho chic women's flowy maxi skirt",
                "artistic bohemian women's natural fabric"
            ],
            'men': [
                "bohemian boho men's style linen natural",
                "relaxed boho men's earthy tones",
                "free-spirited artistic men's casual"
            ]
        },
        'color': '#D68910'
    },
    'Athleisure': {
        'description': 'Sporty, comfortable, active',
        'prompts': {
            'women': [
                "athleisure sporty women's yoga leggings",
                "athletic women's workout wear gym",
                "sporty casual women's activewear sneakers"
            ],
            'men': [
                "athleisure sporty men's workout joggers",
                "athletic men's gym wear performance",
                "sporty casual men's activewear sneakers"
            ]
        },
        'color': '#27AE60'
    },
    'Vintage': {
        'description': '70s/80s/90s inspired retro looks',
        'prompts': {
            'women': [
                "vintage retro 70s 80s women's fashion",
                "retro vintage women's classic style",
                "70s inspired women's fashion boho"
            ],
            'men': [
                "vintage retro men's classic style",
                "retro 70s 80s men's fashion",
                "classic vintage men's timeless"
            ]
        },
        'color': '#8E44AD'
    },
    'Preppy': {
        'description': 'Classic, polished, collegiate style',
        'prompts': {
            'women': [
                "preppy classic polished women's collegiate",
                "preppy women's style blazer skirt",
                "classic collegiate women's fashion polo"
            ],
            'men': [
                "preppy classic men's polo collegiate",
                "preppy men's style chinos blazer",
                "classic collegiate men's fashion boat shoes"
            ]
        },
        'color': '#2980B9'
    },
    'Romantic': {
        'description': 'Soft, feminine, floral patterns',
        'prompts': {
            'women': [
                "romantic soft feminine women's floral lace",
                "delicate romantic women's dress pastel",
                "feminine romantic women's blouse ruffles"
            ],
            'men': [
                "romantic soft men's elegant pastel",
                "refined elegant men's style soft colors",
                "gentle sophisticated men's fashion"
            ]
        },
        'color': '#FFB6C1'
    }
}


def load_fashion_clip():
    """Load FashionCLIP model"""
    from fashion_clip.fashion_clip import FashionCLIP
    print("Loading FashionCLIP model...")
    fclip = FashionCLIP('fashion-clip')
    print("FashionCLIP loaded")
    return fclip


def load_data():
    """Load embeddings and gender mapping"""
    data_dir = Path(__file__).parent.parent / "data" / "polyvore_u"
    models_dir = Path(__file__).parent.parent / "models"

    # Load CLIP embeddings
    embeddings_path = data_dir / "polyvore_u_clip_embeddings.npy"
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")

    # Load image paths
    image_paths_path = data_dir / "all_item_image_paths.npy"
    image_paths = np.load(image_paths_path, allow_pickle=True)
    print(f"  Image paths: {len(image_paths)}")

    # Load gender mapping
    gender_path = data_dir / "gender_mapping.pkl"
    with open(gender_path, 'rb') as f:
        gender_data = pickle.load(f)

    women_ids = set(gender_data.get('women_ids', []))
    men_ids = set(gender_data.get('men_ids', []))
    unisex_ids = set(gender_data.get('unisex_ids', []))

    print(f"  Gender: {len(women_ids)} women, {len(men_ids)} men, {len(unisex_ids)} unisex")

    return embeddings, image_paths, women_ids, men_ids, unisex_ids


def text_to_embedding(fclip, text):
    """Convert text to CLIP embedding"""
    with torch.no_grad():
        text_embedding = fclip.encode_text([text], batch_size=1)
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
    return text_embedding[0]


def find_style_items(fclip, embeddings, prompts, valid_ids, top_k=30):
    """Find items matching style prompts using FashionCLIP"""
    all_scores = np.zeros(len(embeddings))

    for prompt in prompts:
        text_emb = text_to_embedding(fclip, prompt)
        # Cosine similarity (embeddings already normalized)
        scores = embeddings @ text_emb
        all_scores += scores

    # Average across prompts
    all_scores /= len(prompts)

    # Filter to valid gender IDs
    mask = np.zeros(len(embeddings), dtype=bool)
    for idx in valid_ids:
        if idx < len(mask):
            mask[idx] = True

    all_scores[~mask] = -np.inf

    # Get top-k
    top_indices = np.argsort(all_scores)[::-1][:top_k]
    top_scores = all_scores[top_indices]

    return [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores) if score > -np.inf]


def generate_style_quiz_items(output_dir=None, items_per_style=30):
    """Generate style items for all 9 styles, both genders"""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "generated_outfits" / "style_quiz"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models and data
    fclip = load_fashion_clip()
    embeddings, image_paths, women_ids, men_ids, unisex_ids = load_data()

    # Normalize embeddings if needed
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    # Valid IDs per gender (include unisex for both)
    valid_women = women_ids | unisex_ids
    valid_men = men_ids | unisex_ids

    print(f"\nValid items: {len(valid_women)} women, {len(valid_men)} men")

    # Generate items for each style
    style_items = {}

    for style_name, config in STYLE_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Processing style: {style_name}")
        print(f"Description: {config['description']}")

        style_items[style_name] = {'women': [], 'men': []}

        for gender in ['women', 'men']:
            valid_ids = valid_women if gender == 'women' else valid_men
            prompts = config['prompts'][gender]

            print(f"\n  {gender.capitalize()}:")
            print(f"    Prompts: {prompts}")

            items = find_style_items(fclip, embeddings, prompts, valid_ids, top_k=items_per_style)
            style_items[style_name][gender] = items

            print(f"    Found {len(items)} items")
            if items:
                print(f"    Top scores: {[f'{s:.3f}' for _, s in items[:5]]}")

    # Generate HTML preview
    generate_html_preview(style_items, image_paths, output_dir)

    # Save raw results
    results_path = output_dir / "style_quiz_candidates.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(style_items, f)
    print(f"\nSaved candidates to {results_path}")

    return style_items


def generate_html_preview(style_items, image_paths, output_dir):
    """Generate HTML file for visual review"""

    html_parts = ['''<!DOCTYPE html>
<html>
<head>
    <title>Style Quiz - Visual Review (All 9 Styles)</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            padding: 30px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            min-height: 100vh;
        }
        .container { max-width: 1800px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #888; margin-bottom: 10px; }
        .instructions {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .instructions h3 { margin-top: 0; color: #00d9ff; }
        .instructions ol { margin: 0; padding-left: 20px; }
        .instructions li { margin: 8px 0; }

        .style-section {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid;
        }
        .style-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .style-name { font-size: 1.8em; }
        .style-desc { color: #888; font-style: italic; }

        .gender-row {
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
        }
        .gender-column { flex: 1; }
        .gender-label {
            font-size: 1.2em;
            margin-bottom: 15px;
            padding: 8px 20px;
            border-radius: 5px;
            display: inline-block;
            font-weight: bold;
        }
        .women-label { background: linear-gradient(135deg, #ff69b4, #ff1493); }
        .men-label { background: linear-gradient(135deg, #4169e1, #1e90ff); }

        .items-grid {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .item {
            background: white;
            padding: 8px;
            border-radius: 10px;
            text-align: center;
            width: 120px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 3px solid transparent;
        }
        .item:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        .item.selected {
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0,255,136,0.5);
        }
        .item img {
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: 8px;
        }
        .item-id { color: #333; font-size: 0.75em; margin-top: 5px; }
        .item-score { color: #666; font-size: 0.65em; }

        .selection-count {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            color: #1a1a2e;
            padding: 15px 25px;
            border-radius: 30px;
            font-weight: bold;
            box-shadow: 0 5px 20px rgba(0,217,255,0.4);
        }

        .export-btn {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: #00ff88;
            color: #1a1a2e;
            padding: 15px 25px;
            border-radius: 30px;
            border: none;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 5px 20px rgba(0,255,136,0.4);
        }
        .export-btn:hover { background: #00e078; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Style Quiz - Item Selection Preview</h1>
        <p class="subtitle">All 9 Styles - Click items to select, then export curated list</p>

        <div class="instructions">
            <h3>Manual Curation Instructions</h3>
            <ol>
                <li>Review items for each style category</li>
                <li>Click items that best represent the style (aim for 15-20 per style per gender)</li>
                <li>Look for variety: different item types, colors within the style</li>
                <li>Click "Export Selected" to get the curated item IDs</li>
            </ol>
        </div>
''']

    # Copy images directory reference
    images_dir = Path(__file__).parent.parent / "data" / "polyvore_u" / "291x291"

    for style_name, config in STYLE_CONFIG.items():
        items = style_items[style_name]
        color = config['color']

        html_parts.append(f'''
        <div class="style-section" style="border-left-color: {color};">
            <div class="style-header">
                <span class="style-name" style="color: {color};">{style_name}</span>
                <span class="style-desc">{config['description']}</span>
            </div>
            <div class="gender-row">
''')

        for gender in ['women', 'men']:
            label_class = f"{gender}-label"
            gender_items = items[gender]

            html_parts.append(f'''
                <div class="gender-column">
                    <span class="gender-label {label_class}">{gender.capitalize()}</span>
                    <div class="items-grid">
''')

            for item_id, score in gender_items:
                if item_id < len(image_paths):
                    img_path = image_paths[item_id]
                    # Create relative path to images
                    img_src = f"../../data/polyvore_u/291x291/{img_path}"

                    html_parts.append(f'''
                        <div class="item" data-style="{style_name}" data-gender="{gender}" data-id="{item_id}" onclick="toggleSelect(this)">
                            <img src="{img_src}" alt="Item {item_id}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect fill=%22%23ddd%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2250%22 text-anchor=%22middle%22 fill=%22%23999%22>No img</text></svg>'">
                            <div class="item-id">#{item_id}</div>
                            <div class="item-score">{score:.3f}</div>
                        </div>
''')

            html_parts.append('''
                    </div>
                </div>
''')

        html_parts.append('''
            </div>
        </div>
''')

    # Add JavaScript for selection
    html_parts.append('''
    </div>

    <div class="selection-count" id="selectionCount">Selected: 0</div>
    <button class="export-btn" onclick="exportSelected()">Export Selected</button>

    <script>
        let selected = {};

        function toggleSelect(el) {
            const style = el.dataset.style;
            const gender = el.dataset.gender;
            const id = parseInt(el.dataset.id);
            const key = `${style}_${gender}`;

            if (!selected[key]) selected[key] = new Set();

            if (el.classList.contains('selected')) {
                el.classList.remove('selected');
                selected[key].delete(id);
            } else {
                el.classList.add('selected');
                selected[key].add(id);
            }

            updateCount();
        }

        function updateCount() {
            let total = 0;
            for (const key in selected) {
                total += selected[key].size;
            }
            document.getElementById('selectionCount').textContent = `Selected: ${total}`;
        }

        function exportSelected() {
            const result = {};
            for (const key in selected) {
                if (selected[key].size > 0) {
                    const [style, gender] = key.split('_');
                    if (!result[style]) result[style] = {};
                    result[style][gender] = Array.from(selected[key]);
                }
            }

            // Summary
            let summary = "Style Quiz Curated Items\\n" + "=".repeat(50) + "\\n\\n";
            for (const style in result) {
                summary += `${style}:\\n`;
                for (const gender in result[style]) {
                    summary += `  ${gender}: ${result[style][gender].length} items\\n`;
                    summary += `    IDs: [${result[style][gender].join(', ')}]\\n`;
                }
                summary += "\\n";
            }

            // Python dict format
            summary += "\\n\\nPython dict format:\\n";
            summary += JSON.stringify(result, null, 2);

            // Create download
            const blob = new Blob([summary], {type: 'text/plain'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'style_quiz_curated.txt';
            a.click();

            // Also show in console
            console.log("Curated items:", result);
            alert("Exported! Check your downloads folder for style_quiz_curated.txt");
        }
    </script>
</body>
</html>
''')

    # Write HTML file
    html_path = output_dir / "style_preview_all_9_styles.html"
    with open(html_path, 'w') as f:
        f.write(''.join(html_parts))

    print(f"\nGenerated HTML preview: {html_path}")

    # Also create symlink for images if needed
    images_link = output_dir / "images"
    if not images_link.exists():
        source_images = Path(__file__).parent.parent / "data" / "polyvore_u" / "291x291"
        try:
            images_link.symlink_to(source_images)
            print(f"Created symlink: {images_link} -> {source_images}")
        except Exception as e:
            print(f"Note: Could not create symlink: {e}")


def save_curated_items(curated_dict, output_path=None):
    """Save manually curated items to pickle file"""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "polyvore_u" / "style_quiz_items.pkl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(curated_dict, f)

    print(f"Saved curated items to {output_path}")

    # Print summary
    print("\nCurated items summary:")
    for style, genders in curated_dict.items():
        print(f"  {style}:")
        for gender, items in genders.items():
            print(f"    {gender}: {len(items)} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate style quiz items")
    parser.add_argument("--items-per-style", type=int, default=30,
                        help="Number of candidate items per style per gender")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for HTML preview")

    args = parser.parse_args()

    print("="*70)
    print("STYLE QUIZ ITEM GENERATION")
    print("="*70)
    print(f"\nStyles: {list(STYLE_CONFIG.keys())}")
    print(f"Items per style: {args.items_per_style}")

    style_items = generate_style_quiz_items(
        output_dir=args.output_dir,
        items_per_style=args.items_per_style
    )

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Open the HTML preview in a browser")
    print("2. Select the best 15-20 items per style per gender")
    print("3. Click 'Export Selected' to save your curation")
    print("4. Use save_curated_items() with the exported data")
