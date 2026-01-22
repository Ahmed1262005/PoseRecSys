"""
Image Cleanup Script for Women's Fashion Dataset

Uses FashionCLIP to detect and flag problematic images:
1. Images with text overlays (brand names, website UI, promotional text)
2. Zoomed-in/detail shots (fabric closeups, pattern details)
3. Back-facing images (showing back of clothing/model)

Output: JSON file with images to exclude and their reasons
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import torch
from fashion_clip.fashion_clip import FashionCLIP

# Configuration
IMAGES_DIR = "/home/ubuntu/recSys/outfitTransformer/data/women_fashion/images_webp"
OUTPUT_DIR = "/home/ubuntu/recSys/outfitTransformer/data/women_fashion/processed"
BATCH_SIZE = 32

# Detection thresholds (higher = more strict filtering)
TEXT_THRESHOLD = 0.25       # Images with text/UI elements
ZOOMED_THRESHOLD = 0.26     # Zoomed in / detail / cropped shots
BACK_VIEW_THRESHOLD = 0.26  # Back-facing images (model facing away)


@dataclass
class ImageClassification:
    """Classification result for an image."""
    path: str
    is_good: bool
    issues: List[str]
    scores: Dict[str, float]


class FashionImageCleaner:
    """
    Uses FashionCLIP zero-shot classification to detect problematic images.
    """

    # Prompts for detecting GOOD images (what we want to keep)
    GOOD_PROMPTS = [
        "a woman modeling clothing from the front",
        "front view of a woman wearing fashion",
        "model wearing clothes facing camera",
        "clean product photo of clothing",
        "woman in outfit facing forward",
        "fashion model front facing pose",
    ]

    # Prompts for detecting TEXT/UI overlays
    TEXT_PROMPTS = [
        "image with text overlay",
        "website screenshot with navigation menu",
        "brand logo and text on image",
        "promotional banner with text",
        "image with watermark text",
        "screenshot of online store",
        "image with website header",
        "product photo with feature callouts and annotations",
        "image with text labels and arrows pointing to features",
        "infographic style product image with text descriptions",
    ]

    # Prompts for detecting ZOOMED/DETAIL/CROPPED shots
    ZOOMED_PROMPTS = [
        "close up of fabric texture",
        "zoomed in detail of clothing",
        "macro shot of material pattern",
        "extreme closeup of garment detail",
        "cropped photo without head or face",
        "headless fashion photo showing only body",
        "photo cropped at neck without face",
    ]

    # Prompts for detecting BACK VIEW images (model facing away, not flat lays)
    BACK_PROMPTS = [
        "woman showing her back to camera",
        "model facing away from camera showing back",
        "woman photographed from behind showing back",
        "person with back turned to camera",
        "rear view of woman in clothing",
        "model looking away showing back of outfit",
    ]

    def __init__(
        self,
        device: str = "cuda",
        text_threshold: float = TEXT_THRESHOLD,
        zoom_threshold: float = ZOOMED_THRESHOLD,
        back_threshold: float = BACK_VIEW_THRESHOLD
    ):
        """Initialize FashionCLIP model."""
        print("Loading FashionCLIP model...")
        self.device = device
        self.text_threshold = text_threshold
        self.zoom_threshold = zoom_threshold
        self.back_threshold = back_threshold
        self.fclip = FashionCLIP('fashion-clip')

        # Pre-encode all text prompts
        print("Encoding text prompts...")
        self.good_embeddings = self._encode_texts(self.GOOD_PROMPTS)
        self.text_embeddings = self._encode_texts(self.TEXT_PROMPTS)
        self.zoomed_embeddings = self._encode_texts(self.ZOOMED_PROMPTS)
        self.back_embeddings = self._encode_texts(self.BACK_PROMPTS)

        print("FashionImageCleaner ready!")

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text prompts to embeddings."""
        embeddings = self.fclip.encode_text(texts, batch_size=len(texts))
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """Encode a single image to embedding."""
        try:
            image = Image.open(image_path).convert('RGB')
            embedding = self.fclip.encode_images([image], batch_size=1)
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding[0]
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return None

    def _encode_images_batch(self, image_paths: List[str]) -> List[Tuple[str, Optional[np.ndarray]]]:
        """Encode multiple images in a batch for efficiency."""
        results = []
        valid_images = []
        valid_paths = []

        # Load all images
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                valid_images.append(image)
                valid_paths.append(path)
            except Exception as e:
                results.append((path, None))

        if not valid_images:
            return results

        # Batch encode
        try:
            embeddings = self.fclip.encode_images(valid_images, batch_size=len(valid_images))
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            for path, emb in zip(valid_paths, embeddings):
                results.append((path, emb))
        except Exception as e:
            print(f"Batch encoding error: {e}")
            # Fall back to individual encoding
            for path, img in zip(valid_paths, valid_images):
                try:
                    emb = self.fclip.encode_images([img], batch_size=1)
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                    results.append((path, emb[0]))
                except:
                    results.append((path, None))

        return results

    def _compute_category_score(
        self,
        image_embedding: np.ndarray,
        category_embeddings: np.ndarray
    ) -> float:
        """Compute max similarity to a category of prompts."""
        similarities = image_embedding @ category_embeddings.T
        return float(np.max(similarities))

    def classify_image(self, image_path: str) -> ImageClassification:
        """
        Classify an image and determine if it should be kept or excluded.

        Returns classification with scores for each category.
        """
        embedding = self._encode_image(image_path)

        if embedding is None:
            return ImageClassification(
                path=image_path,
                is_good=False,
                issues=["failed_to_load"],
                scores={}
            )

        # Compute scores for each category
        good_score = self._compute_category_score(embedding, self.good_embeddings)
        text_score = self._compute_category_score(embedding, self.text_embeddings)
        zoomed_score = self._compute_category_score(embedding, self.zoomed_embeddings)
        back_score = self._compute_category_score(embedding, self.back_embeddings)

        scores = {
            "good": good_score,
            "text_overlay": text_score,
            "zoomed_detail": zoomed_score,
            "back_view": back_score,
        }

        # Determine issues
        issues = []

        if text_score > self.text_threshold:
            issues.append("text_overlay")

        if zoomed_score > self.zoom_threshold:
            issues.append("zoomed_detail")

        if back_score > self.back_threshold:
            issues.append("back_view")

        # Image is good if no issues found
        is_good = len(issues) == 0

        return ImageClassification(
            path=image_path,
            is_good=is_good,
            issues=issues,
            scores=scores
        )

    def classify_embedding(self, embedding: np.ndarray) -> Tuple[bool, List[str], Dict[str, float]]:
        """Classify an embedding and return (is_good, issues, scores)."""
        good_score = self._compute_category_score(embedding, self.good_embeddings)
        text_score = self._compute_category_score(embedding, self.text_embeddings)
        zoomed_score = self._compute_category_score(embedding, self.zoomed_embeddings)
        back_score = self._compute_category_score(embedding, self.back_embeddings)

        scores = {
            "good": good_score,
            "text_overlay": text_score,
            "zoomed_detail": zoomed_score,
            "back_view": back_score,
        }

        issues = []
        if text_score > self.text_threshold:
            issues.append("text_overlay")
        if zoomed_score > self.zoom_threshold:
            issues.append("zoomed_detail")
        if back_score > self.back_threshold:
            issues.append("back_view")

        return len(issues) == 0, issues, scores

    def process_directory(
        self,
        images_dir: str,
        output_path: str,
        batch_size: int = 32
    ) -> Dict:
        """
        Process all images in directory using batch processing for speed.
        """
        images_dir = Path(images_dir)

        # Find all image files
        image_files = []
        for ext in ['*.webp', '*.jpg', '*.jpeg', '*.png']:
            image_files.extend(images_dir.rglob(ext))

        print(f"Found {len(image_files)} images to process")

        results = {
            "total_images": len(image_files),
            "good_images": 0,
            "excluded_images": 0,
            "exclusions": [],
            "by_issue": {
                "text_overlay": [],
                "zoomed_detail": [],
                "back_view": [],
                "failed_to_load": [],
            },
            "thresholds": {
                "text": self.text_threshold,
                "zoomed": self.zoom_threshold,
                "back_view": self.back_threshold,
            }
        }

        # Process in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_paths = [str(p) for p in image_files[i:i + batch_size]]
            batch_results = self._encode_images_batch(batch_paths)

            for path, embedding in batch_results:
                rel_path = str(Path(path).relative_to(images_dir))

                if embedding is None:
                    results["excluded_images"] += 1
                    results["exclusions"].append({
                        "path": rel_path,
                        "issues": ["failed_to_load"],
                        "scores": {},
                    })
                    results["by_issue"]["failed_to_load"].append(rel_path)
                    continue

                is_good, issues, scores = self.classify_embedding(embedding)

                if is_good:
                    results["good_images"] += 1
                else:
                    results["excluded_images"] += 1
                    results["exclusions"].append({
                        "path": rel_path,
                        "issues": issues,
                        "scores": scores,
                    })
                    for issue in issues:
                        if issue in results["by_issue"]:
                            results["by_issue"][issue].append(rel_path)

        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"Total: {results['total_images']} images")
        print(f"Good: {results['good_images']} ({100*results['good_images']/results['total_images']:.1f}%)")
        print(f"Excluded: {results['excluded_images']} ({100*results['excluded_images']/results['total_images']:.1f}%)")
        print(f"\nBy issue:")
        for issue, paths in results["by_issue"].items():
            print(f"  {issue}: {len(paths)}")

        return results


def test_on_samples(cleaner: FashionImageCleaner, sample_paths: List[str]):
    """Test the cleaner on specific sample images."""
    print("\n" + "="*60)
    print("Testing on sample images")
    print("="*60)

    for path in sample_paths:
        if not os.path.exists(path):
            print(f"Skip (not found): {path}")
            continue

        result = cleaner.classify_image(path)

        status = "GOOD" if result.is_good else f"EXCLUDE ({', '.join(result.issues)})"
        print(f"\n{Path(path).name}: {status}")
        print(f"  Scores: good={result.scores.get('good', 0):.3f}, "
              f"text={result.scores.get('text_overlay', 0):.3f}, "
              f"zoom={result.scores.get('zoomed_detail', 0):.3f}, "
              f"back={result.scores.get('back_view', 0):.3f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up women's fashion images")
    parser.add_argument("--test", action="store_true", help="Run on test samples only")
    parser.add_argument("--images-dir", default=IMAGES_DIR, help="Images directory")
    parser.add_argument("--output", default=f"{OUTPUT_DIR}/image_exclusions.json", help="Output JSON path")
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--zoom-threshold", type=float, default=ZOOMED_THRESHOLD)
    parser.add_argument("--back-threshold", type=float, default=BACK_VIEW_THRESHOLD)
    args = parser.parse_args()

    # Store thresholds from args
    text_thresh = args.text_threshold
    zoom_thresh = args.zoom_threshold
    back_thresh = args.back_threshold

    # Initialize cleaner with custom thresholds
    cleaner = FashionImageCleaner(
        text_threshold=text_thresh,
        zoom_threshold=zoom_thresh,
        back_threshold=back_thresh
    )

    if args.test:
        # Test on known samples
        test_samples = [
            # Known good images
            f"{args.images_dir}/tops_woven/shirts/200.webp",
            f"{args.images_dir}/outerwear/blazers/138.webp",
            # Known problematic images
            f"{args.images_dir}/tops_woven/shirts/375.webp",  # Has text overlay
            f"{args.images_dir}/sportswear/leggings/50.webp",  # Back view
        ]
        test_on_samples(cleaner, test_samples)
    else:
        # Process full directory
        cleaner.process_directory(args.images_dir, args.output)


if __name__ == "__main__":
    main()
