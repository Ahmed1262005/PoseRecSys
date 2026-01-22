"""
Fashion Style DNA Taxonomy Classifier

Uses FashionCLIP to classify product images against the 201-attribute
Fashion Style DNA taxonomy for use in the onboarding/style quiz UI.

Taxonomy Structure:
- 4 Archetypes: Classic, Natural/Sporty, Dramatic/Street, Creative/Artistic
- 18 Visual Anchors (mid-level features)
- 201 Attributions (leaf-level descriptors with CLIP prompts)
"""
import os
import csv
import logging
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TaxonomyNode:
    """Single node in the taxonomy tree."""
    attribution_id: int
    attribution_name: str
    visual_anchor_id: str
    visual_anchor_name: str
    archetype_id: int
    archetype_name: str
    clip_positive_prompt: str
    clip_negative_prompt: str


@dataclass
class TaxonomyMatch:
    """Match result for a single attribution."""
    attribution_id: int
    attribution_name: str
    visual_anchor_id: str
    visual_anchor_name: str
    archetype_id: int
    archetype_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'attribution_id': self.attribution_id,
            'attribution_name': self.attribution_name,
            'visual_anchor_id': self.visual_anchor_id,
            'visual_anchor_name': self.visual_anchor_name,
            'archetype_id': self.archetype_id,
            'archetype_name': self.archetype_name,
            'confidence': self.confidence,
        }


@dataclass
class ClassificationResult:
    """Full classification result for an image."""
    image_id: str
    top_matches: List[TaxonomyMatch]
    archetype_scores: Dict[int, float]  # Aggregated archetype scores
    visual_anchor_scores: Dict[str, float]  # Aggregated anchor scores

    def get_primary_match(self) -> Optional[TaxonomyMatch]:
        """Get the highest confidence match."""
        return self.top_matches[0] if self.top_matches else None

    def get_primary_archetype(self) -> Tuple[int, str, float]:
        """Get the dominant archetype."""
        if not self.archetype_scores:
            return (0, "", 0.0)
        best_id = max(self.archetype_scores, key=self.archetype_scores.get)
        archetype_names = {
            1: "Classic",
            2: "Natural/Sporty",
            3: "Dramatic/Street",
            4: "Creative/Artistic"
        }
        return (best_id, archetype_names.get(best_id, ""), self.archetype_scores[best_id])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_id': self.image_id,
            'top_matches': [m.to_dict() for m in self.top_matches],
            'archetype_scores': self.archetype_scores,
            'visual_anchor_scores': self.visual_anchor_scores,
            'primary_archetype': self.get_primary_archetype(),
        }


class TaxonomyClassifier:
    """
    CLIP-based classifier for the 201-attribute Fashion Style DNA taxonomy.

    Uses pre-computed CLIP embeddings for images and encodes taxonomy prompts
    to perform zero-shot classification.
    """

    def __init__(
        self,
        taxonomy_csv_path: str,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the taxonomy classifier.

        Args:
            taxonomy_csv_path: Path to fashion_taxonomy_dataset.csv
            cache_dir: Directory to cache computed embeddings
        """
        self.taxonomy_csv_path = taxonomy_csv_path
        self.cache_dir = cache_dir or os.path.dirname(taxonomy_csv_path)

        self._model = None
        self._taxonomy_nodes: List[TaxonomyNode] = []
        self._query_embeddings: Optional[np.ndarray] = None  # Shape: (201, 512)

        # Load taxonomy
        self._load_taxonomy()

    def _load_taxonomy(self):
        """Load taxonomy from CSV file."""
        logger.info(f"Loading taxonomy from {self.taxonomy_csv_path}")

        with open(self.taxonomy_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node = TaxonomyNode(
                    attribution_id=int(row['attribution_id']),
                    attribution_name=row['attribution_name'],
                    visual_anchor_id=row['visual_anchor_id'],
                    visual_anchor_name=row['visual_anchor_name'],
                    archetype_id=int(row['archetype_id']),
                    archetype_name=row['archetype_name'],
                    clip_positive_prompt=row['clip_positive_prompt'],
                    clip_negative_prompt=row['clip_negative_prompt'],
                )
                self._taxonomy_nodes.append(node)

        logger.info(f"Loaded {len(self._taxonomy_nodes)} taxonomy nodes")

        # Validate structure
        archetypes = set(n.archetype_id for n in self._taxonomy_nodes)
        anchors = set(n.visual_anchor_id for n in self._taxonomy_nodes)
        logger.info(f"  Archetypes: {len(archetypes)}, Visual Anchors: {len(anchors)}")

    @property
    def model(self):
        """Lazy load FashionCLIP model."""
        if self._model is None:
            logger.info("Loading FashionCLIP model...")
            from fashion_clip.fashion_clip import FashionCLIP
            self._model = FashionCLIP('fashion-clip')
            logger.info("FashionCLIP loaded")
        return self._model

    @property
    def query_embeddings(self) -> np.ndarray:
        """Get or compute query embeddings for all 201 taxonomy prompts."""
        if self._query_embeddings is None:
            cache_path = os.path.join(self.cache_dir, 'taxonomy_query_embeddings.pkl')

            if os.path.exists(cache_path):
                logger.info(f"Loading cached query embeddings from {cache_path}")
                with open(cache_path, 'rb') as f:
                    self._query_embeddings = pickle.load(f)
            else:
                self._query_embeddings = self._compute_query_embeddings()

                # Cache for future use
                logger.info(f"Caching query embeddings to {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(self._query_embeddings, f)

        return self._query_embeddings

    def _compute_query_embeddings(self) -> np.ndarray:
        """Compute CLIP embeddings for all 201 taxonomy prompts."""
        logger.info("Computing embeddings for 201 taxonomy prompts...")

        embeddings = []
        prompts = [node.clip_positive_prompt for node in self._taxonomy_nodes]

        # Batch encode for efficiency
        batch_size = 32
        for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding prompts"):
            batch = prompts[i:i+batch_size]
            batch_emb = self.model.encode_text(batch, batch_size=len(batch))

            # Normalize each embedding
            for emb in batch_emb:
                emb_norm = emb / np.linalg.norm(emb)
                embeddings.append(emb_norm)

        result = np.array(embeddings, dtype=np.float32)
        logger.info(f"Query embeddings shape: {result.shape}")
        return result

    def classify_embedding(
        self,
        image_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[TaxonomyMatch]:
        """
        Classify a single image embedding against all 201 taxonomy prompts.

        Args:
            image_embedding: 512-dim FashionCLIP embedding
            top_k: Number of top matches to return

        Returns:
            List of top-k TaxonomyMatch objects sorted by confidence
        """
        # Normalize image embedding
        if np.linalg.norm(image_embedding) > 0:
            image_norm = image_embedding / np.linalg.norm(image_embedding)
        else:
            image_norm = image_embedding

        # Compute cosine similarity with all 201 prompts
        similarities = np.dot(self.query_embeddings, image_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build match objects
        matches = []
        for idx in top_indices:
            node = self._taxonomy_nodes[idx]
            matches.append(TaxonomyMatch(
                attribution_id=node.attribution_id,
                attribution_name=node.attribution_name,
                visual_anchor_id=node.visual_anchor_id,
                visual_anchor_name=node.visual_anchor_name,
                archetype_id=node.archetype_id,
                archetype_name=node.archetype_name,
                confidence=float(similarities[idx]),
            ))

        return matches

    def classify_image(
        self,
        image_path: str,
        top_k: int = 10,
    ) -> Optional[List[TaxonomyMatch]]:
        """
        Classify an image file directly.

        Args:
            image_path: Path to image file
            top_k: Number of top matches to return

        Returns:
            List of TaxonomyMatch objects or None if failed
        """
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            embedding = self.model.encode_images([image], batch_size=1)[0]
            return self.classify_embedding(embedding, top_k)
        except Exception as e:
            logger.error(f"Error classifying {image_path}: {e}")
            return None

    def batch_classify(
        self,
        embeddings: Dict[str, np.ndarray],
        top_k: int = 5,
        show_progress: bool = True,
    ) -> Dict[str, ClassificationResult]:
        """
        Classify all embeddings in batch.

        Args:
            embeddings: Dict mapping image_id to 512-dim embedding
            top_k: Number of top matches per image
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping image_id to ClassificationResult
        """
        logger.info(f"Classifying {len(embeddings)} images against 201 taxonomy prompts...")

        # Stack all embeddings for batch processing
        image_ids = list(embeddings.keys())
        image_embs = np.array([embeddings[id_] for id_ in image_ids], dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(image_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        image_embs_norm = image_embs / norms

        # Compute all similarities at once: (N, 201)
        # image_embs_norm: (N, 512), query_embeddings: (201, 512)
        all_similarities = np.dot(image_embs_norm, self.query_embeddings.T)

        results = {}
        iterator = tqdm(range(len(image_ids)), desc="Building results") if show_progress else range(len(image_ids))

        for i in iterator:
            image_id = image_ids[i]
            similarities = all_similarities[i]

            # Get top-k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]

            matches = []
            for idx in top_indices:
                node = self._taxonomy_nodes[idx]
                matches.append(TaxonomyMatch(
                    attribution_id=node.attribution_id,
                    attribution_name=node.attribution_name,
                    visual_anchor_id=node.visual_anchor_id,
                    visual_anchor_name=node.visual_anchor_name,
                    archetype_id=node.archetype_id,
                    archetype_name=node.archetype_name,
                    confidence=float(similarities[idx]),
                ))

            # Compute aggregated scores
            archetype_scores = self._aggregate_archetype_scores(similarities)
            anchor_scores = self._aggregate_anchor_scores(similarities)

            results[image_id] = ClassificationResult(
                image_id=image_id,
                top_matches=matches,
                archetype_scores=archetype_scores,
                visual_anchor_scores=anchor_scores,
            )

        logger.info(f"Classification complete for {len(results)} images")
        return results

    def _aggregate_archetype_scores(self, similarities: np.ndarray) -> Dict[int, float]:
        """Aggregate similarity scores by archetype (average of all attributions in archetype)."""
        archetype_sums = {1: [], 2: [], 3: [], 4: []}

        for i, node in enumerate(self._taxonomy_nodes):
            archetype_sums[node.archetype_id].append(similarities[i])

        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in archetype_sums.items()
        }

    def _aggregate_anchor_scores(self, similarities: np.ndarray) -> Dict[str, float]:
        """Aggregate similarity scores by visual anchor (average of all attributions in anchor)."""
        anchor_sums = {}

        for i, node in enumerate(self._taxonomy_nodes):
            anchor_id = node.visual_anchor_id
            if anchor_id not in anchor_sums:
                anchor_sums[anchor_id] = []
            anchor_sums[anchor_id].append(similarities[i])

        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in anchor_sums.items()
        }

    def get_taxonomy_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded taxonomy."""
        archetypes = {}
        anchors = {}

        for node in self._taxonomy_nodes:
            # Count by archetype
            if node.archetype_id not in archetypes:
                archetypes[node.archetype_id] = {
                    'name': node.archetype_name,
                    'count': 0,
                    'anchors': set()
                }
            archetypes[node.archetype_id]['count'] += 1
            archetypes[node.archetype_id]['anchors'].add(node.visual_anchor_id)

            # Count by anchor
            if node.visual_anchor_id not in anchors:
                anchors[node.visual_anchor_id] = {
                    'name': node.visual_anchor_name,
                    'archetype': node.archetype_name,
                    'count': 0
                }
            anchors[node.visual_anchor_id]['count'] += 1

        # Convert sets to counts
        for a in archetypes.values():
            a['anchors'] = len(a['anchors'])

        return {
            'total_attributions': len(self._taxonomy_nodes),
            'archetypes': {
                k: {'name': v['name'], 'attributions': v['count'], 'visual_anchors': v['anchors']}
                for k, v in archetypes.items()
            },
            'visual_anchors': anchors,
        }

    def get_attribution_by_id(self, attribution_id: int) -> Optional[TaxonomyNode]:
        """Get a taxonomy node by its attribution ID."""
        for node in self._taxonomy_nodes:
            if node.attribution_id == attribution_id:
                return node
        return None

    def get_attributions_by_archetype(self, archetype_id: int) -> List[TaxonomyNode]:
        """Get all attributions for a given archetype."""
        return [n for n in self._taxonomy_nodes if n.archetype_id == archetype_id]

    def get_attributions_by_anchor(self, visual_anchor_id: str) -> List[TaxonomyNode]:
        """Get all attributions for a given visual anchor."""
        return [n for n in self._taxonomy_nodes if n.visual_anchor_id == visual_anchor_id]


def main():
    """CLI for testing taxonomy classification."""
    import argparse

    parser = argparse.ArgumentParser(description='Fashion Style DNA Taxonomy Classifier')
    parser.add_argument('--taxonomy', type=str,
                       default='/home/ubuntu/recSys/outfitTransformer/fashion_taxonomy_dataset.csv',
                       help='Path to taxonomy CSV')
    parser.add_argument('--stats', action='store_true', help='Show taxonomy statistics')
    parser.add_argument('--test-image', type=str, help='Test classification on single image')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top matches to show')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    classifier = TaxonomyClassifier(args.taxonomy)

    if args.stats:
        print("\n=== Taxonomy Statistics ===")
        stats = classifier.get_taxonomy_stats()
        print(f"Total Attributions: {stats['total_attributions']}")
        print("\nArchetypes:")
        for k, v in stats['archetypes'].items():
            print(f"  {k}. {v['name']}: {v['attributions']} attributions, {v['visual_anchors']} anchors")
        print("\nVisual Anchors:")
        for k, v in stats['visual_anchors'].items():
            print(f"  {k} ({v['archetype']}): {v['name']} - {v['count']} attributions")

    if args.test_image:
        print(f"\n=== Classifying: {args.test_image} ===")
        matches = classifier.classify_image(args.test_image, top_k=args.top_k)

        if matches:
            print(f"\nTop {len(matches)} Taxonomy Matches:")
            for i, m in enumerate(matches):
                print(f"\n[{i+1}] {m.attribution_name} (ID: {m.attribution_id})")
                print(f"    Archetype: {m.archetype_name}")
                print(f"    Visual Anchor: {m.visual_anchor_name}")
                print(f"    Confidence: {m.confidence:.4f}")
        else:
            print("Classification failed")


if __name__ == '__main__':
    main()
