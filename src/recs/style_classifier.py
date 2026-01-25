"""
Style Classifier Module

Zero-shot product classification using FashionCLIP text embeddings.
Classifies products for coverage styles and occasion attributes.

Usage:
    classifier = StyleClassifier()
    scores = classifier.classify_product(product_embedding)
    # Returns: {'styles': {'sheer': 0.35, 'deep-necklines': 0.12, ...},
    #           'occasions': {'casual': 0.45, 'office': 0.32, ...}}

The scores are stored in the database, allowing threshold adjustment
without re-running classification.
"""

import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class StyleClassifierConfig:
    """Configuration for style classifier."""

    # Model configuration
    model_name: str = "patrickjohncyh/fashion-clip"
    device: str = "auto"  # "auto", "cuda", or "cpu"

    # Default thresholds (can be overridden at query time in SQL)
    default_style_threshold: float = 0.25
    default_occasion_threshold: float = 0.18  # Lowered - median occasion score is ~0.19
    default_pattern_threshold: float = 0.30

    # Coverage/Style concepts (things user might want to avoid)
    # Matches frontend: deep-necklines, open-back, sheer, cutouts, high-slits, crop-tops, mini-lengths, sleeveless
    # Each concept has multiple text descriptions for robust matching
    style_concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        'deep-necklines': [
            'deep v-neck top',
            'plunging neckline dress',
            'low cut blouse',
            'deep scoop neck shirt',
        ],
        'open-back': [
            'open back dress',
            'backless top',
            'low back gown',
            'backless blouse',
            'exposed back dress',
        ],
        'sheer': [
            'sheer blouse',
            'see-through fabric top',
            'transparent mesh clothing',
            'sheer lace dress',
        ],
        'cutouts': [
            'cutout dress with exposed skin',
            'side cutout top',
            'midriff cutout bodysuit',
            'back cutout dress',
        ],
        'high-slits': [
            'high slit dress',
            'thigh-high slit skirt',
            'leg slit evening gown',
            'side slit maxi dress',
        ],
        'crop-tops': [
            'crop top',
            'cropped shirt showing midriff',
            'belly-baring top',
            'short cropped tee',
        ],
        'mini-lengths': [
            'mini skirt',
            'mini dress',
            'very short skirt above knee',
            'short hemline dress',
        ],
        'sleeveless': [
            'sleeveless top',
            'tank top',
            'no sleeves blouse',
            'sleeveless dress',
        ],
    })

    # Occasion concepts - matches frontend: casual, office, smart-casual, evening, events, beach
    # EXPANDED vocabulary for better discrimination - especially office vs active
    occasion_concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        'casual': [
            'casual everyday t-shirt and jeans outfit',
            'relaxed weekend wear sweatshirt',
            'comfortable casual loungewear',
            'laid-back casual hoodie style',
            'informal everyday streetwear',
            'cozy at-home casual clothes',
            'effortless casual denim look',
            'weekend errand running outfit',
        ],
        'office': [
            # Highly specific professional office items
            'formal business suit blazer and dress pants',
            'corporate professional button-down shirt and slacks',
            'office workwear tailored blouse and pencil skirt',
            'boardroom meeting professional attire',
            'conservative office appropriate dress',
            'executive business formal clothing',
            'structured tailored professional jacket',
            'crisp dress shirt with collar for office',
            # Explicitly NOT athletic/casual
            'non-athletic formal work clothing',
            'traditional office professional wear',
            'polished business meeting attire',
            'sophisticated workplace fashion',
        ],
        'smart-casual': [
            'smart casual dinner outfit',
            'polished casual blazer with jeans',
            'elevated everyday chinos and button-up',
            'dressed up casual brunch look',
            'refined casual date night outfit',
            'nice restaurant casual attire',
            'sophisticated weekend outing clothes',
            'upscale casual social gathering wear',
        ],
        'evening': [
            'evening cocktail dress formal',
            'night out party dress',
            'glamorous evening gown',
            'formal evening black tie attire',
            'upscale dinner date outfit',
            'elegant nightclub party wear',
            'sophisticated evening event dress',
            'dressy going out nighttime look',
        ],
        'events': [
            'special event formal dress',
            'party occasion cocktail outfit',
            'celebration wedding guest wear',
            'festive event holiday clothing',
            'graduation ceremony dress',
            'gala event formal attire',
            'birthday party outfit',
            'special occasion dressy look',
        ],
        'beach': [
            'beach vacation swimwear cover-up',
            'resort wear tropical outfit',
            'summer beach casual clothes',
            'tropical vacation linen clothing',
            'seaside holiday outfit',
            'poolside resort wear',
            'coastal beach day attire',
            'sunny beach vacation look',
        ],
        'active': [
            # Highly specific athletic/activewear items
            'athletic gym workout leggings',
            'sports performance running clothes',
            'yoga studio fitness wear',
            'moisture-wicking athletic apparel',
            'stretchy exercise workout clothes',
            'sweat-wicking gym training outfit',
            'high-performance athletic sportswear',
            'breathable running jogging attire',
            'compression athletic training gear',
            'flexible yoga pilates clothes',
            'sporty athletic brand activewear',
            'technical fitness performance wear',
        ],
    })

    # Pattern concepts - matches frontend: stripes, floral, animal-print, polka-dots, plaid, paisley, abstract, geometric
    pattern_concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        'stripes': [
            'striped pattern clothing',
            'horizontal stripes garment',
            'vertical stripes fabric',
            'pinstripe pattern',
        ],
        'floral': [
            'floral pattern clothing',
            'flower print dress',
            'botanical floral fabric',
            'rose flower pattern',
        ],
        'animal-print': [
            'animal print clothing',
            'leopard print pattern',
            'zebra stripes animal pattern',
            'snake skin print fabric',
        ],
        'polka-dots': [
            'polka dot clothing',
            'dotted pattern fabric',
            'small dots pattern garment',
            'classic polka dots',
        ],
        'plaid': [
            'plaid pattern clothing',
            'tartan plaid fabric',
            'checkered plaid garment',
            'scottish plaid pattern',
        ],
        'paisley': [
            'paisley pattern clothing',
            'paisley print fabric',
            'teardrop paisley design',
            'bohemian paisley pattern',
        ],
        'abstract': [
            'abstract pattern clothing',
            'artistic abstract print',
            'modern abstract design fabric',
            'contemporary abstract pattern',
        ],
        'geometric': [
            'geometric pattern clothing',
            'abstract geometric shapes fabric',
            'triangles circles geometric print',
            'modern geometric design',
        ],
        'logo': [
            'clothing with visible brand logo',
            'shirt with logo emblem on chest',
            'garment with brand name text printed',
            'polo shirt with small brand logo',
            'sportswear with athletic brand logo',
            'clothing with company logo marking',
        ],
        'graphic': [
            't-shirt with graphic print artwork',
            'clothing with printed illustration design',
            'shirt with text slogan graphic print',
            'top with decorative picture print',
            'garment with artistic graphic design',
            'tee with screen printed graphic image',
        ],
    })

    # Fit concepts - matches frontend: slim, fitted, regular, relaxed, oversized
    # Used for all categories (tops, bottoms, dresses)
    fit_concepts_tops: Dict[str, List[str]] = field(default_factory=lambda: {
        'slim': [
            'slim fit top clothing',
            'skinny tight top',
            'body hugging slim shirt',
            'narrow slim top',
        ],
        'fitted': [
            'fitted top clothing',
            'form fitting shirt',
            'tailored fitted blouse',
            'close fitting top',
        ],
        'regular': [
            'regular fit top clothing',
            'standard fit shirt',
            'classic fit top',
            'normal fit blouse',
        ],
        'relaxed': [
            'relaxed fit top clothing',
            'loose comfortable top',
            'casual relaxed shirt',
            'easy fit blouse',
        ],
        'oversized': [
            'oversized fit top clothing',
            'baggy oversized shirt',
            'extra large loose top',
            'slouchy oversized tee',
        ],
    })

    # BOTTOMS fit concepts
    fit_concepts_bottoms: Dict[str, List[str]] = field(default_factory=lambda: {
        'slim': [
            'slim fit pants jeans',
            'skinny tight trousers',
            'narrow slim leg pants',
            'tapered slim bottoms',
        ],
        'fitted': [
            'fitted pants jeans',
            'form fitting trousers',
            'tailored fitted bottoms',
            'close fit pants',
        ],
        'regular': [
            'regular fit pants jeans',
            'straight leg trousers',
            'standard fit bottoms',
            'classic fit pants',
        ],
        'relaxed': [
            'relaxed fit pants jeans',
            'loose comfortable trousers',
            'easy fit casual pants',
            'roomier relaxed bottoms',
        ],
        'oversized': [
            'oversized fit pants jeans',
            'baggy oversized trousers',
            'extra wide loose pants',
            'super relaxed bottoms',
        ],
    })

    # DRESSES fit concepts
    fit_concepts_dresses: Dict[str, List[str]] = field(default_factory=lambda: {
        'slim': [
            'slim fit dress',
            'narrow silhouette dress',
            'body skimming slim dress',
            'pencil slim dress',
        ],
        'fitted': [
            'fitted bodycon dress',
            'figure hugging fitted dress',
            'form fitting dress',
            'tailored fitted gown',
        ],
        'regular': [
            'regular fit dress',
            'standard fit dress',
            'classic silhouette dress',
            'normal fit dress',
        ],
        'relaxed': [
            'relaxed loose fitting dress',
            'flowy comfortable dress',
            'easy casual loose dress',
            'shift relaxed dress',
        ],
        'oversized': [
            'oversized loose dress',
            'boxy oversized dress',
            'extra roomy dress',
            'tent style oversized dress',
        ],
    })

    # Length concepts for tops/bottoms
    length_concepts_tops: Dict[str, List[str]] = field(default_factory=lambda: {
        'cropped': [
            'cropped short top',
            'belly showing crop top',
            'short cropped shirt',
            'midriff baring crop',
        ],
        'standard': [
            'standard regular length top',
            'hip length normal top',
            'regular length shirt',
            'classic length blouse',
        ],
        'long': [
            'longline long top tunic',
            'extra long shirt',
            'extended length tunic',
            'longer than hip top',
        ],
    })

    length_concepts_bottoms: Dict[str, List[str]] = field(default_factory=lambda: {
        'cropped': [
            'cropped short pants',
            'ankle cropped trousers',
            'capri cropped bottoms',
            'shorter cropped jeans',
        ],
        'standard': [
            'standard regular length pants',
            'full length trousers',
            'normal length jeans',
            'regular inseam bottoms',
        ],
        'long': [
            'extra long pants',
            'tall length trousers',
            'longer inseam jeans',
            'floor length bottoms',
        ],
    })

    # Length concepts for dresses/skirts (different scale)
    length_concepts_dresses: Dict[str, List[str]] = field(default_factory=lambda: {
        'mini': [
            'mini short dress above knee',
            'mini skirt thigh length',
            'very short mini hemline',
            'above knee mini dress',
        ],
        'midi': [
            'midi dress knee length',
            'midi skirt below knee',
            'mid calf midi length',
            'knee to calf midi',
        ],
        'maxi': [
            'maxi long dress floor length',
            'maxi skirt ankle length',
            'full length maxi gown',
            'floor sweeping maxi dress',
        ],
    })

    # Sleeve concepts - matches frontend: sleeveless, short-sleeve, 3-4-sleeve, long-sleeve
    sleeve_concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        'sleeveless': [
            'sleeveless no sleeve clothing',
            'tank top no sleeves',
            'armhole sleeveless garment',
            'without sleeves top',
        ],
        'short-sleeve': [
            'short sleeve clothing',
            'cap sleeve short top',
            'half sleeve shirt',
            'tee shirt short sleeves',
        ],
        '3-4-sleeve': [
            'three quarter 3/4 sleeve clothing',
            'elbow length sleeve',
            'mid arm 3/4 sleeve',
            'bracelet length sleeve top',
        ],
        'long-sleeve': [
            'long sleeve clothing',
            'full length long sleeves',
            'wrist length sleeve top',
            'long arm coverage sleeves',
        ],
    })

    # Rise concepts for bottoms - matches frontend: high-rise, mid-rise, low-rise
    rise_concepts: Dict[str, List[str]] = field(default_factory=lambda: {
        'high-rise': [
            'high rise waist pants',
            'high waisted jeans',
            'above belly button rise pants',
            'high rise trousers',
        ],
        'mid-rise': [
            'mid rise waist pants',
            'medium rise jeans',
            'regular rise trousers',
            'mid waist bottoms',
        ],
        'low-rise': [
            'low rise waist pants',
            'low waisted jeans',
            'hip hugging low rise',
            'low rise trousers',
        ],
    })

    # =================================================================
    # NEGATIVE OCCASION CONCEPTS - What each occasion is NOT
    # Used for contrastive scoring to improve hard gating accuracy
    # =================================================================
    occasion_concepts_negative: Dict[str, List[str]] = field(default_factory=lambda: {
        'office': [
            # Athletic/activewear - explicitly NOT office
            'athletic gym workout leggings activewear',
            'sports performance running clothes',
            'yoga studio fitness wear stretchy',
            'moisture-wicking athletic sportswear',
            'compression athletic training gear',
            # Casual beach/vacation - NOT office
            'casual beach vacation swimwear',
            'pool party resort wear',
            'tropical vacation casual outfit',
            # Revealing/casual items - NOT office
            'crop top tank top sleeveless casual',
            'sexy bodycon mini dress nightclub',
            'distressed ripped jeans casual weekend',
            'hoodie sweatshirt athleisure lounge',
            # Athletic brands product style
            'sporty athletic brand activewear',
            'gym training workout clothes',
        ],
        'casual': [
            # Formal business wear - NOT casual
            'formal business suit blazer',
            'corporate professional attire',
            'boardroom meeting executive clothing',
            # Evening formal - NOT casual
            'elegant evening gown cocktail dress',
            'black tie formal event attire',
            'glamorous party sequin dress',
        ],
        'evening': [
            # Daytime casual - NOT evening
            'casual everyday t-shirt jeans',
            'weekend relaxed loungewear',
            'athletic gym workout clothes',
            # Office business - NOT evening
            'conservative office workwear',
            'professional business meeting attire',
        ],
        'active': [
            # Formal/office - NOT active
            'formal business suit office wear',
            'elegant evening cocktail dress',
            'professional corporate attire',
            'conservative work appropriate dress',
            # Dressy items - NOT active
            'silk blouse formal shirt',
            'structured blazer jacket',
            'pencil skirt dress pants',
        ],
        'beach': [
            # Office/formal - NOT beach
            'formal business office attire',
            'corporate professional clothing',
            'executive boardroom wear',
            # Winter/covered items - NOT beach
            'heavy coat winter jacket',
            'formal evening gown',
        ],
        'smart-casual': [
            # Very formal - NOT smart-casual
            'black tie formal evening gown',
            'full business suit formal',
            # Very casual - NOT smart-casual
            'athletic gym workout wear',
            'distressed ripped casual clothes',
            'beach swimwear vacation',
        ],
        'events': [
            # Everyday casual - NOT events
            'casual everyday loungewear',
            'athletic gym workout clothes',
            'basic t-shirt jeans weekend',
        ],
    })


class StyleClassifier:
    """
    Zero-shot product classification using FashionCLIP text embeddings.

    Computes similarity between product image embeddings and text concept embeddings.
    Returns raw similarity scores (not binary tags) so thresholds can be adjusted
    at query time without re-running classification.
    """

    def __init__(self, config: Optional[StyleClassifierConfig] = None):
        """
        Initialize the style classifier.

        Args:
            config: Configuration options. If None, uses defaults.
        """
        self.config = config or StyleClassifierConfig()
        self.model = None
        self.processor = None
        self.device = None

        # Pre-computed concept embeddings (lazily loaded)
        self._style_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._occasion_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._occasion_negative_embeddings: Optional[Dict[str, np.ndarray]] = None  # NEW: Negative occasion concepts
        self._pattern_embeddings: Optional[Dict[str, np.ndarray]] = None
        # Fit embeddings by category
        self._fit_embeddings_tops: Optional[Dict[str, np.ndarray]] = None
        self._fit_embeddings_bottoms: Optional[Dict[str, np.ndarray]] = None
        self._fit_embeddings_dresses: Optional[Dict[str, np.ndarray]] = None
        # Length embeddings by category
        self._length_embeddings_tops: Optional[Dict[str, np.ndarray]] = None
        self._length_embeddings_bottoms: Optional[Dict[str, np.ndarray]] = None
        self._length_embeddings_dresses: Optional[Dict[str, np.ndarray]] = None
        # Sleeve embeddings
        self._sleeve_embeddings: Optional[Dict[str, np.ndarray]] = None
        # Rise embeddings (bottoms only)
        self._rise_embeddings: Optional[Dict[str, np.ndarray]] = None

    def _ensure_model_loaded(self):
        """Lazily load the model and compute concept embeddings."""
        if self.model is not None:
            return

        import torch
        from transformers import CLIPProcessor, CLIPModel

        print(f"[StyleClassifier] Loading model: {self.config.model_name}")

        # Determine device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        print(f"[StyleClassifier] Using device: {self.device}")

        # Load model and processor
        self.model = CLIPModel.from_pretrained(self.config.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-compute concept embeddings
        self._precompute_concept_embeddings()

        print(f"[StyleClassifier] Model loaded successfully")

    def _precompute_concept_embeddings(self):
        """Pre-compute and cache text embeddings for all concepts."""
        import torch

        self._style_embeddings = {}
        self._occasion_embeddings = {}
        self._occasion_negative_embeddings = {}  # NEW: Negative occasion concepts
        self._pattern_embeddings = {}
        self._fit_embeddings_tops = {}
        self._fit_embeddings_bottoms = {}
        self._fit_embeddings_dresses = {}
        self._length_embeddings_tops = {}
        self._length_embeddings_bottoms = {}
        self._length_embeddings_dresses = {}
        self._sleeve_embeddings = {}
        self._rise_embeddings = {}

        # Compute style concept embeddings
        for style_name, text_descriptions in self.config.style_concepts.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._style_embeddings[style_name] = embedding

        # Compute occasion concept embeddings
        for occasion_name, text_descriptions in self.config.occasion_concepts.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._occasion_embeddings[occasion_name] = embedding

        # Compute NEGATIVE occasion concept embeddings (what each occasion is NOT)
        for occasion_name, text_descriptions in self.config.occasion_concepts_negative.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._occasion_negative_embeddings[occasion_name] = embedding

        # Compute pattern concept embeddings
        for pattern_name, text_descriptions in self.config.pattern_concepts.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._pattern_embeddings[pattern_name] = embedding

        # Compute fit embeddings for each category
        for fit_name, text_descriptions in self.config.fit_concepts_tops.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._fit_embeddings_tops[fit_name] = embedding

        for fit_name, text_descriptions in self.config.fit_concepts_bottoms.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._fit_embeddings_bottoms[fit_name] = embedding

        for fit_name, text_descriptions in self.config.fit_concepts_dresses.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._fit_embeddings_dresses[fit_name] = embedding

        # Compute length embeddings for each category
        for length_name, text_descriptions in self.config.length_concepts_tops.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._length_embeddings_tops[length_name] = embedding

        for length_name, text_descriptions in self.config.length_concepts_bottoms.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._length_embeddings_bottoms[length_name] = embedding

        for length_name, text_descriptions in self.config.length_concepts_dresses.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._length_embeddings_dresses[length_name] = embedding

        # Compute sleeve embeddings
        for sleeve_name, text_descriptions in self.config.sleeve_concepts.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._sleeve_embeddings[sleeve_name] = embedding

        # Compute rise embeddings (bottoms only)
        for rise_name, text_descriptions in self.config.rise_concepts.items():
            embedding = self._get_text_embedding(text_descriptions)
            self._rise_embeddings[rise_name] = embedding

        print(f"[StyleClassifier] Pre-computed {len(self._style_embeddings)} style embeddings")
        print(f"[StyleClassifier] Pre-computed {len(self._occasion_embeddings)} occasion embeddings (positive)")
        print(f"[StyleClassifier] Pre-computed {len(self._occasion_negative_embeddings)} occasion embeddings (negative)")
        print(f"[StyleClassifier] Pre-computed {len(self._pattern_embeddings)} pattern embeddings")
        print(f"[StyleClassifier] Pre-computed fit embeddings: tops={len(self._fit_embeddings_tops)}, bottoms={len(self._fit_embeddings_bottoms)}, dresses={len(self._fit_embeddings_dresses)}")
        print(f"[StyleClassifier] Pre-computed length embeddings: tops={len(self._length_embeddings_tops)}, bottoms={len(self._length_embeddings_bottoms)}, dresses={len(self._length_embeddings_dresses)}")
        print(f"[StyleClassifier] Pre-computed {len(self._sleeve_embeddings)} sleeve embeddings")
        print(f"[StyleClassifier] Pre-computed {len(self._rise_embeddings)} rise embeddings")

    def _get_text_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Get averaged, normalized text embedding for multiple descriptions.

        Uses multiple descriptions per concept for more robust matching.
        """
        import torch

        with torch.no_grad():
            # Process texts
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get text features
            outputs = self.model.get_text_features(**inputs)

            # Average across all descriptions
            averaged = outputs.mean(dim=0)

            # Normalize using torch for numerical stability
            normalized = torch.nn.functional.normalize(averaged, p=2, dim=0)

            # Convert to numpy
            return normalized.cpu().numpy().astype(np.float32)

    def classify_product(
        self,
        product_embedding: Any,
        include_patterns: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Classify a product's styles, occasions, and patterns based on its embedding.

        Args:
            product_embedding: Product's FashionCLIP image embedding.
                Can be a list (from JSON), numpy array, or torch tensor.
            include_patterns: Whether to include pattern classification (default True)
            metadata: Optional dict with 'name', 'brand', 'article_type', 'broad_category'
                      If provided, applies intelligent occasion score adjustments

        Returns:
            {
                'styles': {'sheer': 0.35, 'deep-necklines': 0.12, ...},
                'occasions': {'casual': 0.45, 'office': 0.32, ...},
                'patterns': {'solid': 0.80, 'stripes': 0.15, ...}  # if include_patterns
            }
        """
        self._ensure_model_loaded()

        # Convert embedding to normalized numpy array
        product_emb = self._to_normalized_embedding(product_embedding)

        # Compute style scores
        style_scores = {}
        for style_name, concept_embedding in self._style_embeddings.items():
            similarity = float(np.dot(product_emb, concept_embedding))
            style_scores[style_name] = round(similarity, 4)

        # Compute occasion scores
        occasion_scores = {}
        for occasion_name, concept_embedding in self._occasion_embeddings.items():
            similarity = float(np.dot(product_emb, concept_embedding))
            occasion_scores[occasion_name] = round(similarity, 4)

        # Apply metadata-based adjustments if provided
        if metadata:
            occasion_scores = self._adjust_occasion_scores(
                occasion_scores, style_scores, metadata
            )

        result = {
            'styles': style_scores,
            'occasions': occasion_scores
        }

        # Compute pattern scores
        if include_patterns:
            pattern_scores = {}
            for pattern_name, concept_embedding in self._pattern_embeddings.items():
                similarity = float(np.dot(product_emb, concept_embedding))
                pattern_scores[pattern_name] = round(similarity, 4)
            result['patterns'] = pattern_scores

        return result

    def classify_attributes(
        self,
        product_embedding: Any,
        broad_category: str,
    ) -> Dict[str, Optional[str]]:
        """
        Classify a product's fit, length, sleeve, and rise attributes based on its category.

        Uses the appropriate concept embeddings based on the product's broad category.
        Returns the attribute value with the highest score.

        Args:
            product_embedding: Product's FashionCLIP image embedding.
            broad_category: The broad category (e.g., 'tops', 'bottoms', 'dresses')

        Returns:
            {
                'fit': 'regular',       # or 'slim', 'fitted', 'relaxed', 'oversized'
                'length': 'standard',   # or 'cropped', 'long' (tops/bottoms) / 'mini', 'midi', 'maxi' (dresses)
                'sleeve': 'short-sleeve', # or 'long-sleeve', 'sleeveless', '3-4-sleeve' (for applicable categories)
                'rise': 'mid-rise',     # or 'high-rise', 'low-rise' (bottoms only)
            }
        """
        self._ensure_model_loaded()

        # Convert embedding to normalized numpy array
        product_emb = self._to_normalized_embedding(product_embedding)
        broad_category_lower = broad_category.lower() if broad_category else ''

        result = {
            'fit': None,
            'length': None,
            'sleeve': None,
            'rise': None,
        }

        # Determine which fit embeddings to use based on category
        fit_embeddings = None
        if 'top' in broad_category_lower or 'knit' in broad_category_lower or 'woven' in broad_category_lower:
            fit_embeddings = self._fit_embeddings_tops
        elif 'bottom' in broad_category_lower or 'pant' in broad_category_lower or 'trouser' in broad_category_lower or 'jean' in broad_category_lower:
            fit_embeddings = self._fit_embeddings_bottoms
        elif 'dress' in broad_category_lower or 'skirt' in broad_category_lower:
            fit_embeddings = self._fit_embeddings_dresses

        if fit_embeddings:
            best_fit = None
            best_score = -1
            for fit_name, concept_embedding in fit_embeddings.items():
                similarity = float(np.dot(product_emb, concept_embedding))
                if similarity > best_score:
                    best_score = similarity
                    best_fit = fit_name
            result['fit'] = best_fit

        # Determine which length embeddings to use
        length_embeddings = None
        if 'top' in broad_category_lower or 'knit' in broad_category_lower or 'woven' in broad_category_lower:
            length_embeddings = self._length_embeddings_tops
        elif 'bottom' in broad_category_lower or 'pant' in broad_category_lower or 'trouser' in broad_category_lower or 'jean' in broad_category_lower:
            length_embeddings = self._length_embeddings_bottoms
        elif 'dress' in broad_category_lower or 'skirt' in broad_category_lower:
            length_embeddings = self._length_embeddings_dresses

        if length_embeddings:
            best_length = None
            best_score = -1
            for length_name, concept_embedding in length_embeddings.items():
                similarity = float(np.dot(product_emb, concept_embedding))
                if similarity > best_score:
                    best_score = similarity
                    best_length = length_name
            result['length'] = best_length

        # Sleeve classification (for tops, dresses, outerwear)
        if ('top' in broad_category_lower or 'dress' in broad_category_lower or
            'outer' in broad_category_lower or 'knit' in broad_category_lower or
            'woven' in broad_category_lower):
            best_sleeve = None
            best_score = -1
            for sleeve_name, concept_embedding in self._sleeve_embeddings.items():
                similarity = float(np.dot(product_emb, concept_embedding))
                if similarity > best_score:
                    best_score = similarity
                    best_sleeve = sleeve_name
            result['sleeve'] = best_sleeve

        # Rise classification (for bottoms only)
        if ('bottom' in broad_category_lower or 'pant' in broad_category_lower or
            'trouser' in broad_category_lower or 'jean' in broad_category_lower or
            'short' in broad_category_lower):
            best_rise = None
            best_score = -1
            for rise_name, concept_embedding in self._rise_embeddings.items():
                similarity = float(np.dot(product_emb, concept_embedding))
                if similarity > best_score:
                    best_score = similarity
                    best_rise = rise_name
            result['rise'] = best_rise

        return result

    def classify_products_batch(
        self,
        product_embeddings: List[Any],
        batch_size: int = 100,
        include_patterns: bool = True,
        product_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Classify multiple products efficiently.

        Args:
            product_embeddings: List of product embeddings
            batch_size: Process in batches for memory efficiency
            include_patterns: Whether to include pattern classification
            product_metadata: Optional list of dicts with 'name', 'brand', 'article_type', 'broad_category'
                              If provided, applies category-based and keyword-based adjustments

        Returns:
            List of classification results, one per product
        """
        self._ensure_model_loaded()

        results = []

        for i in range(0, len(product_embeddings), batch_size):
            batch = product_embeddings[i:i + batch_size]
            batch_metadata = product_metadata[i:i + batch_size] if product_metadata else None

            # Convert batch to matrix
            emb_matrix = np.stack([
                self._to_normalized_embedding(emb) for emb in batch
            ])

            # Batch compute style scores
            style_matrix = np.stack(list(self._style_embeddings.values()))
            style_similarities = np.dot(emb_matrix, style_matrix.T)

            # Batch compute occasion scores
            occasion_matrix = np.stack(list(self._occasion_embeddings.values()))
            occasion_similarities = np.dot(emb_matrix, occasion_matrix.T)

            # Batch compute pattern scores
            pattern_similarities = None
            if include_patterns:
                pattern_matrix = np.stack(list(self._pattern_embeddings.values()))
                pattern_similarities = np.dot(emb_matrix, pattern_matrix.T)

            # Format results
            style_names = list(self._style_embeddings.keys())
            occasion_names = list(self._occasion_embeddings.keys())
            pattern_names = list(self._pattern_embeddings.keys()) if include_patterns else []

            for j in range(len(batch)):
                style_scores = {
                    name: round(float(style_similarities[j, k]), 4)
                    for k, name in enumerate(style_names)
                }
                occasion_scores = {
                    name: round(float(occasion_similarities[j, k]), 4)
                    for k, name in enumerate(occasion_names)
                }

                # Apply metadata-based adjustments if provided
                if batch_metadata and j < len(batch_metadata):
                    metadata = batch_metadata[j] or {}
                    occasion_scores = self._adjust_occasion_scores(
                        occasion_scores, style_scores, metadata
                    )

                result = {
                    'styles': style_scores,
                    'occasions': occasion_scores
                }

                if include_patterns and pattern_similarities is not None:
                    pattern_scores = {
                        name: round(float(pattern_similarities[j, k]), 4)
                        for k, name in enumerate(pattern_names)
                    }
                    result['patterns'] = pattern_scores

                results.append(result)

        return results

    def _adjust_occasion_scores(
        self,
        occasion_scores: Dict[str, float],
        style_scores: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Adjust occasion scores based on product metadata and style signals.

        This fixes misclassifications where casual items (crop tops, shorts)
        incorrectly score high for office/professional occasions.

        Args:
            occasion_scores: Raw CLIP-based occasion scores
            style_scores: Style scores (crop-tops, mini-lengths, etc.)
            metadata: Product metadata with 'name', 'brand', 'article_type', 'broad_category'

        Returns:
            Adjusted occasion scores
        """
        adjusted = occasion_scores.copy()

        name = (metadata.get('name') or '').lower()
        brand = (metadata.get('brand') or '').lower()
        article_type = (metadata.get('article_type') or '').lower()
        broad_category = (metadata.get('broad_category') or '').lower()

        # =================================================================
        # BRAND-BASED ADJUSTMENTS (Check first - brand is strong signal)
        # =================================================================

        # Athletic/Activewear brands - should NOT be office-appropriate
        athletic_brands = {
            'alo', 'alo yoga', 'lululemon', 'athleta', 'fabletics',
            'outdoor voices', 'beyond yoga', 'sweaty betty', 'gymshark',
            'nike', 'adidas', 'puma', 'under armour', 'reebok',
            'new balance', 'asics', 'brooks', 'champion', 'fila',
            'gap fit', 'old navy active', 'zella', 'vuori', 'rhone',
            'ten thousand', 'tracksmith', 'on running', 'hoka',
            'allbirds', 'nobull', 'rogue fitness', 'lorna jane',
            'carbon38', 'bandier', 'varley', 'splits59', 'year of ours',
            'girlfriend collective', 'girlfriend', 'oiselle', 'janji',
            'cotopaxi', 'patagonia', 'the north face', 'arc\'teryx',
            'salomon', 'columbia sportswear', 'marmot',
        }

        # Check if brand is athletic
        brand_is_athletic = False
        for athletic_brand in athletic_brands:
            if athletic_brand in brand:
                brand_is_athletic = True
                adjusted['active'] = max(adjusted['active'], 0.40)
                adjusted['office'] = min(adjusted['office'], 0.12)
                adjusted['smart-casual'] = min(adjusted['smart-casual'], 0.15)
                break

        # Professional/Office brands - boost office score
        professional_brands = {
            'theory', 'hugo boss', 'brooks brothers', 'ann taylor',
            'banana republic', 'j.crew', 'express', 'calvin klein',
            'ralph lauren', 'tommy hilfiger', 'club monaco', 'ted baker',
            'reiss', 'cos', 'massimo dutti', 'uniqlo', 'everlane',
            'aritzia', 'reformation', 'vince', 'equipment', 'frame',
            'eileen fisher', 'madewell', 'anthropologie', 'nordstrom',
        }

        # Only boost office if brand is explicitly professional (not athletic)
        if not brand_is_athletic:
            for prof_brand in professional_brands:
                if prof_brand in brand:
                    adjusted['office'] = max(adjusted['office'], 0.30)
                    adjusted['smart-casual'] = max(adjusted['smart-casual'], 0.28)
                    break

        # =================================================================
        # ARTICLE TYPE BASED ADJUSTMENTS
        # =================================================================

        # Items that should NEVER be office-appropriate
        non_office_types = {
            'crop top', 'crop', 'bralette', 'bikini', 'swimsuit', 'swimwear',
            'sports bra', 'athletic shorts', 'running shorts', 'yoga pants',
            'leggings', 'sweatpants', 'joggers', 'pajamas', 'sleepwear',
            'robe', 'lingerie', 'underwear', 'bodysuit', 'romper',
            'tube top', 'halter', 'bandeau',
        }

        # Items that are inherently office-appropriate
        office_types = {
            'blazer', 'suit', 'dress shirt', 'button-up', 'button-down',
            'blouse', 'dress pants', 'slacks', 'trousers', 'pencil skirt',
            'midi skirt', 'cardigan', 'sweater', 'turtleneck', 'polo',
            'sheath dress', 'wrap dress', 'shirt dress', 'a-line dress',
        }

        # Items that are casual-only
        casual_types = {
            'graphic tee', 'band tee', 'vintage tee', 'distressed',
            'ripped jeans', 'cutoff', 'denim shorts', 'beach cover',
            'flip flop', 'slides', 'sneakers', 'hoodie', 'sweatshirt',
        }

        # Items that are evening/events
        evening_types = {
            'cocktail', 'evening gown', 'formal', 'sequin', 'glitter',
            'party dress', 'maxi gown', 'ball gown', 'prom',
        }

        # Items that are active/athletic (expanded vocabulary)
        active_types = {
            # Core athletic terms
            'athletic', 'activewear', 'sports', 'workout', 'gym',
            'yoga', 'running', 'fitness', 'tennis', 'golf',
            'track', 'performance', 'moisture-wicking', 'training',
            # Athletic product types
            'legging', 'sports bra', 'tank', 'racerback', 'compression',
            'sweat-wicking', 'breathable mesh', 'stretch fabric',
            # Athleisure terms (often from athletic brands)
            'athleisure', 'studio', 'flow', 'flex', 'move',
            'sculpt', 'align', 'energy', 'power', 'boost',
            # Athletic brand product line names
            'accolade', 'airlift', 'airbrush', 'alosoft', 'alolux',
            'wunder', 'align', 'swiftly', 'define', 'scuba',
            'powersoft', 'breathe', 'supersonic',
            # Common athletic styling terms
            'half-zip', 'quarter-zip', 'pullover crew', 'jogger',
            'warm-up', 'cool-down', 'recovery', 'lounge set',
        }

        # Athletic brand product line prefixes (common in athleisure naming)
        athletic_product_lines = {
            'accolade', 'airlift', 'airbrush', 'alosoft', 'alolux',  # Alo Yoga
            'wunder', 'align', 'swiftly', 'define', 'scuba', 'groove',  # Lululemon
            'elation', 'transcend', 'momentum', 'salutation',  # Athleta
            'powersoft', 'breathe on', 'elevate',  # Old Navy Active
            'dri-fit', 'pro', 'therma',  # Nike
            'climalite', 'climacool', 'techfit',  # Adidas
            'heatgear', 'coldgear', 'rush',  # Under Armour
        }

        # Check article type and name against categories
        combined_text = f"{article_type} {name}"

        # Non-office items: cap office score at 0.15
        for term in non_office_types:
            if term in combined_text:
                adjusted['office'] = min(adjusted['office'], 0.15)
                adjusted['smart-casual'] = min(adjusted['smart-casual'], 0.18)
                break

        # Office items: boost office score
        for term in office_types:
            if term in combined_text:
                adjusted['office'] = max(adjusted['office'], 0.35)
                adjusted['smart-casual'] = max(adjusted['smart-casual'], 0.30)
                break

        # Casual items: boost casual, reduce office
        for term in casual_types:
            if term in combined_text:
                adjusted['casual'] = max(adjusted['casual'], 0.35)
                adjusted['office'] = min(adjusted['office'], 0.12)
                break

        # Evening items: boost evening/events
        for term in evening_types:
            if term in combined_text:
                adjusted['evening'] = max(adjusted['evening'], 0.40)
                adjusted['events'] = max(adjusted['events'], 0.40)
                break

        # Active items: boost active, reduce office
        for term in active_types:
            if term in combined_text:
                adjusted['active'] = max(adjusted['active'], 0.40)
                adjusted['office'] = min(adjusted['office'], 0.10)
                adjusted['smart-casual'] = min(adjusted['smart-casual'], 0.12)
                break

        # Also check athletic product line names (e.g., "Accolade" from Alo Yoga)
        for product_line in athletic_product_lines:
            if product_line in combined_text:
                adjusted['active'] = max(adjusted['active'], 0.35)
                adjusted['office'] = min(adjusted['office'], 0.12)
                adjusted['smart-casual'] = min(adjusted['smart-casual'], 0.15)
                break

        # If brand is athletic AND item has pullover/sweater/crew neck, it's athleisure, not office
        athleisure_keywords = {'pullover', 'crew neck', 'hoodie', 'sweatshirt', 'half zip', 'quarter zip', 'mock neck'}
        if brand_is_athletic:
            for keyword in athleisure_keywords:
                if keyword in combined_text:
                    adjusted['active'] = max(adjusted['active'], 0.38)
                    adjusted['casual'] = max(adjusted['casual'], 0.32)
                    adjusted['office'] = min(adjusted['office'], 0.10)
                    break

        # =================================================================
        # STYLE-BASED CROSS-PENALTIES
        # =================================================================

        # If item has high "crop-tops" style score, reduce office appropriateness
        crop_score = style_scores.get('crop-tops', 0)
        if crop_score > 0.25:
            penalty = min(0.15, (crop_score - 0.25) * 0.5)
            adjusted['office'] = max(0, adjusted['office'] - penalty)
            adjusted['smart-casual'] = max(0, adjusted['smart-casual'] - penalty * 0.5)

        # If item has high "mini-lengths" style score, reduce office appropriateness
        mini_score = style_scores.get('mini-lengths', 0)
        if mini_score > 0.28:
            penalty = min(0.12, (mini_score - 0.28) * 0.4)
            adjusted['office'] = max(0, adjusted['office'] - penalty)

        # If item has high "cutouts" style score, reduce office appropriateness
        cutout_score = style_scores.get('cutouts', 0)
        if cutout_score > 0.25:
            penalty = min(0.15, (cutout_score - 0.25) * 0.5)
            adjusted['office'] = max(0, adjusted['office'] - penalty)
            adjusted['smart-casual'] = max(0, adjusted['smart-casual'] - penalty * 0.5)

        # If item has high "deep-necklines" style score, reduce office appropriateness
        neckline_score = style_scores.get('deep-necklines', 0)
        if neckline_score > 0.28:
            penalty = min(0.12, (neckline_score - 0.28) * 0.4)
            adjusted['office'] = max(0, adjusted['office'] - penalty)

        # If item has high "sheer" style score, reduce office appropriateness
        sheer_score = style_scores.get('sheer', 0)
        if sheer_score > 0.25:
            penalty = min(0.15, (sheer_score - 0.25) * 0.5)
            adjusted['office'] = max(0, adjusted['office'] - penalty)

        # =================================================================
        # KEYWORD-BASED ADJUSTMENTS
        # =================================================================

        # Keywords that indicate casual wear
        casual_keywords = ['weekend', 'casual', 'lounge', 'cozy', 'comfy', 'relaxed']
        for keyword in casual_keywords:
            if keyword in name:
                adjusted['casual'] = max(adjusted['casual'], 0.30)
                adjusted['office'] = min(adjusted['office'], 0.18)
                break

        # Keywords that indicate work/professional
        # Note: Use word boundary check to avoid matching "workout" with "work"
        import re
        work_keywords = ['professional', 'office', 'business', 'career', 'workwear']
        for keyword in work_keywords:
            if keyword in name:
                adjusted['office'] = max(adjusted['office'], 0.35)
                adjusted['smart-casual'] = max(adjusted['smart-casual'], 0.30)
                break
        # Special handling for "work" - must be standalone word, not part of "workout"
        if re.search(r'\bwork\b', name) and 'workout' not in name:
            adjusted['office'] = max(adjusted['office'], 0.35)
            adjusted['smart-casual'] = max(adjusted['smart-casual'], 0.30)

        # Keywords that indicate party/evening
        party_keywords = ['party', 'night out', 'cocktail', 'date night', 'going out']
        for keyword in party_keywords:
            if keyword in name:
                adjusted['evening'] = max(adjusted['evening'], 0.35)
                adjusted['events'] = max(adjusted['events'], 0.30)
                break

        # Keywords that indicate beach/vacation
        beach_keywords = ['beach', 'vacation', 'resort', 'tropical', 'pool']
        for keyword in beach_keywords:
            if keyword in name:
                adjusted['beach'] = max(adjusted['beach'], 0.35)
                adjusted['casual'] = max(adjusted['casual'], 0.30)
                break
        # "summer" should boost beach but not override office
        if 'summer' in name and adjusted['beach'] < 0.30:
            adjusted['beach'] = max(adjusted['beach'], 0.25)

        # Ensure all scores are non-negative and round
        return {k: round(max(0, v), 4) for k, v in adjusted.items()}

    def compute_negative_occasion_scores(
        self,
        product_embedding: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute NEGATIVE occasion scores (how much an item does NOT fit each occasion).

        This is used for contrastive gating - an item passes the office gate if:
        - office_positive_score >= threshold AND
        - office_negative_score <= threshold

        Args:
            product_embedding: Product's FashionCLIP image embedding.
            metadata: Optional dict with 'name', 'brand', 'article_type', 'broad_category'

        Returns:
            Dict mapping occasion names to negative scores (0-1).
            Higher score = item is MORE like the "negative" concept (NOT appropriate for occasion).
        """
        self._ensure_model_loaded()

        # Convert embedding to normalized numpy array
        product_emb = self._to_normalized_embedding(product_embedding)

        negative_scores = {}
        for occasion_name, concept_embedding in self._occasion_negative_embeddings.items():
            similarity = float(np.dot(product_emb, concept_embedding))
            negative_scores[f"{occasion_name}_negative"] = round(similarity, 4)

        return negative_scores

    def compute_negative_occasion_scores_batch(
        self,
        product_embeddings: List[Any],
        batch_size: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Batch compute NEGATIVE occasion scores.

        Args:
            product_embeddings: List of product embeddings
            batch_size: Process in batches for memory efficiency

        Returns:
            List of negative score dicts, one per product
        """
        self._ensure_model_loaded()

        results = []

        for i in range(0, len(product_embeddings), batch_size):
            batch = product_embeddings[i:i + batch_size]

            # Convert batch to matrix
            emb_matrix = np.stack([
                self._to_normalized_embedding(emb) for emb in batch
            ])

            # Batch compute negative occasion scores
            if self._occasion_negative_embeddings:
                negative_matrix = np.stack(list(self._occasion_negative_embeddings.values()))
                negative_similarities = np.dot(emb_matrix, negative_matrix.T)
                negative_names = list(self._occasion_negative_embeddings.keys())

                for j in range(len(batch)):
                    neg_scores = {
                        f"{name}_negative": round(float(negative_similarities[j, k]), 4)
                        for k, name in enumerate(negative_names)
                    }
                    results.append(neg_scores)
            else:
                for _ in range(len(batch)):
                    results.append({})

        return results

    def classify_product_full(
        self,
        product_embedding: Any,
        include_patterns: bool = True,
        include_negative_scores: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Full product classification including negative occasion scores.

        This is an extended version of classify_product() that also returns
        negative occasion scores for hard gating.

        Args:
            product_embedding: Product's FashionCLIP image embedding.
            include_patterns: Whether to include pattern classification
            include_negative_scores: Whether to include negative occasion scores
            metadata: Optional dict with 'name', 'brand', 'article_type', 'broad_category'

        Returns:
            {
                'styles': {'sheer': 0.35, 'deep-necklines': 0.12, ...},
                'occasions': {'casual': 0.45, 'office': 0.32, 'office_negative': 0.28, ...},
                'patterns': {...}  # if include_patterns
            }
        """
        # Get base classification
        result = self.classify_product(product_embedding, include_patterns, metadata)

        # Add negative scores to occasions dict
        if include_negative_scores:
            negative_scores = self.compute_negative_occasion_scores(product_embedding, metadata)
            result['occasions'].update(negative_scores)

        return result

    def classify_products_batch_full(
        self,
        product_embeddings: List[Any],
        batch_size: int = 100,
        include_patterns: bool = True,
        include_negative_scores: bool = True,
        product_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Batch classify products including negative occasion scores.

        Args:
            product_embeddings: List of product embeddings
            batch_size: Process in batches for memory efficiency
            include_patterns: Whether to include pattern classification
            include_negative_scores: Whether to include negative occasion scores
            product_metadata: Optional list of metadata dicts

        Returns:
            List of classification results with negative scores included
        """
        # Get base classifications
        results = self.classify_products_batch(
            product_embeddings,
            batch_size=batch_size,
            include_patterns=include_patterns,
            product_metadata=product_metadata,
        )

        # Add negative scores if requested
        if include_negative_scores:
            negative_scores_list = self.compute_negative_occasion_scores_batch(
                product_embeddings,
                batch_size=batch_size,
            )

            for i, neg_scores in enumerate(negative_scores_list):
                if i < len(results):
                    results[i]['occasions'].update(neg_scores)

        return results

    def classify_attributes_batch(
        self,
        product_embeddings: List[Any],
        broad_categories: List[str],
        batch_size: int = 100,
    ) -> List[Dict[str, Optional[str]]]:
        """
        Classify multiple products' attributes efficiently.

        Args:
            product_embeddings: List of product embeddings
            broad_categories: List of broad categories for each product
            batch_size: Process in batches for memory efficiency

        Returns:
            List of attribute classifications, one per product
            Each dict contains: {'fit': str, 'length': str, 'sleeve': str, 'rise': str}
        """
        self._ensure_model_loaded()

        results = []

        for i in range(0, len(product_embeddings), batch_size):
            batch_embeddings = product_embeddings[i:i + batch_size]
            batch_categories = broad_categories[i:i + batch_size]

            # Convert batch to matrix
            emb_matrix = np.stack([
                self._to_normalized_embedding(emb) for emb in batch_embeddings
            ])

            # Pre-compute similarity matrices for all concept types
            fit_tops_matrix = np.stack(list(self._fit_embeddings_tops.values()))
            fit_bottoms_matrix = np.stack(list(self._fit_embeddings_bottoms.values()))
            fit_dresses_matrix = np.stack(list(self._fit_embeddings_dresses.values()))
            length_tops_matrix = np.stack(list(self._length_embeddings_tops.values()))
            length_bottoms_matrix = np.stack(list(self._length_embeddings_bottoms.values()))
            length_dresses_matrix = np.stack(list(self._length_embeddings_dresses.values()))
            sleeve_matrix = np.stack(list(self._sleeve_embeddings.values()))
            rise_matrix = np.stack(list(self._rise_embeddings.values()))

            fit_tops_sim = np.dot(emb_matrix, fit_tops_matrix.T)
            fit_bottoms_sim = np.dot(emb_matrix, fit_bottoms_matrix.T)
            fit_dresses_sim = np.dot(emb_matrix, fit_dresses_matrix.T)
            length_tops_sim = np.dot(emb_matrix, length_tops_matrix.T)
            length_bottoms_sim = np.dot(emb_matrix, length_bottoms_matrix.T)
            length_dresses_sim = np.dot(emb_matrix, length_dresses_matrix.T)
            sleeve_sim = np.dot(emb_matrix, sleeve_matrix.T)
            rise_sim = np.dot(emb_matrix, rise_matrix.T)

            fit_tops_names = list(self._fit_embeddings_tops.keys())
            fit_bottoms_names = list(self._fit_embeddings_bottoms.keys())
            fit_dresses_names = list(self._fit_embeddings_dresses.keys())
            length_tops_names = list(self._length_embeddings_tops.keys())
            length_bottoms_names = list(self._length_embeddings_bottoms.keys())
            length_dresses_names = list(self._length_embeddings_dresses.keys())
            sleeve_names = list(self._sleeve_embeddings.keys())
            rise_names = list(self._rise_embeddings.keys())

            for j in range(len(batch_embeddings)):
                category = (batch_categories[j] or '').lower()
                result = {'fit': None, 'length': None, 'sleeve': None, 'rise': None}

                # Determine fit
                if 'top' in category or 'knit' in category or 'woven' in category:
                    best_idx = np.argmax(fit_tops_sim[j])
                    result['fit'] = fit_tops_names[best_idx]
                elif 'bottom' in category or 'pant' in category or 'trouser' in category or 'jean' in category:
                    best_idx = np.argmax(fit_bottoms_sim[j])
                    result['fit'] = fit_bottoms_names[best_idx]
                elif 'dress' in category or 'skirt' in category:
                    best_idx = np.argmax(fit_dresses_sim[j])
                    result['fit'] = fit_dresses_names[best_idx]

                # Determine length
                if 'top' in category or 'knit' in category or 'woven' in category:
                    best_idx = np.argmax(length_tops_sim[j])
                    result['length'] = length_tops_names[best_idx]
                elif 'bottom' in category or 'pant' in category or 'trouser' in category or 'jean' in category:
                    best_idx = np.argmax(length_bottoms_sim[j])
                    result['length'] = length_bottoms_names[best_idx]
                elif 'dress' in category or 'skirt' in category:
                    best_idx = np.argmax(length_dresses_sim[j])
                    result['length'] = length_dresses_names[best_idx]

                # Determine sleeve (for applicable categories)
                if ('top' in category or 'dress' in category or 'outer' in category or
                    'knit' in category or 'woven' in category):
                    best_idx = np.argmax(sleeve_sim[j])
                    result['sleeve'] = sleeve_names[best_idx]

                # Determine rise (for bottoms only)
                if ('bottom' in category or 'pant' in category or 'trouser' in category or
                    'jean' in category or 'short' in category):
                    best_idx = np.argmax(rise_sim[j])
                    result['rise'] = rise_names[best_idx]

                results.append(result)

        return results

    def _to_normalized_embedding(self, embedding: Any) -> np.ndarray:
        """
        Convert various embedding formats to normalized numpy array.

        Handles:
        - List (from JSON/Supabase)
        - Numpy array
        - Torch tensor
        - String (pgvector format)
        """
        import torch

        # Convert to numpy array
        if isinstance(embedding, list):
            arr = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, str):
            # Handle pgvector string format: "[0.1,0.2,...]"
            cleaned = embedding.strip('[]')
            arr = np.array([float(x) for x in cleaned.split(',')], dtype=np.float32)
        elif isinstance(embedding, torch.Tensor):
            arr = embedding.detach().cpu().numpy().astype(np.float32)
        elif isinstance(embedding, np.ndarray):
            arr = embedding.astype(np.float32)
        else:
            raise ValueError(f"Unsupported embedding type: {type(embedding)}")

        # Ensure 1D
        arr = arr.flatten()

        # Normalize
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        return arr

    def get_tags_with_threshold(
        self,
        classification: Dict[str, Dict[str, float]],
        style_threshold: Optional[float] = None,
        occasion_threshold: Optional[float] = None,
        pattern_threshold: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """
        Convert scores to binary tags using thresholds.

        Useful for debugging/auditing, but the SQL functions
        use the raw scores directly for flexibility.

        Args:
            classification: Output from classify_product()
            style_threshold: Override default style threshold
            occasion_threshold: Override default occasion threshold
            pattern_threshold: Override default pattern threshold

        Returns:
            {
                'styles': ['sheer', 'deep-necklines'],  # styles above threshold
                'occasions': ['casual', 'office'],      # occasions above threshold
                'patterns': ['solid', 'stripes'],       # patterns above threshold (if present)
            }
        """
        style_thresh = style_threshold or self.config.default_style_threshold
        occasion_thresh = occasion_threshold or self.config.default_occasion_threshold
        pattern_thresh = pattern_threshold or self.config.default_pattern_threshold

        detected_styles = [
            style for style, score in classification['styles'].items()
            if score > style_thresh
        ]

        detected_occasions = [
            occasion for occasion, score in classification['occasions'].items()
            if score > occasion_thresh
        ]

        result = {
            'styles': detected_styles,
            'occasions': detected_occasions
        }

        # Include patterns if present in classification
        if 'patterns' in classification:
            detected_patterns = [
                pattern for pattern, score in classification['patterns'].items()
                if score > pattern_thresh
            ]
            result['patterns'] = detected_patterns

        return result


# =============================================================================
# Testing
# =============================================================================

def test_style_classifier():
    """Test the style classifier."""
    import json

    print("=" * 70)
    print("Testing Style Classifier")
    print("=" * 70)

    # Test 1: Initialize classifier
    print("\n1. Initializing classifier...")
    classifier = StyleClassifier()

    # Force model load
    classifier._ensure_model_loaded()

    print(f"   Style concepts: {list(classifier._style_embeddings.keys())}")
    print(f"   Occasion concepts: {list(classifier._occasion_embeddings.keys())}")
    print(f"   Pattern concepts: {list(classifier._pattern_embeddings.keys())}")
    print(f"   Fit concepts (tops): {list(classifier._fit_embeddings_tops.keys())}")
    print(f"   Fit concepts (bottoms): {list(classifier._fit_embeddings_bottoms.keys())}")
    print(f"   Length concepts (dresses): {list(classifier._length_embeddings_dresses.keys())}")
    print(f"   Sleeve concepts: {list(classifier._sleeve_embeddings.keys())}")

    # Test 2: Create a mock embedding (random for testing)
    print("\n2. Testing with random embedding (no metadata)...")
    mock_embedding = np.random.randn(512).astype(np.float32)

    result = classifier.classify_product(mock_embedding, include_patterns=True)
    print(f"   Style scores: {json.dumps(result['styles'], indent=2)}")
    print(f"   Occasion scores: {json.dumps(result['occasions'], indent=2)}")
    print(f"   Pattern scores: {json.dumps(result['patterns'], indent=2)}")

    # Test 3: Test threshold-based tagging
    print("\n3. Testing threshold-based tagging...")
    tags = classifier.get_tags_with_threshold(result)
    print(f"   Detected styles (threshold {classifier.config.default_style_threshold}): {tags['styles']}")
    print(f"   Detected occasions (threshold {classifier.config.default_occasion_threshold}): {tags['occasions']}")
    print(f"   Detected patterns (threshold {classifier.config.default_pattern_threshold}): {tags.get('patterns', [])}")

    # Test 4: Test metadata-based occasion adjustments (NEW)
    print("\n4. Testing metadata-based occasion adjustments...")

    # Test case: Crop top should NOT be office appropriate
    print("\n   4a. Crop top (should have LOW office score):")
    result_crop = classifier.classify_product(mock_embedding, include_patterns=False, metadata={
        'name': 'Sexy Crop Top',
        'article_type': 'Crop Top',
        'broad_category': 'tops',
    })
    print(f"       Raw office score (no metadata): {result['occasions']['office']:.4f}")
    print(f"       Adjusted office score: {result_crop['occasions']['office']:.4f}")
    print(f"       Adjusted casual score: {result_crop['occasions']['casual']:.4f}")

    # Test case: Blazer should be office appropriate
    print("\n   4b. Blazer (should have HIGH office score):")
    result_blazer = classifier.classify_product(mock_embedding, include_patterns=False, metadata={
        'name': 'Professional Wool Blazer',
        'article_type': 'Blazer',
        'broad_category': 'outerwear',
    })
    print(f"       Raw office score (no metadata): {result['occasions']['office']:.4f}")
    print(f"       Adjusted office score: {result_blazer['occasions']['office']:.4f}")
    print(f"       Adjusted smart-casual: {result_blazer['occasions']['smart-casual']:.4f}")

    # Test case: Athletic wear should be active, not office
    print("\n   4c. Athletic leggings (should have HIGH active, LOW office):")
    result_active = classifier.classify_product(mock_embedding, include_patterns=False, metadata={
        'name': 'High-Waist Yoga Leggings',
        'article_type': 'Leggings',
        'broad_category': 'bottoms',
    })
    print(f"       Adjusted active score: {result_active['occasions']['active']:.4f}")
    print(f"       Adjusted office score: {result_active['occasions']['office']:.4f}")

    # Test case: Alo Yoga Accolade Crew Neck Pullover (SPECIFIC FIX)
    print("\n   4c2. Alo Yoga Accolade Pullover (should be ACTIVE, NOT office):")
    result_alo = classifier.classify_product(mock_embedding, include_patterns=False, metadata={
        'name': 'Accolade Crew Neck Pullover',
        'brand': 'Alo Yoga',
        'article_type': 'Pullover',
        'broad_category': 'tops',
    })
    print(f"       Brand detected: Alo Yoga (athletic brand)")
    print(f"       Adjusted active score: {result_alo['occasions']['active']:.4f}")
    print(f"       Adjusted casual score: {result_alo['occasions']['casual']:.4f}")
    print(f"       Adjusted office score: {result_alo['occasions']['office']:.4f}")
    assert result_alo['occasions']['office'] <= 0.12, "Alo Yoga pullover should NOT be office!"
    assert result_alo['occasions']['active'] >= 0.35, "Alo Yoga pullover should be active!"
    print(f"       ✓ PASS: Office capped at {result_alo['occasions']['office']:.2f}, Active boosted to {result_alo['occasions']['active']:.2f}")

    # Test case: Beach wear
    print("\n   4d. Beach cover-up (should have HIGH beach/casual):")
    result_beach = classifier.classify_product(mock_embedding, include_patterns=False, metadata={
        'name': 'Tropical Beach Cover Up',
        'article_type': 'Cover Up',
        'broad_category': 'tops',
    })
    print(f"       Adjusted beach score: {result_beach['occasions']['beach']:.4f}")
    print(f"       Adjusted casual score: {result_beach['occasions']['casual']:.4f}")
    print(f"       Adjusted office score: {result_beach['occasions']['office']:.4f}")

    # Test 5: Test batch classification with metadata
    print("\n5. Testing batch classification with metadata...")
    batch_embeddings = [np.random.randn(512).astype(np.float32) for _ in range(5)]
    batch_metadata = [
        {'name': 'Crop Top', 'article_type': 'Crop Top', 'broad_category': 'tops'},
        {'name': 'Dress Shirt', 'article_type': 'Shirt', 'broad_category': 'tops'},
        {'name': 'Yoga Pants', 'article_type': 'Leggings', 'broad_category': 'bottoms'},
        {'name': 'Cocktail Dress', 'article_type': 'Dress', 'broad_category': 'dresses'},
        {'name': 'Beach Shorts', 'article_type': 'Shorts', 'broad_category': 'bottoms'},
    ]
    batch_results = classifier.classify_products_batch(
        batch_embeddings,
        include_patterns=False,
        product_metadata=batch_metadata
    )
    print(f"   Processed {len(batch_results)} products")
    for i, res in enumerate(batch_results):
        print(f"   {batch_metadata[i]['name']:15} | office: {res['occasions']['office']:.2f} | casual: {res['occasions']['casual']:.2f} | active: {res['occasions']['active']:.2f}")

    # Test 6: Test attribute classification
    print("\n6. Testing attribute classification...")
    attributes_top = classifier.classify_attributes(mock_embedding, 'tops')
    attributes_dress = classifier.classify_attributes(mock_embedding, 'dresses')
    attributes_bottom = classifier.classify_attributes(mock_embedding, 'bottoms')

    print(f"   Top attributes: {attributes_top}")
    print(f"   Dress attributes: {attributes_dress}")
    print(f"   Bottom attributes: {attributes_bottom}")

    # Test 7: Test batch attribute classification
    print("\n7. Testing batch attribute classification...")
    categories = ['tops', 'bottoms', 'dresses', 'tops', 'bottoms']
    batch_attrs = classifier.classify_attributes_batch(batch_embeddings, categories)
    print(f"   Processed {len(batch_attrs)} products")
    for i, attrs in enumerate(batch_attrs):
        print(f"   {categories[i]}: {attrs}")

    # Test 8: Test various input formats
    print("\n8. Testing input format handling...")

    # List format (from JSON/Supabase)
    list_emb = mock_embedding.tolist()
    result_list = classifier.classify_product(list_emb)
    print(f"   List input: OK")

    # String format (pgvector)
    str_emb = f"[{','.join(map(str, mock_embedding[:10]))}]"  # Truncated for test
    # Note: This would fail with full embedding due to dimension mismatch, just testing parsing

    print("\n" + "=" * 70)
    print("Style Classifier test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_style_classifier()
