# OutfitTransformer - Fashion Recommendation System

## Project Overview

A production-ready multimodal fashion recommendation system combining visual compatibility, sequential modeling, and personalized ranking.

**Core Technologies:**
- **Visual**: FashionCLIP (512-dim embeddings) for outfit compatibility
- **Sequential**: BERT4Rec/SASRec for user behavior modeling
- **Compatibility**: OutfitTransformer (4-layer Transformer) for FITB task

---

## Current Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **FITB Accuracy** | **65.31%** | >60% | ✅ PASS |
| Compatibility AUC | 0.72 | >0.85 | ❌ FAIL |
| Random Baseline | 25% | - | - |

**Best Model**: OutfitTransformer @ Epoch 70 (`models/outfit_transformer_best.pth`)
**Key Insight**: Test items have 0% overlap with training → model learns visual compatibility patterns, not memorization

---

## Project Structure

```
outfitTransformer/
├── src/
│   ├── recs/                              # Main Recommendation API (Supabase)
│   │   ├── api_endpoints.py               # FastAPI routes (/api/recs/v2/*)
│   │   ├── candidate_selection.py         # Candidate retrieval from pgvector
│   │   ├── models.py                      # Pydantic models
│   │   ├── pipeline.py                    # Main recommendation pipeline
│   │   ├── recommendation_service.py      # Service layer
│   │   ├── sasrec_ranker.py               # SASRec model ranking
│   │   └── session_state.py               # Session management
│   │
│   ├── engines/                           # UI Engines for style discovery
│   │   ├── swipe_engine.py                # Base Tinder-style engine
│   │   ├── four_choice_engine.py          # Four-choice selection
│   │   ├── ranking_engine.py              # Drag-to-rank interaction
│   │   ├── attribute_test_engine.py       # Attribute preference testing
│   │   └── predictive_four_engine.py      # Predictive four-choice
│   │
│   ├── swipe_server.py                    # Main FastAPI server (women's fashion)
│   ├── api.py                             # Legacy Polyvore API
│   ├── outfit_transformer.py              # OutfitTransformer model (65.31% FITB)
│   ├── embeddings.py                      # FashionCLIP embeddings + Faiss
│   ├── feed_generator.py                  # Hybrid recommendation engine
│   ├── outrove_filter.py                  # Onboarding preference filtering
│   ├── women_search_engine.py             # Women's fashion search (Supabase)
│   └── gender_config.py                   # Gender-specific configuration
│
├── sql/                                   # Database migrations (16 files)
├── tests/                                 # Pytest test suite
├── docs/                                  # API documentation
├── config/                                # Configuration files
├── models/                                # Trained models (not in git)
└── data/                                  # Datasets (not in git)
```

---

## Model Architectures

### 1. OutfitTransformer (`src/outfit_transformer.py`)

**Purpose**: Fill-In-The-Blank (FITB) task for outfit completion

**Architecture:**
```
Input: Visual features (FashionCLIP 512d) + Category embeddings (128d)
  ↓
MultimodalItemEncoder: Fusion layer (512d hidden)
  ↓
[OUTFIT_TOKEN] + Item Embeddings (NO positional encoding)
  ↓
Transformer Encoder: 4 layers, 8 heads, 512 hidden dim
  ↓
Outputs:
  - compatibility_head: Binary outfit compatibility (0-1)
  - item_prediction_head: Masked item prediction (vocab size)
```

**Training:**
- **Loss 1**: Masked item prediction (cross-entropy)
- **Loss 2**: Compatibility scoring (BCE, positive=1, negative=0)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR
- **Best Result**: Epoch 70, FITB 65.31%

**Key Design Choices:**
- No positional encoding → treats outfits as sets, not sequences
- Dual loss function → compatibility head critical for FITB on unseen items
- Random negative sampling → corrupts one item per outfit for contrastive learning

**Checkpoint:** `models/outfit_transformer_best.pth` (1.4 GB)

---

### 2. BERT4Rec (`src/train_models.py`)

**Purpose**: Sequential next-item prediction from user history

**Architecture:**
- 4 Transformer layers, 8 heads
- 256 hidden dim, 50 max sequence length
- Masked item prediction (BERT-style)

**Dataset**: Polyvore-U (user-item interaction sequences)

**Checkpoint**: `models/BERT4Rec-Dec-08-2025_03-24-36.pth` (101 MB)

**Note**: BERT4Rec performs at random baseline on Polyvore-U FITB → sequential models don't capture visual compatibility

---

### 3. SASRec (`src/train_amazon_sasrec.py`)

**Purpose**: Sequential recommendation for Amazon Fashion dataset

**Architecture:**
- Self-attention based sequential model
- Left-to-right unidirectional (vs BERT4Rec bidirectional)
- 262 MB model trained on 304K users, 59K items

**Use Case**: Ranking candidates in `/outrove/feed` endpoint after preference filtering

**Checkpoint**: `models/SASRec-Dec-11-2025_18-20-35.pth` (262 MB)

---

### 4. FashionCLIP Embeddings (`src/embeddings.py`)

**Model**: `patrickjohncyh/fashion-clip`

**Features:**
- 512-dimensional visual embeddings
- Faiss GPU indexing for fast similarity search
- 142,480 Polyvore items embedded

**Files:**
- `models/polyvore_embeddings.pkl` (224 MB)
- `models/polyvore_faiss_index.bin` (279 MB)

**Usage:**
```python
from embeddings import FaissSearcher

searcher = FaissSearcher('models/polyvore_embeddings.pkl', use_gpu=True)
results = searcher.search(item_id="194508109", k=10)
```

---

## Datasets

### Polyvore (`data/polyvore/`)

**Source**: [Polyvore Dataset](https://github.com/xthan/polyvore-dataset)

**Files:**
- `train_no_dup.json` - 17,316 training outfits (44 MB)
- `test_no_dup.json` - 3,076 test outfits (7.2 MB)
- `valid_no_dup.json` - 1,407 validation outfits (3.5 MB)
- `fill_in_blank_test.json` - 3,076 FITB questions (1.1 MB)
- `polyvore_item_metadata.json` - 142,480 items with categories (31 MB)
- `image_to_tid_mapping.json` - Image URL → item ID mapping (3.8 MB)

**Statistics:**
- 142,480 unique items
- 380 categories
- 21,799 outfits total
- Average outfit size: 4-6 items

---

### Polyvore-U (`data/polyvore_u/`)

**Source**: Sequential user-item interaction data

**Format**: User ID → Item sequence → Timestamp

**RecBole Format**: `data/polyvore_u_recbole/polyvore_u/`
- `.inter` file for training BERT4Rec/SASRec

**Note**: Different from Polyvore - focuses on user browsing/purchase sequences

---

### Amazon Fashion (`data/amazon_fashion/`)

**Source**: Amazon Fashion review dataset

**Statistics:**
- 59,413 items
- 304,422 users
- Men's clothing focused

**Processed Data:**
- `processed/tops_enriched.pkl` (10 MB) - 3,666 tops with rich attributes
  - t-shirts: 2,114 items
  - shirts: 815 items
  - hoodies: 297 items
  - polos: 254 items
  - sweaters: 125 items
  - henleys: 61 items

**Attribute Coverage:**
- Colors: 39% (extracted from titles/descriptions)
- Materials: 47% (cotton, polyester, wool, etc.)
- Price: 93% (median $25-35)
- Visual attributes: 100% (FashionCLIP)

**RecBole Format**: `data/amazon_fashion/recbole/amazon_mens/`

**Embeddings**: `processed/amazon_mens_embeddings.pkl` (119 MB)

---

## Training Commands

### Train OutfitTransformer

```bash
cd /home/ubuntu/recSys/outfitTransformer

python src/outfit_transformer.py \
  --data_dir data/polyvore \
  --embeddings models/polyvore_embeddings.pkl \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-4 \
  --hidden_dim 512 \
  --n_layers 4 \
  --n_heads 8 \
  --compat_weight 1.0 \
  --save_dir models
```

**Training Time**: ~2 hours on single GPU (NVIDIA T4/V100)
**GPU Memory**: ~9 GB

**Critical Parameters:**
- `--compat_weight 1.0` - MUST be >0 to train compatibility_head (used in FITB)
- Without compatibility loss, FITB accuracy drops to random baseline

---

### Train BERT4Rec

```bash
python src/train_models.py BERT4Rec data/polyvore_u_recbole polyvore_u
```

**Config** (`configs/BERT4Rec.yaml`):
```yaml
n_layers: 4
n_heads: 8
hidden_size: 256
MAX_ITEM_LIST_LENGTH: 50
train_batch_size: 256
```

**Training Time**: ~3-4 hours
**Convergence**: Epoch 100-120

---

### Train SASRec (Amazon Fashion)

```bash
python src/train_amazon_sasrec.py
```

**Dataset**: Amazon Fashion mens (304K users, 59K items)
**Training Time**: ~8-10 hours on GPU

---

### Generate Embeddings

```bash
python src/embeddings.py data/polyvore data/polyvore/images
```

**Output:**
- `models/polyvore_embeddings.pkl` - FashionCLIP embeddings
- `models/polyvore_faiss_index.bin` - Faiss GPU index

**Processing Time**: ~30 mins for 142K items

---

## Evaluation Commands

### FITB on Polyvore

```bash
python src/evaluate.py data/polyvore models/polyvore_embeddings.pkl
```

**Output:**
```
Compatibility AUC: 0.7215
FITB Accuracy: 0.01  # Low because using simple visual similarity
recall@10: 0.0074
ndcg@10: 0.0090
```

### FITB with OutfitTransformer

```bash
# Run during training (automatic every 5 epochs)
# Or load checkpoint and evaluate:

python -c "
from outfit_transformer import OutfitTransformer, FITBDataset, evaluate_fitb
import torch

device = torch.device('cuda')
checkpoint = torch.load('models/outfit_transformer_best.pth')
config = checkpoint['config']

model = OutfitTransformer(**config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

fitb_dataset = FITBDataset(
    'data/polyvore/fill_in_blank_test.json',
    'data/polyvore/image_to_tid_mapping.json',
    'models/polyvore_embeddings.pkl',
    'data/polyvore/polyvore_item_metadata.json'
)

accuracy = evaluate_fitb(model, fitb_dataset, device)
print(f'FITB Accuracy: {accuracy*100:.2f}%')
"
```

**Expected Output**: 65.31%

---

### FITB on Polyvore-U with BERT4Rec

```bash
python src/evaluate.py polyvore_u \
  data/polyvore_u \
  models/polyvore_embeddings.pkl \
  models/BERT4Rec-Dec-08-2025_03-24-36.pth
```

**Result**: ~20% (random baseline) - Sequential model doesn't capture visual compatibility

---

## API Server

### Start Server

```bash
cd /home/ubuntu/recSys/outfitTransformer/src
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## API Documentation

All endpoints support **pagination** with consistent response format:
```json
{
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 142480,
    "total_pages": 7124,
    "has_next": true,
    "has_prev": false
  }
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "items_count": 142480,
  "categories_count": 380,
  "embeddings_loaded": true,
  "faiss_index_size": 142480
}
```

---

### Outfit Compatibility Score

Score how well items go together (0-1 scale).

```bash
curl -X POST http://localhost:8000/compatibility \
  -H "Content-Type: application/json" \
  -d '{"item_ids": ["194508109", "188778349", "188977857"]}'
```

**Response:**
```json
{
  "item_ids": ["194508109", "188778349", "188977857"],
  "compatibility_score": 0.98,
  "item_count": 3
}
```

---

### Fill-In-The-Blank (FITB)

Rank candidate items by how well they complete an outfit.

```bash
curl -X POST http://localhost:8000/fitb \
  -H "Content-Type: application/json" \
  -d '{
    "context_items": ["194508109", "188778349"],
    "candidate_items": ["188977857", "194942557", "194941874"]
  }'
```

**Response:**
```json
{
  "context_items": ["194508109", "188778349"],
  "ranked_candidates": [
    {"item_id": "194942557", "score": 0.96},
    {"item_id": "188977857", "score": 0.92},
    {"item_id": "194941874", "score": 0.85}
  ],
  "best_match": "194942557"
}
```

---

### Visual Similarity (Paginated)

Find visually similar items using FashionCLIP + Faiss.

```bash
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"item_id": "194508109", "page": 1, "page_size": 10}'
```

**Response:**
```json
{
  "item_id": "194508109",
  "similar": [
    {"item_id": "199234567", "similarity": 0.94, "category": "Sweatshirts"},
    {"item_id": "189876543", "similarity": 0.91, "category": "T-Shirts"}
  ],
  "pagination": {"page": 1, "page_size": 10, "total_items": 1000, ...}
}
```

---

### Style This Item (Paginated)

Generate multiple complete outfit sets featuring a single anchor product.

```bash
curl -X POST http://localhost:8000/style-this-item \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "194508109",
    "page": 1,
    "page_size": 3,
    "items_per_outfit": 4
  }'
```

**Response:**
```json
{
  "anchor_item": {
    "item_id": "194508109",
    "category": "Sweatshirts",
    "name": "Nike Sportswear Club Fleece"
  },
  "outfits": [
    {
      "items": [
        {"item_id": "194508109", "category": "Sweatshirts", "is_anchor": true},
        {"item_id": "199234567", "category": "Jeans", "is_anchor": false},
        {"item_id": "189876543", "category": "Sneakers", "is_anchor": false},
        {"item_id": "177654321", "category": "Backpacks", "is_anchor": false}
      ],
      "compatibility_score": 0.98,
      "style_description": "Casual Streetwear"
    }
  ],
  "pagination": {"page": 1, "page_size": 3, "total_items": 6, ...}
}
```

**Use Case**: User views a t-shirt → show them multiple outfit ideas

---

### Personalized Feed (Paginated)

Get personalized recommendations combining visual similarity + collaborative filtering.

```bash
curl -X POST http://localhost:8000/feed \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "page": 1, "page_size": 20}'
```

**Response:**
```json
{
  "user_id": "user123",
  "items": [...],
  "ranking_method": "hybrid",
  "pagination": {...}
}
```

---

## Outrove API (Amazon Fashion + Onboarding)

Two-stage recommendation pipeline:
1. **Stage 1**: Strict filtering based on user onboarding preferences
2. **Stage 2**: SASRec sequential ranking

**Data**: 3,666 tops with pre-computed attributes.

### Outrove Health

```bash
curl http://localhost:8000/outrove/health
```

**Response:**
```json
{
  "status": "healthy",
  "tops_items_count": 3666,
  "categories": {
    "t-shirts": 2114,
    "hoodies": 297,
    "shirts": 815,
    "polos": 254,
    "sweaters": 125,
    "henleys": 61
  },
  "sasrec_loaded": true
}
```

---

### Outrove Available Options

Get all filter options for onboarding UI.

```bash
curl http://localhost:8000/outrove/options
```

**Response:**
```json
{
  "categories": ["t-shirts", "hoodies", "shirts", "polos", "sweaters", "henleys"],
  "colors": ["black", "white", "gray", "blue", "red", "green", "navy", ...],
  "materials": ["cotton", "polyester", "wool", "linen", "silk", ...],
  "fits": ["slim", "regular", "relaxed", "athletic"],
  "necklines": ["crew", "v-neck", "henley", "polo", "mock", "turtle"],
  "graphics_tolerance": ["none", "minimal", "moderate", "any"],
  "price_ranges": {
    "t-shirts": [10, 50],
    "hoodies": [40, 120],
    "shirts": [30, 100]
  }
}
```

---

### Outrove Personalized Feed

Generate feed based on onboarding profile + SASRec ranking.

```bash
curl -X POST http://localhost:8000/outrove/feed \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "AVU1ILDDYW301",
    "selectedCoreTypes": ["t-shirts", "hoodies"],
    "colorsToAvoid": ["pink", "yellow"],
    "materialsToAvoid": ["polyester"],
    "tshirts": {
      "fit": "regular",
      "necklines": ["crew", "v-neck"],
      "graphicsTolerance": "minimal",
      "priceRange": [10, 50]
    },
    "page": 1,
    "page_size": 20
  }'
```

**Response:**
```json
{
  "user_id": "AVU1ILDDYW301",
  "items": [
    {
      "item_id": "B01FN87DE4",
      "score": 4.355,
      "category": "t-shirts",
      "title": "Hanes Men's Short Sleeve Graphic T-Shirt",
      "brand": "Hanes",
      "price": 19.99,
      "colors": ["gray", "blue"],
      "materials": ["cotton"],
      "fit": "regular",
      "image_url": "https://..."
    }
  ],
  "filter_stats": {
    "candidates_before_filter": 2411,
    "candidates_after_filter": 1295,
    "filter_rate_percent": 46.3,
    "by_category": {"t-shirts": 1196, "hoodies": 99}
  },
  "ranking_source": "sasrec",
  "pagination": {...}
}
```

---

## Dependencies

```
# Core ML/DL
recbole==1.2.0
torch>=1.10.0
torchvision

# FashionCLIP
fashion-clip

# Vector Search
faiss-gpu

# API
fastapi
uvicorn[standard]

# Data Processing
pandas
numpy
pillow
tqdm
requests

# Testing
pytest
pytest-asyncio
httpx

# Utilities
pyyaml
scikit-learn
```

Install:
```bash
pip install -r requirements.txt
```

---

## File Sizes

| File | Size |
|------|------|
| `outfit_transformer_best.pth` | 1.4 GB |
| `polyvore_embeddings.pkl` | 224 MB |
| `polyvore_faiss_index.bin` | 279 MB |
| `BERT4Rec-Dec-08-2025_03-24-36.pth` | 101 MB |
| `SASRec-Dec-11-2025_18-20-35.pth` | 262 MB |
| `amazon_mens_embeddings.pkl` | 119 MB |
| `tops_enriched.pkl` | 10 MB |

**Total Model Storage**: ~2.5 GB
**Total Data Storage**: ~13 GB (including images)

---

## Training Notes

### OutfitTransformer

**Critical Requirements:**
- MUST train with both masked prediction AND compatibility scoring
- `compat_weight` > 0 required to train `compatibility_head` (used in FITB)
- Without compatibility loss, FITB accuracy = random baseline

**Why Dual Loss?**
1. **Masked prediction** trains `item_prediction_head` → learns item relationships
2. **Compatibility scoring** trains `compatibility_head` → learns outfit coherence

FITB evaluation uses `compatibility_head` to score visual features → works on unseen items.

**Training Curve:**
- Epoch 1-10: Rapid improvement (25% → 45%)
- Epoch 10-30: Steady climb (45% → 60%)
- Epoch 30-70: Fine-tuning (60% → 65.31%)
- Epoch 70+: Plateau/slight overfitting

**Best Epoch**: 70 (65.31% FITB)

---

### BERT4Rec

- Converges around epoch 100-120
- GPU memory: ~9 GB on 22GB GPU
- Sequential modeling helps with next-item prediction
- Does NOT help with FITB (visual compatibility task)

---

### SASRec (Amazon Fashion)

- Training time: ~8-10 hours on V100
- 304K users, 59K items
- Used for ranking in Outrove onboarding pipeline

---

## Known Issues

1. **Compatibility AUC** (0.72 vs target 0.85)
   - Current pairwise similarity approach may need improvement
   - Consider contrastive learning with harder negatives

2. **BERT4Rec FITB** performs at random baseline
   - Sequential model doesn't capture visual compatibility
   - Need visual + sequential hybrid approach

3. **Memory usage**
   - OutfitTransformer training: ~9 GB GPU
   - May need gradient checkpointing for larger models

---

## Future Improvements

### Short Term
1. **Improve Compatibility AUC** (0.72 → 0.85+)
   - Use contrastive learning (SimCLR/MoCo style)
   - Category-aware negative sampling
   - Triplet loss with hard negatives

2. **Hybrid FITB Model**
   - Combine OutfitTransformer (visual) + BERT4Rec (sequential)
   - Weighted ensemble or late fusion

3. **Data Augmentation**
   - Color jittering, cropping for training
   - Synthetic outfit generation

### Long Term
1. **Text Integration**
   - Add item descriptions/titles to multimodal encoder
   - Use CLIP text encoder

2. **Style Classification**
   - Classify outfits into style categories (casual, formal, streetwear)
   - Style-aware recommendation

3. **Real-time Inference**
   - Model quantization (FP16/INT8)
   - ONNX export for faster serving
   - Batch inference optimization

4. **User Personalization**
   - User embedding layer in OutfitTransformer
   - Meta-learning for cold-start users

---

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test module:
```bash
pytest tests/test_api.py -v
```

Skip slow tests:
```bash
pytest tests/ -v -m "not slow"
```

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/polyvore/polyvore_item_metadata.json data/polyvore/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:
```bash
docker build -t outfit-transformer .
docker run -p 8000:8000 -v $(pwd)/models:/app/models outfit-transformer
```

---

### Environment Variables

```bash
# API
export HOST=0.0.0.0
export PORT=8000
export WORKERS=4

# Models
export EMBEDDINGS_PATH=models/polyvore_embeddings.pkl
export FAISS_PATH=models/polyvore_faiss_index.bin
export OUTFIT_TRANSFORMER_PATH=models/outfit_transformer_best.pth

# GPU
export USE_GPU=true
export CUDA_VISIBLE_DEVICES=0
```

---

## Performance Benchmarks

### Latency (P95)

| Endpoint | Latency | Notes |
|----------|---------|-------|
| `/health` | <10ms | No computation |
| `/similar` | ~30ms | Faiss search |
| `/compatibility` | ~20ms | Pairwise similarity |
| `/fitb` | ~50ms | Multiple forward passes |
| `/style-this-item` | ~100ms | Generate multiple outfits |
| `/feed` | ~80ms | Hybrid ranking |

### Throughput

- Single GPU: ~100 req/s
- 4 workers: ~400 req/s

---

## Citation

If you use this code or models, please cite:

**OutfitTransformer**: Based on Amazon's OutfitTransformer and Alibaba's POG

**FashionCLIP**: `patrickjohncyh/fashion-clip`

**RecBole**: RecBole library for BERT4Rec/SASRec

**Polyvore Dataset**: Original Polyvore dataset

---

## License

MIT
