# OutfitTransformer - Fashion Recommendation System

## Project Overview

A production fashion recommendation system with style preference learning and personalized recommendations.

**Core Features:**
- **Style Learning**: Tinder-style 4-choice interface to learn user preferences
- **Personalized Feed**: Recommendations based on learned taste vectors
- **Supabase Integration**: pgvector for similarity search, persistent storage

---

## Project Structure

```
outfitTransformer/
├── src/
│   ├── swipe_server.py                    # Main FastAPI server
│   ├── recs/                              # Recommendation API (Supabase)
│   │   ├── api_endpoints.py               # FastAPI routes (/api/recs/v2/*)
│   │   ├── candidate_selection.py         # Candidate retrieval from pgvector
│   │   ├── candidate_factory.py           # Candidate generation
│   │   ├── models.py                      # Pydantic models
│   │   ├── pipeline.py                    # Main recommendation pipeline
│   │   ├── recommendation_service.py      # Service layer
│   │   ├── sasrec_ranker.py               # SASRec model ranking
│   │   ├── session_state.py               # Session management
│   │   ├── style_classifier.py            # Style classification
│   │   ├── occasion_gate.py               # Occasion filtering
│   │   └── filter_utils.py                # Filter utilities
│   │
│   ├── engines/                           # UI Engines for style discovery
│   │   ├── swipe_engine.py                # Base Tinder-style engine
│   │   ├── four_choice_engine.py          # Four-choice selection
│   │   ├── ranking_engine.py              # Drag-to-rank interaction
│   │   ├── attribute_test_engine.py       # Attribute preference testing
│   │   └── predictive_four_engine.py      # Predictive four-choice
│   │
│   ├── women_search_engine.py             # Women's fashion search (Supabase)
│   ├── outrove_filter.py                  # Onboarding preference filtering
│   ├── gender_config.py                   # Gender-specific configuration
│   └── tests/                             # Test suite
│
├── sql/                                   # Database migrations
├── tests/                                 # Pytest test suite
├── docs/                                  # API documentation
└── config/                                # Configuration files
```

---

## API Server

### Start Server

```bash
cd /home/ubuntu/recSys/outfitTransformer/src
python3 -m uvicorn swipe_server:app --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn swipe_server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## API Endpoints

### Women's Fashion Style Learning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/women/options` | GET | Get available categories and attributes |
| `/api/women/session/start` | POST | Start a new style learning session |
| `/api/women/session/choose` | POST | Record user's choice (1 of 4) |
| `/api/women/session/skip` | POST | Skip all 4 items |
| `/api/women/session/{user_id}/summary` | GET | Get learned preferences |
| `/api/women/feed/{user_id}` | GET | Get personalized feed |
| `/api/women/search` | POST | Search women's fashion |

### Recommendation Pipeline (v2)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recs/v2/onboarding` | POST | Save 9-module onboarding profile |
| `/api/recs/v2/feed` | GET | Get feed using full pipeline |
| `/api/recs/v2/info` | GET | Get pipeline configuration info |
| `/api/recs/v2/action` | POST | Record user interaction |

### Legacy Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recs/save-preferences` | POST | Save Tinder test results |
| `/api/recs/feed/{user_id}` | GET | Get personalized feed |
| `/api/recs/similar/{product_id}` | GET | Get similar products |
| `/api/recs/trending` | GET | Get trending products |
| `/api/recs/categories` | GET | Get product categories |

---

## Style Learning Flow

1. **Start Session** - Initialize with optional category/color preferences
2. **Choose Items** - User picks favorite from 4 items shown
3. **System Learns** - Bayesian preference learning on attributes
4. **Get Feed** - Personalized recommendations based on learned taste

### Attribute Dimensions Learned

| Attribute | Example Values |
|-----------|---------------|
| pattern | solid, striped, floral, geometric |
| style | casual, office, evening, bohemian |
| color_family | neutral, bright, cool, pastel, dark |
| fit_vibe | fitted, relaxed, oversized, cropped |
| neckline | crew, v_neck, off_shoulder, sweetheart |
| occasion | everyday, work, date_night, party |
| sleeve_type | sleeveless, short, long, puff |

### Women's Fashion Categories

- `tops_knitwear` - Sweaters, cardigans, knit tops
- `tops_woven` - Blouses, shirts, woven tops
- `tops_sleeveless` - Tank tops, camisoles
- `tops_special` - Bodysuits, special tops
- `dresses` - All dress types
- `bottoms_trousers` - Pants, jeans, trousers
- `bottoms_skorts` - Skirts, shorts
- `outerwear` - Jackets, coats
- `sportswear` - Athletic wear, leggings

---

## Dependencies

```
# API
fastapi
uvicorn[standard]

# Database
supabase

# Data Processing
pandas
numpy
pillow

# Testing
pytest
pytest-asyncio
httpx

# Utilities
pydantic
```

Install:
```bash
pip install -r requirements.txt
```

---

## Testing

Run all tests:
```bash
pytest tests/ -v
pytest src/recs/test_preference_weighting.py -v
pytest src/tests/ -v
```

Skip slow tests:
```bash
pytest tests/ -v -m "not slow"
```

---

## Environment Variables

```bash
# API
export HOST=0.0.0.0
export PORT=8000
export WORKERS=4

# Supabase
export SUPABASE_URL=your_supabase_url
export SUPABASE_KEY=your_supabase_key
```

---

## License

MIT
