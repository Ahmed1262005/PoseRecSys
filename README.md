# Fashion Personalized Feed

Production-ready personalized fashion recommendation system combining visual similarity (FashionCLIP) with collaborative filtering (RecBole).

## Features

- **Visual Similarity**: FashionCLIP embeddings for outfit compatibility
- **Collaborative Filtering**: RecBole BPR model for user preference learning
- **Hybrid Ranking**: Weighted combination of visual + CF signals
- **REST API**: FastAPI server for production deployment
- **Evaluation Suite**: Comprehensive benchmarks (AUC, FITB, NDCG)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Polyvore Dataset

```bash
mkdir -p data/polyvore
cd data/polyvore
wget https://github.com/xthan/polyvore-dataset/archive/refs/heads/master.zip
unzip master.zip
mv polyvore-dataset-master/* .
rm -rf polyvore-dataset-master master.zip
cd ../..
```

### 3. Process Data

```bash
python src/data_processing.py data/polyvore
```

### 4. Generate Embeddings

```bash
python src/embeddings.py data/polyvore data/polyvore/images
```

### 5. Train Model (Optional)

```bash
python src/train_models.py BPR data polyvore
```

### 6. Start API Server

```bash
python src/api.py
# or
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 7. Run Evaluation

```bash
python src/evaluate.py data/polyvore models/polyvore_embeddings.pkl
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/feed` | POST | Get personalized feed |
| `/similar` | POST | Get visually similar items |
| `/feedback` | POST | Record user interaction |
| `/item/{id}` | GET | Get item details |
| `/categories` | GET | List categories |

### Example: Get Feed

```bash
curl -X POST http://localhost:8000/feed \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "k": 20}'
```

### Example: Get Similar Items

```bash
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"item_id": "12345", "k": 10}'
```

## Project Structure

```
outfitTransformer/
├── src/
│   ├── data_processing.py   # Polyvore → RecBole conversion
│   ├── embeddings.py        # FashionCLIP + Faiss
│   ├── train_models.py      # RecBole training
│   ├── feed_generator.py    # Hybrid recommendation
│   ├── api.py               # FastAPI server
│   └── evaluate.py          # Benchmarking
├── tests/
│   ├── test_data_processing.py
│   ├── test_embeddings.py
│   ├── test_train_models.py
│   ├── test_feed_generator.py
│   ├── test_api.py
│   └── test_evaluate.py
├── data/
│   └── polyvore/            # Dataset files
├── models/                  # Trained models & embeddings
├── configs/                 # Configuration files
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_api.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Compatibility AUC | > 0.85 | Outfit compatibility prediction |
| FITB Accuracy | > 60% | Fill-in-the-blank task |
| NDCG@10 | > 0.40 | Retrieval quality |
| API Latency P95 | < 100ms | Response time |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDINGS_PATH` | `models/polyvore_embeddings.pkl` | Path to embeddings |
| `FAISS_PATH` | `models/polyvore_faiss_index.bin` | Path to Faiss index |
| `METADATA_PATH` | `data/polyvore/polyvore_item_metadata.json` | Item metadata |
| `RECBOLE_CHECKPOINT` | None | RecBole model checkpoint |
| `USE_GPU` | `true` | Enable GPU acceleration |
| `HOST` | `0.0.0.0` | API host |
| `PORT` | `8000` | API port |

## License

MIT
