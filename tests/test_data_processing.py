"""
Tests for data processing module
"""
import os
import json
import tempfile
import pytest
from pathlib import Path

# Import module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import (
    load_polyvore_outfits,
    load_item_metadata,
    create_recbole_interactions,
    save_recbole_inter_file,
    create_recbole_item_file,
    validate_data_integrity,
    get_outfit_splits,
    get_item_categories,
)


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample Polyvore data for testing"""
    # Create train.json
    train_data = {
        "outfit_001": {
            "set_id": "outfit_001",
            "name": "Casual Look",
            "items": [
                {"item_id": "item_001", "index": 0},
                {"item_id": "item_002", "index": 1},
                {"item_id": "item_003", "index": 2},
            ]
        },
        "outfit_002": {
            "set_id": "outfit_002",
            "name": "Formal Look",
            "items": [
                {"item_id": "item_004", "index": 0},
                {"item_id": "item_005", "index": 1},
            ]
        }
    }

    valid_data = {
        "outfit_003": {
            "set_id": "outfit_003",
            "name": "Evening Look",
            "items": [
                {"item_id": "item_001", "index": 0},
                {"item_id": "item_006", "index": 1},
            ]
        }
    }

    test_data = {
        "outfit_004": {
            "set_id": "outfit_004",
            "name": "Weekend Look",
            "items": [
                {"item_id": "item_002", "index": 0},
                {"item_id": "item_007", "index": 1},
            ]
        }
    }

    # Create metadata
    metadata = {
        "item_001": {"semantic_category": "tops", "name": "White T-Shirt"},
        "item_002": {"semantic_category": "bottoms", "name": "Blue Jeans"},
        "item_003": {"semantic_category": "shoes", "name": "White Sneakers"},
        "item_004": {"semantic_category": "tops", "name": "Black Blazer"},
        "item_005": {"semantic_category": "bottoms", "name": "Dress Pants"},
        "item_006": {"semantic_category": "accessories", "name": "Gold Necklace"},
        "item_007": {"semantic_category": "outerwear", "name": "Denim Jacket"},
    }

    # Write files
    with open(tmp_path / "train.json", 'w') as f:
        json.dump(train_data, f)
    with open(tmp_path / "valid.json", 'w') as f:
        json.dump(valid_data, f)
    with open(tmp_path / "test.json", 'w') as f:
        json.dump(test_data, f)
    with open(tmp_path / "polyvore_item_metadata.json", 'w') as f:
        json.dump(metadata, f)

    return str(tmp_path)


class TestLoadPolyvoreOutfits:
    """Tests for load_polyvore_outfits function"""

    def test_loads_all_splits(self, sample_data_dir):
        """Test loading train/valid/test JSON files"""
        train, valid, test = load_polyvore_outfits(sample_data_dir)

        assert len(train) == 2
        assert len(valid) == 1
        assert len(test) == 1

    def test_outfit_structure(self, sample_data_dir):
        """Test outfit data has expected structure"""
        train, valid, test = load_polyvore_outfits(sample_data_dir)

        sample = train["outfit_001"]
        assert 'items' in sample
        assert 'set_id' in sample
        assert len(sample['items']) == 3

    def test_raises_on_missing_files(self, tmp_path):
        """Test raises FileNotFoundError for missing files"""
        with pytest.raises(FileNotFoundError):
            load_polyvore_outfits(str(tmp_path))


class TestLoadItemMetadata:
    """Tests for load_item_metadata function"""

    def test_loads_metadata(self, sample_data_dir):
        """Test loading item metadata"""
        metadata = load_item_metadata(sample_data_dir)

        assert len(metadata) == 7
        assert "item_001" in metadata

    def test_metadata_structure(self, sample_data_dir):
        """Test metadata has expected fields"""
        metadata = load_item_metadata(sample_data_dir)

        sample = metadata["item_001"]
        assert 'semantic_category' in sample
        assert sample['semantic_category'] == 'tops'


class TestCreateRecboleInteractions:
    """Tests for create_recbole_interactions function"""

    def test_creates_interactions(self, sample_data_dir):
        """Test RecBole format conversion"""
        df = create_recbole_interactions(sample_data_dir)

        assert 'user_id' in df.columns
        assert 'item_id' in df.columns
        assert 'rating' in df.columns
        assert 'timestamp' in df.columns

    def test_interaction_count(self, sample_data_dir):
        """Test correct number of interactions"""
        df = create_recbole_interactions(sample_data_dir)

        # 3 + 2 + 2 + 2 = 9 items across all outfits
        assert len(df) == 9

    def test_unique_users(self, sample_data_dir):
        """Test each outfit becomes a unique user"""
        df = create_recbole_interactions(sample_data_dir)

        # 4 outfits = 4 users
        assert df['user_id'].nunique() == 4

    def test_ratings_are_positive(self, sample_data_dir):
        """Test all ratings are 1.0 (implicit positive)"""
        df = create_recbole_interactions(sample_data_dir)

        assert (df['rating'] == 1.0).all()

    def test_timestamps_are_sequential(self, sample_data_dir):
        """Test timestamps are strictly increasing"""
        df = create_recbole_interactions(sample_data_dir)

        timestamps = df['timestamp'].tolist()
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)  # All unique


class TestSaveRecboleInterFile:
    """Tests for save_recbole_inter_file function"""

    def test_saves_file(self, sample_data_dir, tmp_path):
        """Test output file is created"""
        df = create_recbole_interactions(sample_data_dir)
        output_path = str(tmp_path / "test.inter")

        save_recbole_inter_file(df, output_path)

        assert os.path.exists(output_path)

    def test_file_format(self, sample_data_dir, tmp_path):
        """Test output file has correct header format"""
        df = create_recbole_interactions(sample_data_dir)
        output_path = str(tmp_path / "test.inter")

        save_recbole_inter_file(df, output_path)

        with open(output_path) as f:
            header = f.readline().strip()

        assert 'user_id:token' in header
        assert 'item_id:token' in header
        assert 'rating:float' in header
        assert 'timestamp:float' in header

    def test_file_content(self, sample_data_dir, tmp_path):
        """Test output file has correct data rows"""
        df = create_recbole_interactions(sample_data_dir)
        output_path = str(tmp_path / "test.inter")

        save_recbole_inter_file(df, output_path)

        with open(output_path) as f:
            lines = f.readlines()

        # Header + 9 data rows
        assert len(lines) == 10


class TestCreateRecboleItemFile:
    """Tests for create_recbole_item_file function"""

    def test_creates_item_file(self, sample_data_dir, tmp_path):
        """Test item metadata file is created"""
        output_path = str(tmp_path / "test.item")

        create_recbole_item_file(sample_data_dir, output_path)

        assert os.path.exists(output_path)

    def test_item_file_format(self, sample_data_dir, tmp_path):
        """Test item file has correct header"""
        output_path = str(tmp_path / "test.item")

        create_recbole_item_file(sample_data_dir, output_path)

        with open(output_path) as f:
            header = f.readline().strip()

        assert 'item_id:token' in header
        assert 'category:token' in header


class TestValidateDataIntegrity:
    """Tests for validate_data_integrity function"""

    def test_validates_complete_data(self, sample_data_dir):
        """Test validation with complete data"""
        result = validate_data_integrity(sample_data_dir)

        assert result['total_items'] == 7
        assert result['valid_items'] == 7
        assert result['missing_metadata'] == 0
        assert result['total_outfits'] == 4

    def test_detects_missing_metadata(self, sample_data_dir):
        """Test detection of items without metadata"""
        # Add outfit with unknown item
        with open(os.path.join(sample_data_dir, "train.json")) as f:
            train_data = json.load(f)

        train_data["outfit_005"] = {
            "set_id": "outfit_005",
            "items": [{"item_id": "unknown_item", "index": 0}]
        }

        with open(os.path.join(sample_data_dir, "train.json"), 'w') as f:
            json.dump(train_data, f)

        result = validate_data_integrity(sample_data_dir)

        assert result['missing_metadata'] == 1


class TestGetOutfitSplits:
    """Tests for get_outfit_splits function"""

    def test_returns_splits(self, sample_data_dir):
        """Test returns correct split counts"""
        splits = get_outfit_splits(sample_data_dir)

        assert 'train' in splits
        assert 'valid' in splits
        assert 'test' in splits
        assert len(splits['train']) == 2
        assert len(splits['valid']) == 1
        assert len(splits['test']) == 1


class TestGetItemCategories:
    """Tests for get_item_categories function"""

    def test_returns_categories(self, sample_data_dir):
        """Test returns category mapping"""
        categories = get_item_categories(sample_data_dir)

        assert len(categories) == 7
        assert categories['item_001'] == 'tops'
        assert categories['item_003'] == 'shoes'


# Integration test with real data (skip if data not available)
@pytest.mark.skipif(
    not os.path.exists("data/polyvore/train.json"),
    reason="Real Polyvore data not available"
)
class TestRealPolyvoreData:
    """Integration tests with actual Polyvore dataset"""

    def test_load_real_outfits(self):
        """Test loading real Polyvore outfits"""
        train, valid, test = load_polyvore_outfits("data/polyvore")

        assert len(train) > 0
        assert len(valid) > 0
        assert len(test) > 0

    def test_load_real_metadata(self):
        """Test loading real item metadata"""
        metadata = load_item_metadata("data/polyvore")

        assert len(metadata) > 100000  # ~368K items expected

    def test_create_real_interactions(self):
        """Test creating interactions from real data"""
        df = create_recbole_interactions("data/polyvore")

        assert len(df) > 100000
        assert df['user_id'].nunique() > 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
