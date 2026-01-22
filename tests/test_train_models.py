"""
Tests for RecBole model training module
"""
import os
import pytest
import tempfile
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_models import (
    get_bpr_config,
    get_lightgcn_config,
    train_model,
    find_latest_checkpoint,
)


class TestModelConfigs:
    """Tests for model configuration functions"""

    def test_bpr_config_valid(self):
        """Test BPR configuration has required fields"""
        config = get_bpr_config()

        assert config['dataset'] == 'polyvore'
        assert config['epochs'] > 0
        assert 'embedding_size' in config
        assert 'learning_rate' in config
        assert 'USER_ID_FIELD' in config
        assert 'ITEM_ID_FIELD' in config

    def test_lightgcn_config_valid(self):
        """Test LightGCN configuration has required fields"""
        config = get_lightgcn_config()

        assert config['dataset'] == 'polyvore'
        assert 'n_layers' in config
        assert 'reg_weight' in config

    def test_config_custom_data_path(self):
        """Test config with custom data path"""
        config = get_bpr_config(data_path="custom/path", dataset="custom_dataset")

        assert config['data_path'] == "custom/path"
        assert config['dataset'] == "custom_dataset"

    def test_config_eval_args(self):
        """Test evaluation arguments are set"""
        config = get_bpr_config()

        assert 'eval_args' in config
        assert 'split' in config['eval_args']
        assert 'group_by' in config['eval_args']

    def test_config_metrics(self):
        """Test metrics are configured"""
        config = get_bpr_config()

        assert 'metrics' in config
        assert 'topk' in config
        assert 'valid_metric' in config
        assert 10 in config['topk']


@pytest.fixture
def sample_recbole_data(tmp_path):
    """Create sample data in RecBole format for testing"""
    # Create dataset directory
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    # Create .inter file
    inter_file = dataset_dir / "test_dataset.inter"
    with open(inter_file, 'w') as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        # Generate sample interactions
        timestamp = 1000000
        for user_id in range(100):
            for item_id in range(5):  # 5 interactions per user
                f.write(f"user_{user_id}\titem_{user_id * 10 + item_id}\t1.0\t{timestamp}\n")
                timestamp += 1

    return str(tmp_path), "test_dataset"


class TestModelTraining:
    """Tests for model training"""

    @pytest.mark.slow
    def test_data_loads_in_recbole(self, sample_recbole_data):
        """Test RecBole can load our data format"""
        from recbole.config import Config
        from recbole.data import create_dataset

        data_path, dataset = sample_recbole_data
        config_dict = get_bpr_config(data_path, dataset)
        config_dict['epochs'] = 1

        config = Config(model='BPR', config_dict=config_dict)
        recbole_dataset = create_dataset(config)

        assert recbole_dataset.item_num > 0
        assert recbole_dataset.user_num > 0

    @pytest.mark.slow
    def test_model_trains_without_error(self, sample_recbole_data):
        """Test model trains without errors"""
        data_path, dataset = sample_recbole_data
        config = get_bpr_config(data_path, dataset)
        config['epochs'] = 1
        config['train_batch_size'] = 256
        config['stopping_step'] = 1

        result = train_model('BPR', config)

        assert 'best_valid_score' in result
        assert result['best_valid_score'] >= 0

    @pytest.mark.slow
    def test_model_checkpoint_saves(self, sample_recbole_data, tmp_path):
        """Test model checkpoint is saved"""
        data_path, dataset = sample_recbole_data
        config = get_bpr_config(data_path, dataset)
        config['epochs'] = 1
        config['checkpoint_dir'] = str(tmp_path)
        config['stopping_step'] = 1

        train_model('BPR', config)

        checkpoints = list(tmp_path.glob("*.pth"))
        assert len(checkpoints) > 0

    @pytest.mark.slow
    def test_training_produces_valid_metrics(self, sample_recbole_data):
        """Test training produces valid metrics"""
        data_path, dataset = sample_recbole_data
        config = get_bpr_config(data_path, dataset)
        config['epochs'] = 2
        config['stopping_step'] = 1

        result = train_model('BPR', config)

        # Check metrics are valid
        assert 0 <= result['best_valid_score'] <= 1
        assert 'test_result' in result or 'best_valid_result' in result


class TestCheckpointManagement:
    """Tests for checkpoint management"""

    def test_find_latest_checkpoint_none(self, tmp_path):
        """Test returns None when no checkpoints exist"""
        result = find_latest_checkpoint("BPR", str(tmp_path))
        assert result is None

    def test_find_latest_checkpoint_exists(self, tmp_path):
        """Test finds checkpoint when it exists"""
        import time

        # Create fake checkpoints
        (tmp_path / "BPR-2024-01-01.pth").touch()
        time.sleep(0.1)
        (tmp_path / "BPR-2024-01-02.pth").touch()

        result = find_latest_checkpoint("BPR", str(tmp_path))

        assert result is not None
        assert "BPR-2024-01-02.pth" in result

    def test_find_checkpoint_model_specific(self, tmp_path):
        """Test finds checkpoint for specific model"""
        (tmp_path / "BPR-latest.pth").touch()
        (tmp_path / "LightGCN-latest.pth").touch()

        result = find_latest_checkpoint("LightGCN", str(tmp_path))

        assert result is not None
        assert "LightGCN" in result


# Integration tests with real data
@pytest.mark.skipif(
    not os.path.exists("data/polyvore/polyvore.inter"),
    reason="Real Polyvore data not available"
)
class TestRealPolyvoreTraining:
    """Integration tests with actual Polyvore dataset"""

    @pytest.mark.slow
    def test_load_real_polyvore_data(self):
        """Test loading real Polyvore data in RecBole"""
        from recbole.config import Config
        from recbole.data import create_dataset

        config_dict = get_bpr_config()
        config = Config(model='BPR', config_dict=config_dict)
        dataset = create_dataset(config)

        assert dataset.user_num > 10000
        assert dataset.item_num > 100000

    @pytest.mark.slow
    def test_train_real_bpr_model(self):
        """Test training BPR on real data (quick run)"""
        config = get_bpr_config()
        config['epochs'] = 1
        config['train_batch_size'] = 2048
        config['stopping_step'] = 1

        result = train_model('BPR', config)

        assert result['best_valid_score'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
