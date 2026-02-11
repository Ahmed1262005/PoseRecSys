"""
Test environment setup and dependencies
"""
import pytest


def test_torch_import():
    """Test PyTorch is installed"""
    import torch
    assert torch is not None
    print(f"PyTorch version: {torch.__version__}")


def test_gpu_available():
    """Test GPU availability (informational)"""
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    # GPU recommended but not required
    assert True


def test_recbole_import():
    """Test RecBole is installed"""
    import recbole
    assert recbole is not None
    print(f"RecBole version: {recbole.__version__}")


def test_fashionclip_import():
    """Test FashionCLIP is available via transformers (used by women_search_engine.py)"""
    from transformers import CLIPProcessor, CLIPModel
    assert CLIPProcessor is not None
    assert CLIPModel is not None
    # The codebase loads 'patrickjohncyh/fashion-clip' via transformers directly


def test_faiss_import():
    """Test Faiss is installed"""
    import faiss
    assert faiss is not None
    # Check GPU support
    if hasattr(faiss, 'get_num_gpus'):
        print(f"Faiss GPU count: {faiss.get_num_gpus()}")


def test_fastapi_import():
    """Test FastAPI is installed"""
    import fastapi
    assert fastapi is not None


def test_pandas_numpy():
    """Test data processing libraries"""
    import pandas as pd
    import numpy as np
    assert pd is not None
    assert np is not None


def test_pillow_import():
    """Test PIL is installed for image processing"""
    from PIL import Image
    assert Image is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
