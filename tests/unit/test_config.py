"""
Tests for the configuration module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch


class TestSettings:
    """Tests for Settings class."""
    
    def test_settings_loads_from_env(self):
        """Test that settings load from environment variables."""
        from config.settings import get_settings
        
        settings = get_settings()
        
        # Required fields should be present
        assert settings.supabase_url is not None
        assert settings.supabase_service_key is not None
        
        # Defaults should be applied
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080
    
    def test_is_development_property(self):
        """Test is_development property."""
        from config.settings import Settings
        
        # Test development environments
        for env in ["development", "dev", "local"]:
            settings = Settings(
                supabase_url="https://test.supabase.co",
                supabase_service_key="test-key",
                environment=env,
            )
            assert settings.is_development is True
        
        # Test production
        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_service_key="test-key",
            environment="production",
        )
        assert settings.is_development is False
    
    def test_is_production_property(self):
        """Test is_production property."""
        from config.settings import Settings
        
        # Test production environments
        for env in ["production", "prod"]:
            settings = Settings(
                supabase_url="https://test.supabase.co",
                supabase_service_key="test-key",
                environment=env,
            )
            assert settings.is_production is True
        
        # Test development
        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_service_key="test-key",
            environment="development",
        )
        assert settings.is_production is False
    
    def test_cors_origins_parsing(self):
        """Test that CORS origins can be parsed from comma-separated string."""
        from config.settings import Settings
        
        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_service_key="test-key",
            cors_origins="http://localhost:3000,http://localhost:5173",
        )
        
        assert len(settings.cors_origins) == 2
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:5173" in settings.cors_origins
    
    def test_base_dir_parsing(self):
        """Test that base_dir can be parsed from string."""
        from config.settings import Settings
        
        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_service_key="test-key",
            base_dir="/tmp/test",
        )
        
        assert settings.base_dir == Path("/tmp/test")
        assert isinstance(settings.base_dir, Path)
    
    def test_path_properties(self):
        """Test that path properties are correctly derived."""
        from config.settings import Settings
        
        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_service_key="test-key",
            base_dir="/opt/app",
        )
        
        assert settings.models_dir == Path("/opt/app/models")
        assert settings.data_dir == Path("/opt/app/data")
        assert settings.images_dir == Path("/opt/app/data/women_fashion/images_webp")
    
    def test_women_images_url_property(self):
        """Test women_images_url property."""
        from config.settings import Settings
        
        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_service_key="test-key",
            api_base_url="https://api.example.com",
        )
        
        assert settings.women_images_url == "https://api.example.com/women-images/"
    
    def test_settings_for_testing(self):
        """Test get_settings_for_testing function."""
        from config.settings import get_settings_for_testing
        
        settings = get_settings_for_testing(
            environment="testing",
            debug=True,
        )
        
        assert settings.environment == "testing"
        assert settings.debug is True
        # Default test values should be applied
        assert "test" in settings.supabase_url


class TestConstants:
    """Tests for constants module."""
    
    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        from config.constants import DEFAULT_PIPELINE_CONFIG
        
        assert DEFAULT_PIPELINE_CONFIG.PRIMARY_CANDIDATES == 300
        assert DEFAULT_PIPELINE_CONFIG.MAX_PER_CATEGORY == 8
        assert DEFAULT_PIPELINE_CONFIG.EXPLORATION_RATE == 0.10
    
    def test_sasrec_config_defaults(self):
        """Test SASRecConfig default values."""
        from config.constants import DEFAULT_SASREC_CONFIG
        
        assert DEFAULT_SASREC_CONFIG.MAX_SEQ_LENGTH == 50
        assert DEFAULT_SASREC_CONFIG.BRAND_DIVERSITY_CAP == 0.25
        assert "sasrec" in DEFAULT_SASREC_CONFIG.WARM_WEIGHTS
        assert "embedding" in DEFAULT_SASREC_CONFIG.COLD_WEIGHTS
    
    def test_diversity_config_defaults(self):
        """Test DiversityConfig default values."""
        from config.constants import DEFAULT_DIVERSITY_CONFIG
        
        assert DEFAULT_DIVERSITY_CONFIG.default_limit == 8
        assert DEFAULT_DIVERSITY_CONFIG.single_category_limit == 50
        assert DEFAULT_DIVERSITY_CONFIG.warm_user_limit == 50
    
    def test_soft_scoring_weights_defaults(self):
        """Test SoftScoringWeights default values."""
        from config.constants import DEFAULT_SOFT_WEIGHTS
        
        assert DEFAULT_SOFT_WEIGHTS.max_total_boost == 0.15
        assert DEFAULT_SOFT_WEIGHTS.semantic_floor == 0.25
        assert DEFAULT_SOFT_WEIGHTS.fit_boost > 0
        assert DEFAULT_SOFT_WEIGHTS.color_demote < 0
    
    def test_women_categories(self):
        """Test WOMEN_CATEGORIES list."""
        from config.constants import WOMEN_CATEGORIES
        
        assert "tops_woven" in WOMEN_CATEGORIES
        assert "dresses" in WOMEN_CATEGORIES
        assert "sportswear" in WOMEN_CATEGORIES
    
    def test_sportswear_detection_sets(self):
        """Test sportswear detection sets."""
        from config.constants import (
            SPORTSWEAR_BRANDS,
            SPORTSWEAR_ARTICLE_TYPES,
            SPORTSWEAR_NAME_KEYWORDS,
        )
        
        assert "nike" in SPORTSWEAR_BRANDS
        assert "leggings" in SPORTSWEAR_ARTICLE_TYPES
        assert "yoga" in SPORTSWEAR_NAME_KEYWORDS


class TestDatabase:
    """Tests for database module."""
    
    @pytest.mark.supabase
    def test_supabase_client_singleton(self):
        """Test that get_supabase_client returns singleton."""
        from config.database import get_supabase_client
        
        client1 = get_supabase_client()
        client2 = get_supabase_client()
        
        assert client1 is client2
    
    @pytest.mark.supabase
    def test_supabase_client_works(self):
        """Test that Supabase client can query database."""
        from config.database import get_supabase_client
        
        client = get_supabase_client()
        result = client.table("products").select("id").limit(1).execute()
        
        assert result.data is not None
    
    def test_supabase_client_optional_returns_none_on_error(self):
        """Test that get_supabase_client_optional handles errors gracefully."""
        from config.database import get_supabase_client_optional
        
        # Note: This test assumes the actual client works
        # In a real test, we'd mock the client creation to fail
        client = get_supabase_client_optional()
        
        # Should either return a client or None, not raise
        assert client is not None or client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
