"""
Centralized settings management using pydantic-settings.

All environment variables and configuration values are defined here.
Use get_settings() to access the singleton settings instance.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Required environment variables:
        - SUPABASE_URL: Supabase project URL
        - SUPABASE_SERVICE_KEY: Supabase service role key
    
    Optional environment variables:
        - HOST: Server host (default: 0.0.0.0)
        - PORT: Server port (default: 8080)
        - REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
        - BASE_DIR: Base directory for data files
        - ENVIRONMENT: Environment name (development, staging, production)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==========================================================================
    # Environment
    # ==========================================================================
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() in ("development", "dev", "local")
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")
    
    # ==========================================================================
    # Server Configuration
    # ==========================================================================
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")
    workers: int = Field(default=4, description="Number of uvicorn workers")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=[
            "https://ecommerce.outrove.ai",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        description="Allowed CORS origins"
    )
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("pinterest_scopes", mode="before")
    @classmethod
    def parse_pinterest_scopes(cls, v):
        if isinstance(v, str):
            parts: List[str] = []
            for chunk in v.split(","):
                parts.extend(chunk.split())
            return [scope.strip() for scope in parts if scope.strip()]
        return v
    
    # ==========================================================================
    # Supabase Configuration
    # ==========================================================================
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_key: str = Field(..., description="Supabase service role key")
    supabase_jwt_secret: str = Field(..., description="JWT secret for token verification (from Supabase dashboard)")
    
    # ==========================================================================
    # Redis Configuration
    # ==========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_enabled: bool = Field(
        default=False,
        description="Enable Redis for session state"
    )
    session_ttl_seconds: int = Field(
        default=86400,
        description="Session TTL in seconds (24 hours)"
    )
    
    # ==========================================================================
    # Path Configuration
    # ==========================================================================
    base_dir: Path = Field(
        default=Path("/home/ubuntu/recSys/outfitTransformer"),
        description="Base directory for data files"
    )
    
    @field_validator("base_dir", mode="before")
    @classmethod
    def parse_base_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def images_dir(self) -> Path:
        return self.base_dir / "data" / "women_fashion" / "images_webp"
    
    @property
    def hp_images_dir(self) -> Path:
        return self.base_dir / "HPImages"
    
    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_base_url: str = Field(
        default="http://ecommerce.api.outrove.ai:8080",
        description="Base URL for API responses"
    )
    
    @property
    def women_images_url(self) -> str:
        return f"{self.api_base_url}/women-images/"

    # ==========================================================================
    # Pinterest Integration
    # ==========================================================================
    pinterest_app_id: str = Field(default="", description="Pinterest app ID (client_id)")
    pinterest_app_secret: str = Field(default="", description="Pinterest app secret (client_secret)")
    pinterest_redirect_uri: str = Field(default="", description="Pinterest OAuth redirect URI")
    pinterest_auth_url: str = Field(
        default="https://www.pinterest.com/oauth/",
        description="Pinterest OAuth authorization URL"
    )
    pinterest_api_base_url: str = Field(
        default="https://api.pinterest.com/v5",
        description="Pinterest API base URL"
    )
    pinterest_scopes: List[str] = Field(
        default=["pins:read", "boards:read", "user_accounts:read"],
        description="Pinterest OAuth scopes"
    )
    pinterest_oauth_state_secret: str = Field(
        default="",
        description="Secret used to sign OAuth state (fallback to SUPABASE_JWT_SECRET)"
    )
    pinterest_oauth_state_ttl_seconds: int = Field(
        default=600,
        description="OAuth state TTL in seconds"
    )
    pinterest_request_timeout_seconds: int = Field(
        default=10,
        description="Timeout for Pinterest API and image fetch requests (seconds)"
    )
    pinterest_token_expiry_grace_seconds: int = Field(
        default=600,
        description="Warn when Pinterest token expires within this many seconds"
    )
    pinterest_access_token: str = Field(
        default="",
        description="Optional Pinterest access token for local/dev auto-connect"
    )
    pinterest_access_token_scope: str = Field(
        default="",
        description="Scope string for the Pinterest access token (optional)"
    )
    pinterest_access_token_expires_in: Optional[int] = Field(
        default=None,
        description="Access token lifetime in seconds (optional)"
    )
    pinterest_access_token_token_type: str = Field(
        default="bearer",
        description="Token type for Pinterest access token (optional)"
    )
    pinterest_default_max_pins: int = Field(
        default=120,
        description="Default max pins to fetch per sync"
    )
    pinterest_default_max_images: int = Field(
        default=60,
        description="Default max images to embed per sync"
    )
    pinterest_continuous_refresh: bool = Field(
        default=True,
        description="Request continuous refresh tokens when exchanging/refreshing"
    )
    
    # ==========================================================================
    # Algolia Search Configuration
    # ==========================================================================
    algolia_app_id: str = Field(default="", description="Algolia application ID")
    algolia_search_key: str = Field(default="", description="Algolia search API key")
    algolia_write_key: str = Field(default="", description="Algolia write API key")
    algolia_index_name: str = Field(default="products", description="Algolia index name")

    # ==========================================================================
    # OpenAI (LLM Query Planner)
    # ==========================================================================
    openai_api_key: str = Field(default="", description="OpenAI API key for LLM query planner")
    query_planner_model: str = Field(
        default="gpt-5-mini",
        description="OpenAI model for query planning"
    )
    query_planner_enabled: bool = Field(
        default=True,
        description="Enable LLM query planner (falls back to regex if disabled or fails)"
    )
    query_planner_timeout_seconds: float = Field(
        default=90.0,
        description="Timeout for LLM query planner call (seconds)"
    )

    # ==========================================================================
    # Multimodal Embeddings
    # ==========================================================================
    multimodal_search_enabled: bool = Field(
        default=True,
        description="Use multimodal embeddings (image+text) for semantic search. "
        "Falls back to image-only if disabled or multimodal table is empty."
    )
    multimodal_embedding_version: int = Field(
        default=1,
        description="Multimodal embedding version: 1=attributes only, 2=attributes+description"
    )

    # ==========================================================================
    # Weather / Context Scoring
    # ==========================================================================
    openweather_api_key: str = Field(
        default="",
        description="OpenWeatherMap API key for weather-based scoring (optional)"
    )

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    enable_sasrec: bool = Field(
        default=True,
        description="Enable SASRec model for ranking"
    )
    enable_exploration: bool = Field(
        default=True,
        description="Enable exploration in recommendations"
    )
    
    # ==========================================================================
    # Model Paths
    # ==========================================================================
    sasrec_model_path: Optional[str] = Field(
        default=None,
        description="Path to SASRec model checkpoint"
    )
    
    @property
    def sasrec_model_file(self) -> Path:
        if self.sasrec_model_path:
            return Path(self.sasrec_model_path)
        return self.models_dir / "SASRec-Dec-11-2025_18-20-35.pth"
    
    hp_embeddings_path: Optional[str] = Field(
        default=None,
        description="Path to HP embeddings pickle"
    )
    
    @property
    def hp_embeddings_file(self) -> Path:
        if self.hp_embeddings_path:
            return Path(self.hp_embeddings_path)
        return self.models_dir / "hp_embeddings.pkl"


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings singleton.
    
    Uses lru_cache to ensure only one instance is created.
    Settings are loaded from environment variables and .env file.
    
    Returns:
        Settings: The application settings instance
        
    Raises:
        ValidationError: If required environment variables are missing
    """
    # Try to find .env file in project root
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        os.environ.setdefault("ENV_FILE", str(env_file))
    
    return Settings(_env_file=env_file if env_file.exists() else None)


def get_settings_for_testing(**overrides) -> Settings:
    """
    Create a settings instance for testing with optional overrides.
    
    This bypasses the cache to allow different settings in tests.
    
    Args:
        **overrides: Setting values to override
        
    Returns:
        Settings: A new settings instance with overrides applied
    """
    # Set defaults for testing
    test_defaults = {
        "supabase_url": "https://test.supabase.co",
        "supabase_service_key": "test-key",
        "environment": "testing",
        "debug": True,
    }
    test_defaults.update(overrides)
    
    return Settings(**test_defaults)
