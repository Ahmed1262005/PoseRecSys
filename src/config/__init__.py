"""
Configuration module for the recommendation system.

This module provides centralized configuration management using pydantic-settings.
All environment variables and configuration values should be accessed through this module.

Usage:
    from config import get_settings, settings
    
    # Get settings instance (cached)
    settings = get_settings()
    
    # Access values
    supabase_url = settings.supabase_url
    is_dev = settings.is_development
"""

from config.settings import Settings, get_settings

# Convenience: create a default settings instance
# Note: This will raise if required env vars are missing
try:
    settings = get_settings()
except Exception:
    settings = None  # Allow import even if env vars not set (for testing)

__all__ = ["Settings", "get_settings", "settings"]
