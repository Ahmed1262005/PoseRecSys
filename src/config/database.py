"""
Database client singletons.

This module provides singleton instances for database connections,
ensuring efficient resource usage across the application.
"""

from functools import lru_cache
from typing import Optional

from supabase import Client, create_client

from config.settings import get_settings


class SupabaseClientError(Exception):
    """Raised when Supabase client cannot be created."""
    pass


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Get the singleton Supabase client instance.
    
    Uses lru_cache to ensure only one client is created and reused.
    
    Returns:
        Client: The Supabase client instance
        
    Raises:
        SupabaseClientError: If client cannot be created
    """
    try:
        settings = get_settings()
        return create_client(settings.supabase_url, settings.supabase_service_key)
    except Exception as e:
        raise SupabaseClientError(f"Failed to create Supabase client: {e}") from e


def get_supabase_client_optional() -> Optional[Client]:
    """
    Get the Supabase client, returning None if it cannot be created.
    
    Useful for graceful degradation when Supabase is not configured.
    
    Returns:
        Optional[Client]: The Supabase client or None
    """
    try:
        return get_supabase_client()
    except SupabaseClientError:
        return None


# For dependency injection in FastAPI
def get_db() -> Client:
    """
    FastAPI dependency for getting the Supabase client.
    
    Usage:
        @app.get("/items")
        async def get_items(db: Client = Depends(get_db)):
            ...
    """
    return get_supabase_client()


# Type alias for cleaner type hints
SupabaseClient = Client
